import os
import asyncio
import json
import tempfile
from datetime import datetime
import re
from typing import Any, Dict, Optional
import httpx
from fastapi import BackgroundTasks, HTTPException, status

from pydub import AudioSegment
from util.logger import get_logger
from logic.models import AnalysisWork, AudioStatus, analysis_jobs
from logic.voice_analysis.process import process_audio_file

logger = get_logger(__name__)


async def run_stt_from_url(
    stt_model: str,
    audio_url: str,
    session_id: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> str:
    """
    stt_model에 따라 STT 요청을 수행하고, 원본 응답을 문자열로 반환.
    """
    model_key = stt_model.lower()
    if model_key == "whisper":
        return await run_stt_with_whisper(
            audio_url, session_id=session_id, background_tasks=background_tasks
        )
    elif model_key in ("gemini-3", "gemini-3-pro-preview", "gemini3"):
        return await run_stt_with_gemini(audio_url)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 stt_model: {stt_model}",
        )


async def run_stt_with_whisper(
    audio_url: str,
    session_id: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> str:
    """
    기존 /api/v1 Whisper 파이프라인(process_audio_file) 재사용.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        download_resp = await client.get(audio_url)
        if download_resp.status_code != 200:
            logger.error(
                f"오디오 다운로드 실패(whisper): status={download_resp.status_code} body={download_resp.text[:200]}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="오디오 다운로드에 실패했습니다.",
            )

    suffix = os.path.splitext(audio_url.split("?")[0])[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(download_resp.content)
        tmp.flush()
        local_path = tmp.name

    audio_uuid = session_id or os.path.basename(local_path)

    job = AnalysisWork(
        id=audio_uuid,
        filename=os.path.basename(local_path),
        total_chunks=1,
        status=AudioStatus.PENDING,
    )
    job.file_path = local_path
    job.audio_uuid = audio_uuid
    job.options.update(
        {
            "diarization_method": "stt_apigpt_diar_mlpyan2",
            "is_limit_time": False,
            "is_merge_segments": True,
        }
    )
    analysis_jobs[audio_uuid] = job

    bg = background_tasks or BackgroundTasks()
    await process_audio_file(job, bg)

    result_dict = job.result.dict() if job.result else {}

    try:
        os.unlink(local_path)
    except OSError:
        pass

    return json.dumps(result_dict, ensure_ascii=False)


async def run_stt_with_gemini(audio_url: str) -> str:
    """
    Google Gemini-3 모델로 STT 요청. presigned URL의 오디오를 파일 업로드 후 전송.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY가 설정되지 않았습니다.",
        )

    try:
        from google import genai
    except ImportError:
        logger.error(
            "google-genai 패키지가 설치되어 있지 않습니다. requirements에 추가하세요."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai가 설치되어 있지 않습니다.",
        )

    async with httpx.AsyncClient(timeout=120) as client:
        download_resp = await client.get(audio_url)
        if download_resp.status_code != 200:
            logger.error(
                f"오디오 다운로드 실패: status={download_resp.status_code} body={download_resp.text[:200]}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="오디오 다운로드에 실패했습니다.",
            )

    def _call_gemini_from_file(path: str) -> str:
        client = genai.Client(api_key=api_key)
        audioFile = client.files.upload(file=path)
        prompt = """
        당신은 심리상담 정밀 분석을 위한 데이터 생성기입니다.
제공된 오디오/텍스트를 분석하여 아래 규칙에 맞춰 **데이터 포맷만** 출력하십시오.

## **I. 핵심 작성 원칙**
1. **Full-Verbatim:** 처음부터 끝까지 들리는 모든 소리를 누락 없이 받아 적으십시오.
2. **No Chatter:** 서론, 결론 없이 오직 데이터 라인만 출력하십시오.

## **II. 데이터 코드 매핑 (Sequential Speaker Mode)**
상담자(C)를 제외한 모든 화자는 등장 순서대로 P1, P2...로 번호를 매깁니다.

1. **구조:** '화자코드||내용'
   - **Counselor (상담자):** 'C'
   - **Participants (참여자):** 상담자 외의 목소리는 구분되는 순서대로 'P1', 'P2', 'P3'... 코드를 부여하십시오.
     - (예: 내담자A -> P1, 내담자B -> P2, 내담자모친 -> P3)
   
   - **Example:** 'C||안녕하세요.'
   - **Example:** 'P1||안녕하세요.'
   - **Example:** 'P2||저도 왔습니다.'

2. **비언어/행동 태그 (Short Tags)**
   - 텍스트 내 구분자는 충돌 방지를 위해 '{}'와 '%'를 사용합니다.
   - 침묵/멈춤: '{%S%}' (Silence)
   - 행동/비언어: '{%A%행동내용}' (Action - 예: '{%A%한숨%}', '{%A%울음%}')
   - 감정/강조: '{%E%강조내용}' (Emotion/Emphasis)
   - 개입/겹침: '{%O%}' (Overlap)

## **III. 출력 예시 (Multi-Speaker Shot)**
C||오늘 두 분, 사이가 좀 어떠신가요?
P1||{%A%한숨%} 저는 그냥 답답해요. {%S%} 말이 안 통하거든요.
P2||{%O%} {%E%말을 왜 그렇게 해?} 내가 언제 말을 안 들었어?

## **IV. 수행 명령**
위 포맷에 맞춰 지금 즉시 변환을 시작하십시오.
""".strip()
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt, audioFile],
        )
        return response.text or str(response)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(download_resp.content)
        tmp.flush()
        speed_path = None
        try:
            speed_path = await asyncio.to_thread(_speedup_audio_file, tmp.name, 2.0)
            path_to_use = speed_path
        except Exception as e:
            logger.warning(f"오디오 2배속 변환 실패, 원본 사용: {e}")
            path_to_use = tmp.name

        try:
            raw_resp = await asyncio.to_thread(_call_gemini_from_file, path_to_use)
            return raw_resp
        except Exception as e:
            logger.error(f"Gemini STT 호출 실패: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini STT 호출에 실패했습니다.",
            )
        finally:
            if speed_path and os.path.exists(speed_path):
                try:
                    os.unlink(speed_path)
                except OSError:
                    pass


def _speedup_audio_file(input_path: str, playback_speed: float = 2.0) -> str:
    """
    오디오 파일을 playback_speed 배속으로 변환하여 임시 WAV 파일 경로를 반환.
    """
    audio = AudioSegment.from_file(input_path)
    sped = audio._spawn(
        audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * playback_speed)}
    )
    sped = sped.set_frame_rate(audio.frame_rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
        sped.export(out.name, format="wav")
        return out.name


def parse_stt_response(raw_output: str, stt_model: str) -> Dict[str, Any]:
    """
    STT 결과를 통일된 형태로 변환.
    - segments: [{start, end, text, speaker}]
    - language: 추정값
    - raw_output: 원본
    """
    model_key = stt_model.lower()
    if model_key in ("gemini-3", "gemini-3-pro-preview", "gemini3"):
        return parse_gemini_raw_output(raw_output, stt_model)
    elif model_key == "whisper":
        return parse_whisper_raw_output(raw_output, stt_model)
    else:
        return {
            "language": "ko",
            "segments": [],
            "raw_output": raw_output,
            "stt_model": stt_model,
        }


def parse_gemini_raw_output(raw_output: str, stt_model: str) -> Dict[str, Any]:
    """
    Gemini-3가 반환한 '%T%HH:MM:SS||Speaker||Text' 포맷을 세그먼트 리스트로 파싱하되,
    타임스탬프는 사용하지 않고 제거한다.
    """
    text = raw_output.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1]

    segments = []
    pattern = re.compile(r"^(?:%T%\d{2}:\d{2}:\d{2}\|\|)?(C|P\d+)\|\|(.*)$")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue
        speaker, content = m.groups()
        segments.append(
            {
                "start": None,
                "end": None,
                "speaker": speaker,
                "text": content.strip(),
            }
        )

    transcript_text = "\n".join(seg["text"] for seg in segments)
    return {
        "language": "ko",
        "segments": segments,
        "text": transcript_text,
        "raw_output": raw_output,
        "stt_model": stt_model,
    }


def parse_whisper_raw_output(raw_output: str, stt_model: str) -> Dict[str, Any]:
    """
    Whisper 파이프라인 결과(JSON 문자열)를 파싱.
    """
    try:
        data = json.loads(raw_output)
    except Exception:
        data = {}

    segments = data.get("segments") or []
    text = data.get("text") or " ".join(seg.get("text", "") for seg in segments)

    # end가 없으면 None 유지
    return {
        "language": data.get("language", "ko"),
        "segments": segments,
        "text": text,
        "raw_output": raw_output,
        "stt_model": stt_model,
    }


async def generate_progress_note(
    session_id: str,
    user_id: int,
    transcript_json: Dict[str, Any],
    template_id: int = 1,
) -> Dict[str, Any]:
    """
    Gemini를 사용해 progress note를 생성하고 progress_notes 테이블에 저장.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase 환경변수가 설정되지 않았습니다.",
        )

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }
    template_url = f"{supabase_url.rstrip('/')}/rest/v1/templates?id=eq.{template_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        tmpl_resp = await client.get(template_url, headers=headers)
    if tmpl_resp.status_code != 200:
        logger.error(
            f"Template fetch failed: status={tmpl_resp.status_code} body={tmpl_resp.text}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="템플릿 조회에 실패했습니다.",
        )
    tmpl_data = tmpl_resp.json()
    template = (
        tmpl_data[0]
        if isinstance(tmpl_data, list) and tmpl_data
        else tmpl_data if isinstance(tmpl_data, dict) else None
    )
    if not template or "prompt" not in template:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="템플릿 데이터가 없습니다.",
        )

    # 2) 프롬프트 구성
    transcript_raw_output = transcript_json.get("raw_output") or ""
    system_prompt = "당신은 전문 심리상담 기록 작성자입니다. 제공된 템플릿과 상담 내용을 바탕으로 정확하고 전문적인 상담 기록을 작성해주세요."
    user_prompt = f"""
{template.get('prompt', '')}

[상담 내용]
{transcript_raw_output}

위 상담 내용을 바탕으로 상담 기록을 작성해주세요.
""".strip()

    # 3) Gemini 호출
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY가 설정되지 않았습니다.",
        )
    try:
        from google import genai
    except ImportError:
        logger.error(
            "google-genai 패키지가 설치되어 있지 않습니다. requirements에 추가하세요."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="google-genai가 설치되어 있지 않습니다.",
        )

    client = genai.Client(api_key=api_key)
    model = "gemini-3-pro-preview" if template_id == 1 else "gemini-flash-latest"
    try:
        response = client.models.generate_content(
            model=model,
            contents=[system_prompt, user_prompt],
        )
        summary = response.text or str(response)
    except Exception as e:
        logger.error(f"Gemini progress note 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini progress note 생성에 실패했습니다.",
        )

    # 4) progress_notes 저장
    note_body = {
        "session_id": session_id,
        "user_id": user_id,
        "title": template.get("title", "progress_note"),
        "template_id": template_id,
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
    }
    insert_url = f"{supabase_url.rstrip('/')}/rest/v1/progress_notes"
    async with httpx.AsyncClient(timeout=15) as client:
        ins_resp = await client.post(
            insert_url,
            headers={**headers, "Prefer": "return=representation"},
            json=note_body,
        )
    if ins_resp.status_code not in (200, 201):
        logger.error(
            f"Progress note insert failed: status={ins_resp.status_code} body={ins_resp.text}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="progress_note 저장에 실패했습니다.",
        )

    ins_data = ins_resp.json()
    note_row = (
        ins_data[0]
        if isinstance(ins_data, list) and ins_data
        else ins_data if isinstance(ins_data, dict) else {}
    )
    note_id = note_row.get("id")

    return {
        "progress_note_id": note_id,
        "summary": summary,
        "template_id": template_id,
    }
