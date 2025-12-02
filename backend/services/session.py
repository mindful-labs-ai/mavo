from config import env
from persistence.sessions import update_session_status
from persistence.transcribes import create_transcribe_row
from services.stt import (
    generate_progress_note,
    parse_stt_response,
    run_stt_from_url,
)
from util.s3 import issue_presigned_url
import httpx
from controllers.views.session import SessionRequest
from fastapi import BackgroundTasks, HTTPException, status

from util.logger import get_logger

logger = get_logger(__name__)


async def update_session_progress(session_id: str, percentage: int, step: str) -> None:
    """
    sessions 테이블에서 progress_percentage와 current_step을 업데이트.
    """
    supabase_url = env.SUPABASE_URL
    service_key = env.SUPABASE_SERVICE_KEY
    if not supabase_url or not service_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase 환경변수가 설정되지 않았습니다.",
        )

    # 기본 검증
    if percentage < 0:
        percentage = 0
    if percentage > 100:
        percentage = 100

    update_url = f"{supabase_url.rstrip('/')}/rest/v1/sessions?id=eq.{session_id}"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    body = {
        "progress_percentage": percentage,
        "current_step": step,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.patch(update_url, headers=headers, json=body)

    if resp.status_code not in (200, 204):
        logger.error(
            f"Session progress update failed: status={resp.status_code} body={resp.text}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="세션 진행률 업데이트에 실패했습니다.",
        )


async def process_session_pipeline(session_id: str, payload: SessionRequest) -> None:
    """
    세션 처리 전체 파이프라인을 백그라운드에서 실행.
    """
    try:
        await update_session_progress(
            session_id, percentage=10, step="오디오 파일 처리 중..."
        )

        audio_url = await issue_presigned_url(payload.s3_key)
        await update_session_progress(
            session_id, percentage=30, step="축어록 생성 중..."
        )

        raw_stt_output = await run_stt_from_url(
            stt_model=payload.stt_model,
            audio_url=audio_url,
            session_id=session_id,
            background_tasks=BackgroundTasks(),
        )

        await update_session_progress(
            session_id, percentage=50, step="축어록 변환 중..."
        )
        transcript_json = parse_stt_response(
            raw_output=raw_stt_output, stt_model=payload.stt_model
        )

        await create_transcribe_row(
            session_id=session_id,
            user_id=payload.user_id,
            stt_model=payload.stt_model,
            transcript_json=transcript_json,
        )

        await update_session_progress(
            session_id, percentage=80, step="progress note 생성 중..."
        )

        progress_note = await generate_progress_note(
            session_id=session_id,
            user_id=payload.user_id,
            transcript_json=transcript_json,
            template_id=payload.template_id or 1,
        )

        await update_session_progress(
            session_id, percentage=90, step="마무리 작업 진행 중..."
        )

        await update_session_status(session_id, status_value="succeeded")

    except Exception as e:
        await update_session_status(
            session_id,
            status_value="failed",
            error_message=f"Failed to process session: {str(e)}",
        )
        logger.error(f"Error in v2 session flow for {session_id}: {e}")
