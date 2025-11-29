import asyncio
import json
import os
import tempfile
import whisper
import torch
import numpy as np
from pydub import AudioSegment
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Any, Tuple
from fastapi import BackgroundTasks
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback
import webrtcvad
import wave
import array
import struct
from pyannote.audio import Pipeline
import torchaudio
import copy
import re
import matplotlib.pyplot as plt
import librosa
import librosa.display
import uuid
from backend.logic.stt_utils import get_improved_lines_with_ts
from backend.logic.models import AnalysisWork, AudioStatus, analysis_jobs, TranscriptionResult, Segment, Speaker
import backend.config as config
from backend.logic.voice_analysis.ai_utils import get_openai_client
from backend.util.logger import get_logger
from fuzzywuzzy import fuzz

# Get logger
logger = get_logger(__name__)



def custom_json_encoder(obj):
    """Custom JSON encoder that ignores non-serializable objects"""
    try:
        # Try to get __dict__ first
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}  # Filter out private attributes
        # For other types, just ignore them
        return "<non_serializable_data>"
    except Exception:
        return "<non_serializable_data>"
    


def save_analysis_work_json(analysis_job: AnalysisWork) -> None:
    """
    Save the AnalysisWork state as JSON.
    
    Args:
        analysis_job: The AnalysisWork instance to save
    """
    try:
        # Get audio UUID from file path
        audio_uuid = os.path.basename(os.path.dirname(analysis_job.file_path))
        
        # Create uploads directory if it doesn't exist
        save_dir = Path(config.UPLOADS_DIR) / audio_uuid
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the save path
        save_path = save_dir / f"id[{audio_uuid}]_analysiswork.json"
        
        # Convert AnalysisWork to dict, handling datetime objects
        analysis_dict = {
            "id": analysis_job.id,
            "filename": analysis_job.filename,
            "file_path": analysis_job.file_path,
            "status": analysis_job.status,
            "error": analysis_job.error,
            "options": analysis_job.options,
            "total_chunks": analysis_job.total_chunks,
            "chunks": {str(k): v.dict() for k, v in analysis_job.chunks.items()},
            "steps": analysis_job.steps,
            "created_at": analysis_job.created_at.isoformat(),
            "updated_at": analysis_job.updated_at.isoformat(),
            "result": analysis_job.result.dict() if analysis_job.result else None
        }
        
        # Save as JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved analysis work state to {save_path}")
        
    except Exception as e:
        err_msg = f"ERROR in save_analysis_work_json: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)



def get_cleared_diarization_segments(diarization_segments):
    """
    Ensures no overlapping segments remain by splitting segments at overlap points
    and keeping the shortest segment in overlapping regions.
    
    Algorithm:
    1. Find all unique time points (starts and ends)
    2. Create sub-segments between each pair of consecutive time points
    3. For each sub-segment, if multiple speakers overlap, keep only the one
       with the shortest original segment duration
    
    Args:
        diarization_segments: List of segments with 'start', 'end', and 'speaker' keys
        
    Returns:
        List of non-overlapping segments
    """
    if not diarization_segments:
        return []
    
    # Sort segments by start time and add duration
    segments = []
    for seg in diarization_segments:
        duration = seg['end'] - seg['start']
        if duration > 0:  # Only include valid segments
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'duration': duration,
                'original_duration': duration  # Keep track of original duration
            })
    
    # Get all unique time points
    time_points = set()
    for seg in segments:
        time_points.add(seg['start'])
        time_points.add(seg['end'])
    time_points = sorted(list(time_points))
    
    # Create sub-segments between consecutive time points
    cleared_segments = []
    for i in range(len(time_points) - 1):
        sub_start = time_points[i]
        sub_end = time_points[i + 1]
        
        # Find all segments that overlap with this time window
        overlapping = []
        for seg in segments:
            if seg['start'] <= sub_start and seg['end'] >= sub_end:
                overlapping.append(seg)
        
        if overlapping:
            # If there are overlapping segments, keep the one with shortest original duration
            shortest = min(overlapping, key=lambda x: x['original_duration'])
            cleared_segments.append({
                'start': sub_start,
                'end': sub_end,
                'speaker': shortest['speaker'],
                'duration': sub_end - sub_start
            })
    
    # Merge consecutive segments with the same speaker
    merged = []
    current = None
    
    for seg in cleared_segments:
        if current is None:
            current = seg.copy()
        elif current['speaker'] == seg['speaker'] and current['end'] == seg['start']:
            # Merge if same speaker and consecutive
            current['end'] = seg['end']
            current['duration'] = current['end'] - current['start']
        else:
            merged.append(current)
            current = seg.copy()
    
    if current is not None:
        merged.append(current)
    
    return merged



async def identify_speakers_from_conversation(trans_segs_with_ts, unique_speakers):
    """
    대화 내용을 분석하여 상담사와 내담자를 식별합니다.
    
    Args:
        trans_segs_with_ts: 시간 정보가 포함된 전사 세그먼트
        unique_speakers: 고유한 화자 ID 목록
    
    Returns:
        tuple: (counselor_ids, client_ids) - 상담사와 내담자로 식별된 ID 목록
    """
    # 각 화자별 대화 내용 수집
    speaker_texts = {}
    for seg in trans_segs_with_ts:
        if seg["speaker"] not in speaker_texts:
            speaker_texts[seg["speaker"]] = []
        speaker_texts[seg["speaker"]].append(seg["text"])
    
    # 텍스트가 충분히 긴 화자만 분석 대상으로 선정
    valid_speakers = {
        speaker: texts for speaker, texts in speaker_texts.items() 
        if speaker >= 0 and texts and "".join(texts).strip()
    }
    
    if not valid_speakers:
        logger.warning("No valid speakers found with text content")
        return [0] if 0 in unique_speakers else [unique_speakers[0] if unique_speakers else 0], \
               [s for s in unique_speakers if s != 0 and s != (unique_speakers[0] if unique_speakers else 0)]
    
    # ChatGPT에 화자별 특징 분석 요청
    speaker_analysis_prompt = """
    아래는 상담 대화에서 각 화자가 말한 내용입니다. 각 화자가 상담사인지 내담자인지 분석해주세요.
    
    상담사의 특징:
    - 전문적인 용어 사용
    - 반영적 경청, 개방형 질문
    - 공감적이고 구조화된 대화 방식
    - 내담자의 말에 대한 요약과 명확화
    - 상담 초기와 마무리 과정 주도
    
    내담자의 특징:
    - 개인적 경험과 감정 표현
    - 도움을 요청하는 어투
    - 불확실성이나 고민 표현
    - 자신의 문제에 대한 설명
    
    각 화자에 대해 '상담사', '내담자', 또는 '불확실'로 분류하고, 그 이유를 간략히 설명해주세요.
    정확한 JSON 형식으로 응답해주세요.
    """

    # JSON 스키마 정의
    json_schema = {
        "type": "object",
        "properties": {
            "speaker_roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker_id": {"type": "integer"},
                        "role": {"type": "string", "enum": ["counselor", "client"]},
                        "confidence_norm": {"type": "number"},
                        "reason": {"type": "string"}
                    },
                    "required": ["speaker_id", "role", "confidence_norm", "reason"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["speaker_roles"],
        "additionalProperties": False
    }
    
    speakers_content = ""
    for speaker_id, texts in valid_speakers.items():
        speakers_content += f"\n화자 {speaker_id} 발언:\n"
        # 각 화자당 최대 10개의 발언만 포함 (너무 길지 않게)
        for i, text in enumerate(texts[:10]):
            speakers_content += f"- {text}\n"
        if len(texts) > 10:
            speakers_content += f"... 외 {len(texts)-10}개 발언\n"
    print("speakers_content", speakers_content)
    
    messages = [
        {"role": "system", "content": speaker_analysis_prompt},
        {"role": "user", "content": speakers_content}
    ]
    print("speaker_analysis_prompt", speaker_analysis_prompt)
    print("messages", messages)

    try:
        response = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema
                }
            }
        )
        
        analysis = json.loads(response.choices[0].message.content)
        print("identify speaker response", analysis)
        
        # 결과 정렬 (신뢰도 기준)
        speaker_roles = sorted(
            analysis["speaker_roles"], 
            key=lambda x: (x["role"], -x["confidence_norm"])
        )
        
        # 상담사와 내담자 ID 추출
        # counselor_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "counselor" and s["confidence_norm"] > 0.6]
        # client_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "client" and s["confidence_norm"] > 0.6]
        # uncertain_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "uncertain" or s["confidence_norm"] <= 0.6]
        counselor_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "counselor"]
        client_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "client"]
        uncertain_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "uncertain"]
        
        # 결과 후처리
        if not counselor_ids and client_ids:
            # 상담사가 없으면 불확실한 화자 중 첫 번째를 상담사로 간주
            if uncertain_ids:
                counselor_ids = [uncertain_ids[0]]
                uncertain_ids.pop(0)
            # 여전히 없으면 첫 번째 화자를 상담사로 간주
            elif unique_speakers:
                counselor_ids = [unique_speakers[0]]
        
        # 나머지 불확실한 화자는 내담자로 간주
        client_ids.extend(uncertain_ids)
        
        # 중복 제거
        counselor_ids = list(set(counselor_ids))
        client_ids = list(set(client_ids) - set(counselor_ids))
        
        # 로깅
        logger.info(f"Identified counselor IDs: {counselor_ids}")
        logger.info(f"Identified client IDs: {client_ids}")
        
        return counselor_ids, client_ids
        
    except Exception as e:
        logger.error(f"Error in speaker identification: {e}, {traceback.format_exc()}")
        # 오류 발생 시 기본값 반환 (첫 번째 화자를 상담사로 간주)
        return [0] if 0 in unique_speakers else [unique_speakers[0] if unique_speakers else 0], \
               [s for s in unique_speakers if s != 0 and s != (unique_speakers[0] if unique_speakers else 0)]


def save_transcription_result_json(audio_uuid: str, transcription_result, suffix: str = "transcript") -> None:
    """
    Save the final transcription result as a JSON file.
    
    Args:
        audio_uuid: UUID of the audio file
        transcription_result: TranscriptionResult object to save
    """
    try:
        # Create the uploads directory if it doesn't exist
        uploads_dir = Path(config.UPLOADS_DIR) / audio_uuid
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the transcript file path
        transcript_path = uploads_dir / f"id[{audio_uuid}]_{suffix}.json"
        
        # Convert TranscriptionResult to dictionary
        result_dict = {
            "text": transcription_result.text,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker
                } for seg in transcription_result.segments
            ],
            "speakers": [
                {
                    "id": spk.id,
                    "role": spk.role
                } for spk in transcription_result.speakers
            ]
        }
        
        # Save as JSON
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved final transcription result to {transcript_path}")
        
    except Exception as e:
        logger.error(f"Error saving final transcription result: {e}\n{traceback.format_exc()}")



def save_transcription_result(split_path, result):
    """
    Save transcription result as JSON file.
    
    Args:
        split_path: Path to the original split audio file
        result: Transcription result dictionary
    """
    try:
        # Extract audio_uuid from split_path
        split_dir = os.path.dirname(split_path)
        audio_uuid = os.path.basename(split_dir)
        
        # Create directory for transcripts if it doesn't exist
        transcript_dir = config.TEMP_DIR / "transcripts" / audio_uuid
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSON filename based on split index
        split_idx = result["index"]
        json_filename = f"id[{audio_uuid}]_split[{split_idx}].json"
        json_path = transcript_dir / json_filename
        
        # Save result as JSON
        with open(json_path, "w", encoding="utf-8") as f:
            # Remove is_coroutine from the saved result
            save_result = {k: v for k, v in result.items() if k != "is_coroutine"}
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved transcription result to {json_path}")
    except Exception as e:
        err_msg = f"ERROR in save_transcription_result: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)


def test_main():

    import whisper
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import os
    from pathlib import Path
    
    print("Starting simplified audio analysis...")
    
    # Set file path
    file_path = '/Users/beaver.baek/Documents/audio_samples/audio-test_short_34sec.mp3'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        exit(1)
        
    print(f"Processing file: {file_path}")
    
    # Load audio file
    print("Loading audio file...")
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Transcribe with Whisper
    print("Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    segments = result["segments"]
    
    print("\nTranscription:")
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f} - {end:.2f}] {text}")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Plot transcription timeline
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 1)
    ax2.set_title('Transcription Timeline')
    ax2.set_xlabel('Time (s)')
    ax2.get_yaxis().set_visible(False)
    
    # Add text annotations for segments
    text_height = 0.8
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"][:50] + "..." if len(segment["text"]) > 50 else segment["text"]
        
        # Draw segment boundaries
        ax2.axvline(x=start, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(x=end, color='r', linestyle='--', alpha=0.5)
        
        # Add text annotation
        ax2.text(start, text_height, text, fontsize=8, 
                horizontalalignment='left', verticalalignment='center')
        
        # Draw line to indicate segment
        ax2.plot([start, end], [0.5, 0.5], linewidth=2, color='black')
        
        # Alternate text height for better readability
        text_height = 0.2 if text_height == 0.8 else 0.8
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"analysis_visualization_{os.path.basename(file_path)}.png"
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    
    # Show figure
    plt.show()
    
    print("Analysis complete.")


def create_consecutive_segments(segments):
    """
    같은 역할을 가진 연속된 세그먼트를 병합
    같은 역할의 세그먼트 n초 이내로 다시 나타날 경우에도 연속으로 처리
    동시발화 상황도 처리함
    
    Parameters:
        segments (list): 전사 세그먼트 목록
        
    Returns:
        dict: 병합된 연속 세그먼트 목록과 병합 로그
    """
    
    # 역할별로 세그먼트 그룹화
    role_groups = {}
    for segment in segments:
        role = segment.get("speaker_role", -1)
        if role not in role_groups:
            role_groups[role] = []
        role_groups[role].append(segment.copy())
    
    # 각 역할 그룹 내에서 근접 세그먼트 병합
    merged_groups = {}
    merge_logs = []
    allow_overlap_time = 0.5
    
    for role, role_segments in role_groups.items():
        # 각 역할에 대해 클러스터 형성
        clusters = []
        
        for segment in role_segments:
            added_to_cluster = False
            
            # 기존 클러스터에 추가 가능한지 확인
            for cluster in clusters:
                last_segment = cluster[-1]
                time_gap = segment.get("start", 0) - last_segment.get("end", 0)
                
                # 직접 연속이거나 허용 시간 내 간격이면 클러스터에 추가
                if time_gap <= allow_overlap_time:
                    # 시간 간격이 있는 경우 로그 생성
                    if time_gap > 0:
                        merge_log = {
                            "role": role,
                            "first_segment": {
                                "speaker": last_segment.get("speaker", -1),
                                "start": last_segment.get("start", 0),
                                "end": last_segment.get("end", 0),
                                "text": last_segment.get("text", "")
                            },
                            "second_segment": {
                                "speaker": segment.get("speaker", -1),
                                "start": segment.get("start", 0),
                                "end": segment.get("end", 0),
                                "text": segment.get("text", "")
                            },
                            "time_gap": time_gap
                        }
                        merge_logs.append(merge_log)
                        # print(f"[근접 병합] 역할: {role}, 시간 간격: {time_gap:.2f}초")
                        # print(f"  첫 세그먼트: {last_segment.get('speaker', -1)} - {last_segment.get('text', '')[:30]}...")
                        # print(f"  다음 세그먼트: {segment.get('speaker', -1)} - {segment.get('text', '')[:30]}...")
                    
                    cluster.append(segment)
                    added_to_cluster = True
                    break
            
            # 어떤 클러스터에도 추가되지 않았으면 새 클러스터 생성
            if not added_to_cluster:
                clusters.append([segment])
        
        # 각 클러스터를 하나의 세그먼트로 병합
        merged_segments = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                # 단일 세그먼트는 그대로 유지
                if "id" in cluster[0]:
                    cluster[0]["word_segment_ids"] = [cluster[0]["id"]]
                else:
                    cluster[0]["word_segment_ids"] = []
                merged_segments.append(cluster[0])
            else:
                # 여러 세그먼트 병합
                merged = cluster[0].copy()
                merged["end"] = cluster[-1]["end"]
                
                # 텍스트 병합
                texts = []
                for i, seg in enumerate(cluster):
                    if i > 0:
                        time_gap = seg.get("start", 0) - cluster[i-1].get("end", 0)
                        if time_gap > 0:
                            texts.append("\n" + seg.get("text", ""))
                        else:
                            texts.append(" " + seg.get("text", ""))
                    else:
                        texts.append(seg.get("text", ""))
                
                merged["text"] = "".join(texts)
                
                # 단어 목록 병합 (있는 경우)
                if any("words" in seg for seg in cluster):
                    merged["words"] = []
                    for seg in cluster:
                        if "words" in seg:
                            merged["words"].extend(seg["words"])
                merged["word_segment_ids"] = [seg["id"] for seg in cluster if "id" in seg]
                merged_segments.append(merged)
        
        merged_groups[role] = merged_segments
    
    # 모든 역할의 세그먼트를 하나의 리스트로 결합하고 시간순 정렬
    consecutive_segments = []
    for role_segments in merged_groups.values():
        consecutive_segments.extend(role_segments)
    
    consecutive_segments = sorted(consecutive_segments, key=lambda x: x.get("start", 0))
    
    # 연속 세그먼트에 ID 부여
    for i, segment in enumerate(consecutive_segments):
        segment["contiseg_id"] = i
    
    return {
        "word_segments": segments,
        "consecutive_segments": consecutive_segments,
        "merge_logs": merge_logs
    }