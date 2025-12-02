import os
from typing import Optional
from backend.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
from backend.controllers.views.session import SessionRequest
import httpx

from util.logger import get_logger

logger = get_logger(__name__)


async def create_session_row(payload: SessionRequest) -> str:
    """
    Supabase sessions 테이블에 새 row를 생성하고 id를 반환.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise SystemError("Supabase 환경변수가 설정되지 않았습니다.")

    insert_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/sessions"
    trimmed_title = payload.title[:50] if payload.title else ""
    payload_body = {
        "user_id": payload.user_id,
        "title": trimmed_title,
        "audio_meta_data": {
            "s3_key": payload.s3_key,
            "file_size_mb": payload.file_size_mb,
            "duration_seconds": payload.duration_seconds,
        },
    }
    if payload.client_id:
        payload_body["client_id"] = str(payload.client_id)
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(insert_url, headers=headers, json=payload_body)

    if resp.status_code not in (200, 201):
        logger.error(
            f"Failed to insert session row: status={resp.status_code} body={resp.text}"
        )
        raise SystemError("세션 생성에 실패했습니다.")

    data = resp.json()
    session_row = (
        data[0]
        if isinstance(data, list) and data
        else data if isinstance(data, dict) else {}
    )
    session_id = session_row.get("id")
    if not session_id:
        raise SystemError("세션 ID를 가져오지 못했습니다.")

    return str(session_id)


async def update_session_status(
    session_id: str, status_value: str, error_message: Optional[str] = None
) -> None:
    """
    sessions 테이블의 processing_status 및 error_message를 업데이트.
    status_value: "succeeded" 또는 "failed"
    error_message: 실패 시 상세 메시지
    """

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise SystemError("Supabase 환경변수가 설정되지 않았습니다.")

    update_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/sessions?id=eq.{session_id}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    body = {
        "processing_status": status_value,
        "error_message": error_message,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.patch(update_url, headers=headers, json=body)

    if resp.status_code not in (200, 204):
        logger.error(
            f"Session status update failed: status={resp.status_code} body={resp.text}"
        )
        raise SystemError("세션 상태 업데이트에 실패했습니다.")
