from datetime import datetime
from typing import Any, Dict
from backend.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
import httpx

from util.logger import get_logger

logger = get_logger(__name__)


async def create_transcribe_row(
    session_id: str, user_id: int, stt_model: str, transcript_json: Dict[str, Any]
) -> str:
    """
    transcribes 테이블에 축어록을 저장하고 생성된 id를 반환.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise SystemError("Supabase 환경변수가 설정되지 않았습니다.")

    now = datetime.utcnow()
    title = f"축어록/{now:%Y/%m/%d}"
    counsel_date = f"{now:%Y-%m-%d}"

    insert_url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/transcribes"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    body = {
        "session_id": session_id,
        "user_id": user_id,
        "title": title,
        "counsel_date": counsel_date,
        "contents": transcript_json,
        "stt_model": stt_model,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(insert_url, headers=headers, json=body)

    if resp.status_code not in (200, 201):
        logger.error(
            f"Transcribe insert failed: status={resp.status_code} body={resp.text}"
        )
        raise SystemError("축어록 저장에 실패했습니다.")

    data = resp.json()
    transcribe_row = (
        data[0]
        if isinstance(data, list) and data
        else data if isinstance(data, dict) else {}
    )
    transcribe_id = transcribe_row.get("id")
    return str(transcribe_id) if transcribe_id else ""
