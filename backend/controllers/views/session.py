from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class SessionRequest(BaseModel):
    user_id: int
    title: str
    s3_key: str
    file_size_mb: float
    duration_seconds: float
    client_id: Optional[UUID] = None
    stt_model: str
    template_id: int


class SessionResponse(BaseModel):
    session_id: str
    status: str
    stt_model: str
    note_id: Optional[str] = None
    raw_stt_output: Optional[str] = None
    message: str
