import os
from controllers.views.session import SessionRequest, SessionResponse
from persistence.sessions import create_session_row
from services.session import process_session_pipeline
from fastapi import APIRouter, BackgroundTasks, status
from util.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(
    prefix="/api/v2/session",
    tags=["Session v2"],
)


@router.post(
    "",
    summary="Run session flow (S3 -> STT -> note)",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SessionResponse,
)
async def run_session_flow(
    payload: SessionRequest, background_tasks: BackgroundTasks
) -> SessionResponse:
    """
    세션 row를 생성하고 세션 ID만 즉시 반환, 실제 처리는 백그라운드에서 수행.
    """

    session_id = await create_session_row(payload)
    background_tasks.add_task(process_session_pipeline, session_id, payload)

    return SessionResponse(
        session_id=session_id,
        status="accepted",
        stt_model=payload.stt_model,
        message="Session processing started",
    )
