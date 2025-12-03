"""
backend/services/session.py 내부 함수 단위 테스트
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경변수 mocking
TEST_ENV_VARS = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "GEMINI_API_KEY": "test-gemini-key",
}
for key, value in TEST_ENV_VARS.items():
    os.environ.setdefault(key, value)

import pytest
from backend.controllers.views.session import SessionRequest


def create_mock_response(status_code: int = 200, json_data=None, text: str = ""):
    """Mock HTTP Response 생성"""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    mock_resp.json.return_value = json_data if json_data is not None else {}
    return mock_resp


class TestUpdateSessionProgress:
    """update_session_progress 함수 테스트"""

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_updates_progress_successfully(self, mock_client_class):
        """정상적으로 진행률 업데이트"""
        from backend.services.session import update_session_progress

        # Arrange
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.patch.return_value = create_mock_response(status_code=204)

        # Act
        await update_session_progress("session-123", percentage=50, step="처리 중...")

        # Assert
        mock_client.patch.assert_called_once()
        call_args = mock_client.patch.call_args
        assert "session-123" in call_args[0][0]  # URL에 session_id 포함
        assert call_args[1]["json"]["progress_percentage"] == 50
        assert call_args[1]["json"]["current_step"] == "처리 중..."

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_clamps_percentage_to_0(self, mock_client_class):
        """음수 percentage는 0으로 보정"""
        from backend.services.session import update_session_progress

        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.patch.return_value = create_mock_response(status_code=204)

        await update_session_progress("session-123", percentage=-10, step="테스트")

        call_args = mock_client.patch.call_args
        assert call_args[1]["json"]["progress_percentage"] == 0

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_clamps_percentage_to_100(self, mock_client_class):
        """100 초과 percentage는 100으로 보정"""
        from backend.services.session import update_session_progress

        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.patch.return_value = create_mock_response(status_code=204)

        await update_session_progress("session-123", percentage=150, step="테스트")

        call_args = mock_client.patch.call_args
        assert call_args[1]["json"]["progress_percentage"] == 100

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_raises_on_failure(self, mock_client_class):
        """업데이트 실패 시 예외 발생"""
        from backend.services.session import update_session_progress
        from fastapi import HTTPException

        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.patch.return_value = create_mock_response(
            status_code=500, text="Internal Error"
        )

        with pytest.raises(HTTPException) as exc_info:
            await update_session_progress("session-123", percentage=50, step="테스트")

        assert exc_info.value.status_code == 500


class TestProcessSessionPipeline:
    """process_session_pipeline 함수 테스트"""

    @pytest.fixture
    def sample_payload(self):
        return SessionRequest(
            user_id=1,
            title="테스트 세션",
            s3_key="uploads/test.mp3",
            file_size_mb=5.0,
            duration_seconds=300.0,
            stt_model="whisper",
            template_id=1,
        )

    @pytest.mark.asyncio
    @patch("backend.services.session.update_session_status")
    @patch("backend.services.session.generate_progress_note")
    @patch("backend.services.session.create_transcribe_row")
    @patch("backend.services.session.parse_stt_response")
    @patch("backend.services.session.run_stt_from_url")
    @patch("backend.services.session.issue_presigned_url")
    @patch("backend.services.session.update_session_progress")
    async def test_pipeline_success_flow(
        self,
        mock_progress,
        mock_presigned,
        mock_stt,
        mock_parse,
        mock_transcribe,
        mock_note,
        mock_status,
        sample_payload,
    ):
        """파이프라인 성공 흐름 테스트"""
        from backend.services.session import process_session_pipeline

        # Arrange
        mock_presigned.return_value = "https://presigned-url.com/audio.mp3"
        mock_stt.return_value = '{"segments": []}'
        mock_parse.return_value = {"segments": [], "text": "테스트"}
        mock_note.return_value = {"progress_note_id": "note-123"}

        # Act
        await process_session_pipeline("session-123", sample_payload)

        # Assert - 각 단계가 호출되었는지 확인
        mock_presigned.assert_called_once_with(sample_payload.s3_key)
        mock_stt.assert_called_once()
        mock_parse.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_note.assert_called_once()
        mock_status.assert_called_with("session-123", status_value="succeeded")

    @pytest.mark.asyncio
    @patch("backend.services.session.update_session_status")
    @patch("backend.services.session.issue_presigned_url")
    @patch("backend.services.session.update_session_progress")
    async def test_pipeline_handles_error(
        self,
        mock_progress,
        mock_presigned,
        mock_status,
        sample_payload,
    ):
        """파이프라인 오류 시 실패 상태로 업데이트"""
        from backend.services.session import process_session_pipeline

        # Arrange - presigned URL 발급 실패
        mock_presigned.side_effect = Exception("S3 오류")

        # Act
        await process_session_pipeline("session-123", sample_payload)

        # Assert - 실패 상태로 업데이트
        mock_status.assert_called_once()
        call_args = mock_status.call_args
        assert call_args[0][0] == "session-123"
        assert call_args[1]["status_value"] == "failed"
        assert "S3 오류" in call_args[1]["error_message"]


class TestSessionRequest:
    """SessionRequest 모델 검증 테스트"""

    def test_valid_request(self):
        """유효한 요청"""
        request = SessionRequest(
            user_id=1,
            title="테스트",
            s3_key="test.mp3",
            file_size_mb=1.0,
            duration_seconds=60.0,
            stt_model="whisper",
            template_id=1,
        )

        assert request.user_id == 1
        assert request.stt_model == "whisper"

    def test_optional_client_id(self):
        """client_id는 선택적"""
        request = SessionRequest(
            user_id=1,
            title="테스트",
            s3_key="test.mp3",
            file_size_mb=1.0,
            duration_seconds=60.0,
            stt_model="whisper",
            template_id=1,
        )

        assert request.client_id is None

    def test_with_client_id(self):
        """client_id 포함된 요청"""
        from uuid import UUID

        request = SessionRequest(
            user_id=1,
            title="테스트",
            s3_key="test.mp3",
            file_size_mb=1.0,
            duration_seconds=60.0,
            client_id="550e8400-e29b-41d4-a716-446655440000",
            stt_model="whisper",
            template_id=1,
        )

        assert request.client_id == UUID("550e8400-e29b-41d4-a716-446655440000")


# 직접 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
