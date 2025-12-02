"""
POST /api/v2/session 엔드포인트 간단 테스트
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

TEST_ENV_VARS = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "AWS_ACCESS_KEY_ID": "test-access-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    "AWS_REGION": "ap-northeast-2",
    "S3_BUCKET_NAME": "test-bucket",
    "OPENAI_API_KEY": "test-openai-key",
}
for key, value in TEST_ENV_VARS.items():
    os.environ.setdefault(key, value)

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


# 테스트용 앱 임포트
@pytest.fixture
def client():
    """FastAPI TestClient 생성"""
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def sample_session_request():
    """테스트용 세션 요청 데이터"""
    return {
        "user_id": 1,
        "title": "테스트 세션",
        "s3_key": "uploads/test-audio.mp3",
        "file_size_mb": 5.0,
        "duration_seconds": 300.0,
        "stt_model": "whisper",
        "template_id": 1,
    }


class TestSessionEndpoint:
    """POST /api/v2/session 테스트"""

    def test_session_returns_200_ok(
        self,
        client: TestClient,
        sample_session_request: dict,
        mock_httpx_client,  # conftest.py에서 제공하는 HTTP stub
    ):
        """
        정상 요청 시 200 OK 응답과 session_id를 반환하는지 테스트
        (httpx.AsyncClient가 stub 처리되어 실제 Supabase 호출 없음)
        """
        # Act
        response = client.post("/api/v2/session/", json=sample_session_request)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-id-123"
        assert data["status"] == "accepted"
        assert data["stt_model"] == "whisper"
        assert "message" in data

        # HTTP POST가 호출되었는지 확인 (Supabase insert)
        mock_httpx_client.post.assert_called_once()

    def test_session_requires_all_fields(
        self,
        client: TestClient,
        mock_httpx_client,
    ):
        """
        필수 필드 누락 시 422 에러 반환 테스트
        """
        # Act - 빈 요청
        response = client.post("/api/v2/session/", json={})

        # Assert
        assert response.status_code == 422

    def test_session_validates_user_id(
        self,
        client: TestClient,
        mock_httpx_client,
    ):
        """
        user_id가 정수가 아닐 때 422 에러 반환 테스트
        """
        # Act
        response = client.post(
            "/api/v2/session/",
            json={
                "user_id": "not_an_integer",
                "title": "테스트",
                "s3_key": "test.mp3",
                "file_size_mb": 1.0,
                "duration_seconds": 60.0,
                "stt_model": "whisper",
                "template_id": 1,
            },
        )

        # Assert
        assert response.status_code == 422

    def test_session_handles_db_error(
        self,
        client: TestClient,
        sample_session_request: dict,
        mock_httpx_client,
    ):
        """
        DB 오류 시 500 에러 반환 테스트
        """
        # Arrange - DB 오류 시뮬레이션
        from tests.conftest import create_mock_response

        mock_httpx_client.post.return_value = create_mock_response(
            status_code=500, text="Internal Server Error"
        )

        # Act
        response = client.post("/api/v2/session/", json=sample_session_request)

        # Assert
        assert response.status_code == 500


# 직접 실행용 간단 테스트
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
