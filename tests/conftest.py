"""
Pytest 전역 설정 - 환경변수 Mocking 및 HTTP/DB Stub
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

# 테스트용 환경변수 (실제 값 대신 mock 값 사용)
TEST_ENV_VARS = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "AWS_ACCESS_KEY_ID": "test-access-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    "AWS_REGION": "ap-northeast-2",
    "S3_BUCKET_NAME": "test-bucket",
    "OPENAI_API_KEY": "test-openai-key",
    "HF_TOKEN": "test-hf-token",
}


@pytest.fixture(scope="session", autouse=True)
def mock_environment():
    """테스트 전에 환경변수를 mock 값으로 설정"""
    # 기존 값 백업
    original_env = {key: os.environ.get(key) for key in TEST_ENV_VARS}

    # mock 값 설정
    for key, value in TEST_ENV_VARS.items():
        os.environ[key] = value

    yield

    # 테스트 후 원래 값 복원
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


# =============================================================================
# HTTP Client Stub Fixtures
# =============================================================================


def create_mock_response(
    status_code: int = 200, json_data: dict = None, text: str = ""
):
    """Mock HTTP Response 생성 헬퍼"""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    mock_resp.json.return_value = json_data or {}
    return mock_resp


@pytest.fixture
def mock_httpx_client():
    """
    httpx.AsyncClient를 stub 처리.
    모든 HTTP 요청(POST, GET, PATCH, DELETE)을 mock 응답으로 대체.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # 기본 성공 응답 설정
        mock_client.post.return_value = create_mock_response(
            status_code=201, json_data=[{"id": "test-session-id-123"}]
        )
        mock_client.get.return_value = create_mock_response(
            status_code=200, json_data={}
        )
        mock_client.patch.return_value = create_mock_response(
            status_code=200, json_data={}
        )
        mock_client.delete.return_value = create_mock_response(
            status_code=200, json_data={}
        )

        yield mock_client


@pytest.fixture
def mock_supabase_session_create(mock_httpx_client):
    """
    Supabase sessions 테이블 INSERT 요청 stub.
    create_session_row()에서 사용하는 POST 요청을 mock.
    """
    mock_httpx_client.post.return_value = create_mock_response(
        status_code=201,
        json_data=[{"id": "test-session-id-123", "user_id": 1, "title": "테스트"}],
    )
    return mock_httpx_client


@pytest.fixture
def mock_supabase_session_update(mock_httpx_client):
    """
    Supabase sessions 테이블 UPDATE 요청 stub.
    update_session_status(), update_session_progress()에서 사용하는 PATCH 요청을 mock.
    """
    mock_httpx_client.patch.return_value = create_mock_response(
        status_code=200, json_data={}
    )
    return mock_httpx_client


@pytest.fixture
def mock_s3_presigned_url():
    """
    S3 presigned URL 발급 stub.
    """
    with patch("backend.util.s3.issue_presigned_url") as mock_issue:
        mock_issue.return_value = (
            "https://test-bucket.s3.amazonaws.com/test-audio.mp3?signed=true"
        )
        yield mock_issue


@pytest.fixture
def mock_stt_service():
    """
    STT 서비스 호출 stub.
    run_stt_from_url()을 mock.
    """
    with patch("backend.services.stt.run_stt_from_url") as mock_stt:
        mock_stt.return_value = '{"text": "테스트 음성 텍스트", "segments": []}'
        yield mock_stt


@pytest.fixture
def mock_all_external_services(
    mock_httpx_client,
    mock_s3_presigned_url,
    mock_stt_service,
):
    """
    모든 외부 서비스(HTTP, S3, STT)를 한 번에 stub 처리.
    통합 테스트에서 사용.
    """
    return {
        "httpx": mock_httpx_client,
        "s3": mock_s3_presigned_url,
        "stt": mock_stt_service,
    }
