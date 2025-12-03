"""
backend/services/stt.py 내부 함수 단위 테스트
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경변수 mocking
TEST_ENV_VARS = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "GEMINI_API_KEY": "test-gemini-key",
    "OPENAI_API_KEY": "test-openai-key",
}
for key, value in TEST_ENV_VARS.items():
    os.environ.setdefault(key, value)

import pytest
from backend.services.stt import (
    parse_stt_response,
    parse_gemini_raw_output,
    parse_whisper_raw_output,
)


class TestParseSttResponse:
    """parse_stt_response 함수 테스트"""

    def test_routes_to_gemini_parser(self):
        """gemini 모델일 때 gemini 파서로 라우팅"""
        raw = "%T%00:00:01||C||안녕하세요.\n%T%00:00:03||P1||네, 안녕하세요."

        result = parse_stt_response(raw, "gemini-3")

        assert result["stt_model"] == "gemini-3"
        assert len(result["segments"]) == 2

    def test_routes_to_whisper_parser(self):
        """whisper 모델일 때 whisper 파서로 라우팅"""
        raw = '{"segments": [{"text": "안녕", "start": 0, "end": 1}], "language": "ko"}'

        result = parse_stt_response(raw, "whisper")

        assert result["stt_model"] == "whisper"
        assert result["language"] == "ko"

    def test_unknown_model_returns_empty_segments(self):
        """지원하지 않는 모델은 빈 세그먼트 반환"""
        result = parse_stt_response("some raw output", "unknown_model")

        assert result["segments"] == []
        assert result["raw_output"] == "some raw output"


class TestParseGeminiRawOutput:
    """parse_gemini_raw_output 함수 테스트"""

    def test_parses_single_line(self):
        """단일 라인 파싱"""
        raw = "%T%00:00:01||C||안녕하세요."

        result = parse_gemini_raw_output(raw, "gemini-3")

        assert len(result["segments"]) == 1
        assert result["segments"][0]["speaker"] == "C"
        assert result["segments"][0]["text"] == "안녕하세요."
        assert result["segments"][0]["start"] is None  # 타임스탬프 제거됨

    def test_parses_multiple_lines(self):
        """여러 라인 파싱"""
        raw = """%T%00:00:01||C||안녕하세요.
%T%00:00:03||P1||네, 안녕하세요.
%T%00:00:05||P2||저도요."""

        result = parse_gemini_raw_output(raw, "gemini-3")

        assert len(result["segments"]) == 3
        assert result["segments"][0]["speaker"] == "C"
        assert result["segments"][1]["speaker"] == "P1"
        assert result["segments"][2]["speaker"] == "P2"

    def test_extracts_text_content(self):
        """텍스트 내용 추출"""
        raw = "%T%00:00:01||C||오늘 기분이 어떠세요?"

        result = parse_gemini_raw_output(raw, "gemini-3")

        assert result["text"] == "오늘 기분이 어떠세요?"

    def test_handles_special_tags(self):
        """특수 태그 포함된 텍스트 파싱"""
        raw = "%T%00:00:01||P1||{%A%한숨%} 저는 그냥 답답해요."

        result = parse_gemini_raw_output(raw, "gemini-3")

        assert "{%A%한숨%}" in result["segments"][0]["text"]

    def test_skips_invalid_lines(self):
        """잘못된 형식의 라인은 건너뜀"""
        raw = """이것은 잘못된 라인
%T%00:00:01||C||올바른 라인
또 다른 잘못된 라인"""

        result = parse_gemini_raw_output(raw, "gemini-3")

        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "올바른 라인"

    def test_handles_empty_input(self):
        """빈 입력 처리"""
        result = parse_gemini_raw_output("", "gemini-3")

        assert result["segments"] == []
        assert result["text"] == ""

    def test_strips_quotes_from_raw_output(self):
        """앞뒤 따옴표 제거"""
        raw = '"%T%00:00:01||C||안녕하세요."'

        result = parse_gemini_raw_output(raw, "gemini-3")

        assert len(result["segments"]) == 1

    def test_preserves_raw_output(self):
        """원본 raw_output 보존"""
        raw = "%T%00:00:01||C||테스트"

        result = parse_gemini_raw_output(raw, "gemini-3")

        assert result["raw_output"] == raw


class TestParseWhisperRawOutput:
    """parse_whisper_raw_output 함수 테스트"""

    def test_parses_valid_json(self):
        """유효한 JSON 파싱"""
        raw = '{"segments": [{"text": "안녕하세요", "start": 0.0, "end": 1.5, "speaker": 0}], "language": "ko"}'

        result = parse_whisper_raw_output(raw, "whisper")

        assert result["language"] == "ko"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "안녕하세요"

    def test_handles_multiple_segments(self):
        """여러 세그먼트 처리"""
        raw = '{"segments": [{"text": "첫번째", "start": 0, "end": 1}, {"text": "두번째", "start": 1, "end": 2}], "language": "ko"}'

        result = parse_whisper_raw_output(raw, "whisper")

        assert len(result["segments"]) == 2

    def test_extracts_full_text(self):
        """전체 텍스트 추출"""
        raw = '{"text": "전체 텍스트입니다", "segments": [], "language": "ko"}'

        result = parse_whisper_raw_output(raw, "whisper")

        assert result["text"] == "전체 텍스트입니다"

    def test_builds_text_from_segments_if_missing(self):
        """text 필드 없으면 segments에서 조합"""
        raw = '{"segments": [{"text": "첫번째"}, {"text": "두번째"}], "language": "ko"}'

        result = parse_whisper_raw_output(raw, "whisper")

        assert result["text"] == "첫번째 두번째"

    def test_handles_invalid_json(self):
        """잘못된 JSON 처리"""
        raw = "이건 JSON이 아닙니다"

        result = parse_whisper_raw_output(raw, "whisper")

        assert result["segments"] == []
        assert result["language"] == "ko"  # 기본값

    def test_handles_empty_json(self):
        """빈 JSON 처리"""
        raw = "{}"

        result = parse_whisper_raw_output(raw, "whisper")

        assert result["segments"] == []

    def test_preserves_raw_output(self):
        """원본 raw_output 보존"""
        raw = '{"segments": [], "language": "ko"}'

        result = parse_whisper_raw_output(raw, "whisper")

        assert result["raw_output"] == raw

    def test_preserves_stt_model(self):
        """stt_model 보존"""
        raw = '{"segments": []}'

        result = parse_whisper_raw_output(raw, "whisper")

        assert result["stt_model"] == "whisper"


# 직접 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
