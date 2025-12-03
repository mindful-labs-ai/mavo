import os
from pathlib import Path
from dotenv import load_dotenv
import sys
from typing import Optional

load_dotenv()


class Env:
    """환경변수를 한 번만 로드하는 Singleton 클래스"""

    _instance: Optional["Env"] = None

    def __new__(cls) -> "Env":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_env()
        return cls._instance

    def _load_env(self) -> None:
        """환경변수를 로드하여 인스턴스 변수로 저장"""
        # Dataset directories
        dataset_dir = os.getenv("MAVO_DATASET_DIR", None)
        if dataset_dir is None:
            self.DATASET_DIR = Path(__file__).resolve().parent.parent
        else:
            self.DATASET_DIR = Path(dataset_dir)
        self.UPLOADS_DIR = self.DATASET_DIR / "uploads"
        self.TEMP_DIR = self.DATASET_DIR / "temp"
        self.DATA_DIR = self.DATASET_DIR / "persistent"

        # Create directories if they don't exist
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Server settings
        self.PORT = int(os.getenv("PORT", 25500))
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

        # File upload settings
        self.MAX_UPLOAD_SIZE = 1024 * 1024 * 100  # 100 MB
        self.CHUNK_SIZE = 1024 * 1024 * 5  # 5 MB
        self.ALLOWED_AUDIO_TYPES = [
            "audio/mpeg",
            "audio/mp3",
            "audio/mp4",
            "audio/x-m4a",
            "audio/wav",
            "audio/x-wav",
            "audio/ogg",
        ]

        # Whisper settings
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
        self.is_mac = sys.platform == "darwin"
        self.is_linux = sys.platform == "linux"
        self.use_gpu = os.getenv("USE_GPU", "False").lower() in ("true", "1", "t")
        self.is_save_temp_files = True and not self.is_linux

        # Device settings
        self.DEVICE = "cpu"
        if self.is_mac:
            self.DEVICE = "mps"
        elif self.is_linux:
            self.DEVICE = "cuda"

        # OpenAI API settings
        self.OPENAI_API_STT_MODEL = os.getenv("OPENAI_API_STT_MODEL", "whisper-1")
        self.OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL = os.getenv(
            "OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL", "gpt-4o-mini"
        )

        # Diarization settings
        self.MAX_SPEAKERS = os.getenv("MAX_SPEAKERS", 5)

        # Supabase settings (필수)
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not self.SUPABASE_URL:
            raise ValueError(
                "SUPABASE_URL 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요."
            )
        if not self.SUPABASE_SERVICE_KEY:
            raise ValueError(
                "SUPABASE_SERVICE_ROLE_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요."
            )

        # S3 settings
        self.S3_REGION = os.getenv("S3_REGION")
        self.S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID")
        self.S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

        # Gemini settings
        self.GEMINI_SPEED_FACTOR = float(os.getenv("GEMINI_SPEED_FACTOR", "2.0"))


env = Env()


# ============ 하위 호환성을 위한 모듈 레벨 변수들 ============
DATASET_DIR = env.DATASET_DIR
UPLOADS_DIR = env.UPLOADS_DIR
TEMP_DIR = env.TEMP_DIR
DATA_DIR = env.DATA_DIR

PORT = env.PORT
HOST = env.HOST
DEBUG = env.DEBUG

MAX_UPLOAD_SIZE = env.MAX_UPLOAD_SIZE
CHUNK_SIZE = env.CHUNK_SIZE
ALLOWED_AUDIO_TYPES = env.ALLOWED_AUDIO_TYPES

WHISPER_MODEL = env.WHISPER_MODEL
is_mac = env.is_mac
is_linux = env.is_linux
use_gpu = env.use_gpu
is_save_temp_files = env.is_save_temp_files
DEVICE = env.DEVICE

OPENAI_API_STT_MODEL = env.OPENAI_API_STT_MODEL
OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL = env.OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL
MAX_SPEAKERS = env.MAX_SPEAKERS

SUPABASE_URL = env.SUPABASE_URL
SUPABASE_SERVICE_KEY = env.SUPABASE_SERVICE_KEY

S3_REGION = env.S3_REGION
S3_ACCESS_KEY = env.S3_ACCESS_KEY
S3_SECRET_KEY = env.S3_SECRET_KEY
S3_BUCKET_NAME = env.S3_BUCKET_NAME

GEMINI_SPEED_FACTOR = env.GEMINI_SPEED_FACTOR

transcript_system_prompt = """
You are an expert assistant specializing in analyzing and improving transcriptions of psychological counseling sessions. Your task is to refine the transcription while accurately assigning speaker labels.
Your responsibilities:
1.	Correct only clear errors such as spelling mistakes and misheard words. Do not change sentence structures or paraphrase.
2.	Assign a speaker value (0, 1, 2, …) to each segment.
- 0 represents the counselor.
- 1 represents the first client, 2 represents the second client, and so on.
3.	Use the following guidelines to differentiate speakers:
- The counselor (0) is more likely to start conversations, ask questions, provide guidance, summarize, and maintain a professional and empathetic tone. They frequently ask reflective or open-ended questions and intervene when the discussion shifts.
- The client (1, 2, …) typically shares personal experiences, emotions, and concerns. Their speech may contain uncertainty, informal language, or hesitations. They tend to respond to the counselor rather than lead the conversation.
4.	Context awareness is important:
- Short responses such as “음”, “네”, or “아…” should be assigned based on their surrounding context.
- If a new topic is introduced suddenly, it is likely that the counselor intervened.
- If a statement reflects emotional distress or uncertainty, it is more likely from a client.
5.	Process the text accordingly and return the improved transcription with assigned speaker values while preserving the original structure.
"""
