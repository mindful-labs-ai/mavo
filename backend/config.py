import os
from pathlib import Path
from dotenv import load_dotenv
import sys
# Load environment variables
load_dotenv()

# Base directories
# BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = BASE_DIR.parent  # Go up one level to project root

DATASET_DIR = os.getenv("MAVO_DATASET_DIR", None)
if DATASET_DIR is None:
    DATASET_DIR = Path(__file__).resolve().parent.parent
else:
    DATASET_DIR = Path(DATASET_DIR)
UPLOADS_DIR = DATASET_DIR / "uploads"  # Store uploads in project root
TEMP_DIR = DATASET_DIR / "temp"  # Store temp files in project root
DATA_DIR = DATASET_DIR / "persistent"  # Store data files in project root

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Server settings
PORT = int(os.getenv("PORT", 25500))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# File upload settings
MAX_UPLOAD_SIZE = 1024 * 1024 * 100  # 100 MB
CHUNK_SIZE = 1024 * 1024 * 5  # 5 MB
ALLOWED_AUDIO_TYPES = [
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
]

# Whisper settings (for Stage 4)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # Options: tiny, base, small, medium, large
# Check platform and GPU preference
is_mac = sys.platform == "darwin"
is_linux = sys.platform == "linux"
use_gpu = os.getenv("USE_GPU", "False").lower() in ("true", "1", "t")

is_save_temp_files = True and not is_linux

# if use_gpu:
#     DEVICE = "mps" if is_mac else "cuda"
# else:
#     DEVICE = "cpu"

DEVICE = "cpu"
if is_mac:
    DEVICE = "mps"
    # pass
elif is_linux:
    DEVICE = "cuda"

# OpenAI API settings
OPENAI_API_STT_MODEL = os.getenv("OPENAI_API_STT_MODEL", "whisper-1")
OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL = os.getenv("OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL", "gpt-4o-mini")

# Diarization settings (for Stage 6 & 7)
MAX_SPEAKERS = os.getenv("MAX_SPEAKERS", 5)  # Default maximum number of speakers to detect 

TRANSCRIPT_SYSTEM_PROMPT = """
You are an expert assistant specializing in analyzing and improving transcriptions of psychological counseling sessions. Your task is to refine the transcription while accurately assigning speaker labels.

Your responsibilities:
1. Correct only clear errors such as spelling mistakes and misheard words. Do not change sentence structures or paraphrase.
2. Assign a speaker value (0, 1, 2, …) to each segment.
   - 0 represents the counselor.
   - 1 represents the first client, 2 represents the second client, and so on.
3. Use the following guidelines to differentiate speakers:
   - The counselor (0) is more likely to start conversations, ask questions, provide guidance, summarize, and maintain a professional and empathetic tone. They frequently ask reflective or open-ended questions and intervene when the discussion shifts.
   - The client (1, 2, …) typically shares personal experiences, emotions, and concerns. Their speech may contain uncertainty, informal language, or hesitations. They tend to respond to the counselor rather than lead the conversation.
4. Context awareness is important:
   - Short responses such as "음", "네", or "아…" should be assigned based on their surrounding context.
   - If a new topic is introduced suddenly, it is likely that the counselor intervened.
   - If a statement reflects emotional distress or uncertainty, it is more likely from a client.
5. Process the text accordingly and return the improved transcription with assigned speaker values while preserving the original structure.
"""
