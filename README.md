# Mavo Voice Analysis Server

A local server solution for voice analysis and diarization, designed to provide faster and more accurate speaker analysis.

MAVO: Mindful Analysis server for Voice input

## Project Overview

This project implements a FastAPI-based server that:
- Transcribes audio using speech-to-text technology
- Performs speaker diarization (identifying who is speaking)
- Processes large audio files by chunking them
- Provides a simple web interface for testing

## Setup Instructions

### Option 1: Docker (Recommended)

Docker provides a containerized environment that works across all platforms (macOS, Linux, Windows) with optimized performance and easy deployment.

#### Prerequisites
- Docker (20.10 or higher)
- Docker Compose (optional, but recommended)
- NVIDIA GPU drivers (optional, for GPU acceleration)

#### Quick Start with Docker

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd mavo
   ```

2. Build and run the container:
   ```bash
   # Build the image
   ./docker-build.sh

   # Run the server
   ./docker-run.sh
   ```

   Or use Docker Compose (recommended):
   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit .env with your configuration (especially OPENAI_API_KEY if needed)

   # Build and run
   docker-compose up --build -d
   ```

3. Access the application:
   - Web interface: http://localhost:25500/mavo
   - API documentation: http://localhost:25500/docs
   - Health check: http://localhost:25500/

#### Docker Development Mode

For development with hot reload:
```bash
./docker-dev.sh
```

#### Docker Commands

```bash
# Build image
./docker-build.sh

# Run in production mode
./docker-run.sh

# Run in development mode
./docker-dev.sh

# View logs
docker-compose logs -f

# Stop server
docker-compose down

# Clean up
docker system prune -a
```

#### Docker Configuration

The container supports GPU acceleration and includes:
- Multi-stage build for optimized image size
- Health checks and restart policies
- Volume mounting for persistent data
- Environment-based configuration

### Option 2: Local Development

#### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- FFmpeg (for audio processing)

#### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd mavo
   ```

2. Set up the development environment:
   ```
   # Make the setup script executable if needed
   chmod +x devenv/setup.sh
   
   # Run the setup script
   ./devenv/setup.sh
   ```
   This will create a virtual environment in the project root and install all dependencies.

   Alternatively, you can use the exact version setup script for better reproducibility:
   ```
   chmod +x devenv/setup_exact.sh
   ./devenv/setup_exact.sh
   ```
   This will install the exact versions of dependencies that have been tested with this project.

### Configuration

The server uses environment variables for configuration. These can be set in a `.env` file in the project root:

```
# Server settings
PORT=25500
HOST=0.0.0.0
DEBUG=True

# Whisper settings
WHISPER_MODEL=tiny  # Options: tiny, base, small, medium, large
USE_GPU=False

# OpenAI API settings
OPENAI_API_KEY=
OPENAI_API_STT_MODEL=whisper-1

# Diarization settings
MAX_SPEAKERS=3
```

### Running the Server

From the project root directory, run:
```
./devenv/run_server.sh
```

Or, if you've already activated the virtual environment:
```
cd backend
uvicorn main:app --host 0.0.0.0 --port $PORT --reload
```

The server will start at `http://localhost:$PORT` (default: 25500).

- Main page: `http://localhost:$PORT/` (redirects to the test page)
- API documentation: `http://localhost:$PORT/doc`
- API alternative documentation: `http://localhost:$PORT/redoc`
- API endpoint: `http://localhost:$PORT/api/v1/`
- Test page: `http://localhost:$PORT/public/index.html`

### Development Workflow

To activate the virtual environment for development:
```
source devenv/activate.sh
```

## API Endpoints

The following API endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/` | GET | Get API information |
| `/api/v1/upload/chunk` | POST | Upload a chunk of an audio file |
| `/api/v1/upload/status/{audio_uuid}` | GET | Get the status of an audio file upload and processing |
| `/api/v1/upload/result/{audio_uuid}` | GET | Get the result of audio analysis |

For detailed API documentation, visit the Swagger UI at `/doc`.

## File Upload Process

The file upload process follows these steps:

1. The client generates a unique UUID for the audio file
2. The client uploads the audio file as a single chunk (in Stage 3) or multiple chunks (in Stage 5)
3. The server stores the chunks and assembles them when all chunks are received
4. The server processes the audio file in the background
5. The client polls the server for the status of the processing
6. When processing is complete, the client retrieves the result

## Speech-to-Text Processing

The system supports two methods for speech-to-text processing:

### 1. Local Processing with OpenAI Whisper

The local processing uses OpenAI's Whisper model:

1. The uploaded audio file is converted to a format compatible with Whisper (WAV, 16kHz, mono)
2. The Whisper model transcribes the audio to text
3. A simple speaker diarization algorithm assigns speakers based on pauses in the audio
4. The transcription is returned with speaker labels and timestamps

The Whisper model can be configured in the `.env` file:
- `WHISPER_MODEL`: The model to use (tiny, base, small, medium, large)
- `USE_GPU`: Whether to use GPU acceleration (if available)

### 2. Cloud Processing with OpenAI API

Alternatively, you can use the OpenAI API for speech-to-text:

1. The uploaded audio file is converted to a format compatible with the API
2. The file is sent to the OpenAI API for transcription
3. The same speaker diarization algorithm is applied to the results
4. The transcription is returned with speaker labels and timestamps

To use the OpenAI API, set the following in your `.env` file:
- `OPENAI_API_KEY=your_api_key_here`
- `OPENAI_API_STT_MODEL=whisper-1`

## Project Structure

```
mavo/
├── backend/             # Backend server code
│   ├── controllers/     # API controllers
│   │   ├── core_controller.py  # Core API endpoints
│   │   └── upload_controller.py  # Upload and processing endpoints
│   ├── public/          # Static files served by the server
│   │   └── index.html   # Test page with UI for file upload
│   ├── temp/            # Temporary files
│   ├── uploads/         # Uploaded audio files
│   ├── config.py        # Configuration settings
│   ├── docs.py          # API documentation
│   ├── main.py          # Main application entry point
│   └── models.py        # Data models
├── devenv/              # Development environment scripts
│   ├── activate.sh      # Script to activate the virtual environment
│   ├── run_server.sh    # Script to run the server
│   └── setup.sh         # Script to set up the development environment
├── docker/              # Docker-related files (optional subdirectory)
├── Dockerfile           # Multi-stage Docker build configuration
├── docker-compose.yml   # Docker Compose configuration
├── .dockerignore        # Docker ignore file
├── docker-build.sh      # Docker build script
├── docker-run.sh        # Docker run script
├── docker-dev.sh        # Docker development script
├── .env                 # Environment variables
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
└── requirements.simple.txt # Python dependencies for Docker
```

## Development Stages

This project is being developed in stages:

1. ✅ **Environment Setup** - Basic FastAPI server with static file serving
2. ✅ **UI & API Design** - Frontend implementation and API endpoint definition
3. ✅ **Single-Chunk File Upload** - Basic file upload functionality
4. ✅ **Voice Analysis** (Current) - Speech-to-text transcription
5. 🔄 **Multi-Chunk File Upload** - Support for large file uploads
6. 🔄 **Basic Speaker Diarization** - Initial speaker identification
7. 🔄 **Advanced Speaker Diarization** - Improved accuracy
8. 🔄 **Performance Optimization** - Speed improvements
9. 🔄 **Future Enhancements** - Additional features

## License

[Specify your license here]
