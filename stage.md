# Detailed Stage Plan

This document provides a structured development plan, breaking the project into distinct stages. Each stage should deliver a deployable version of the solution while incrementally adding features.

---

## Stage 1: Environment Setup

### Objective
- Establish the development environment with minimal setup.
- Ensure FastAPI serves a **public test page** alongside its API.

### Tasks
- Set up a **FastAPI** server with **CORS** settings enabled.
- Create a **public folder** in the FastAPI project to serve static files.
- Add an `index.html` file containing a simple UI with a file upload button.
- Verify that the FastAPI server correctly serves the test page.

### Deliverables
- A project folder structure (`backend` with a `public` directory).
- A minimal `index.html` page with a file upload form.

### Success Criteria
- Running `uvicorn main:app --reload` launches the server.
- The test page is accessible at `http://localhost:8000/public/index.html`.

---

## Stage 2: UI & API Design

### Objective
- Implement the UI and define the API endpoints.

### Tasks
- Improve `index.html`:
  - Add a **drag-and-drop area** for file uploads.
  - Display upload progress and results.
- Define API endpoints:
  - `POST /upload-chunk` (stub)
  - `GET /analysis-status` (stub)
  - `GET /analysis-result` (stub)
- Connect the UI to API endpoints with **Vanilla JS** (fetch API).

### Deliverables
- A test UI that interacts with the back-end stubs.

### Success Criteria
- The UI allows file selection and calls the API.
- API endpoints return expected mock responses.

---

## Stage 3: Single-Chunk File Upload

### Objective
- Implement file upload handling in a simple (single-chunk) format.

### Tasks
- Modify the front-end upload logic to send **one** chunk per file.
- Implement back-end logic to receive, store, and acknowledge file uploads.
- Introduce a lightweight class (`AnalysisWork`) to track uploads.
- Return an acknowledgment response after receiving a file.

### Deliverables
- File uploads work from the UI.
- The back end logs received files.

### Success Criteria
- The UI successfully sends files to the API.
- Uploaded files are stored temporarily on the server.

---

## Stage 4: Voice Analysis (Basic STT)

### Objective
- Implement speech-to-text (STT) transcription using **OpenAI Whisper**.

### Tasks
- Install and configure Whisper for transcription.
- Process uploaded audio, generating transcripts asynchronously.
- Store transcripts in `AnalysisWork` objects.
- Provide an endpoint (`GET /analysis-result`) to return transcripts.

### Deliverables
- The back end can transcribe uploaded audio.
- The UI displays transcribed text after processing.

### Success Criteria
- Uploading an audio file results in a transcript appearing in the UI.

---

## Stage 5: Audio Chunking Implementation

### Objective
- Enable chunked file uploads to support larger audio files.
- Implement Voice Activity Detection (VAD) for intelligent audio splitting.
- Process audio segments in parallel for faster transcription.

### Tasks
- Modify the front end to split files into **2 MB** chunks.
- Add metadata to each chunk:
  - `audio_uuid`
  - `chunk_index`
  - `total_chunks`
  - `original_filename`
- Implement organized directory structure:
  - Temporary chunks: `temp/uploading/{uuid}/id[{uuid}]_ch[{idx}].part`
  - Merged file: `uploads/{uuid}/id[{uuid}]_merged.{ext}`
  - Split files: `uploads/splits/{uuid}/id[{uuid}]_split[{idx}].{ext}`
- Implement logic to **reassemble** chunks on the back end.
- Implement Voice Activity Detection (VAD) for intelligent audio splitting.
- Process audio segments in parallel for faster transcription.
- Implement detailed progress tracking with percentage thresholds:
  - Upload Phase (0-30%)
  - Processing Phase (30-50%)
  - Transcription Phase (50-80%)
  - Diarization Phase (80-95%)
  - Completion (100%)

### Deliverables
- The UI can split and upload large files.
- The back end correctly reconstructs full audio files.
- The back end splits audio using VAD and processes segments in parallel.
- The UI displays detailed progress information.

### Success Criteria
- Large files are uploaded in chunks and reassembled.
- Audio is intelligently split using VAD.
- Audio segments are processed in parallel.
- Progress is accurately tracked and displayed.

### Sub-stages

#### Stage 5.1: Basic Chunking Infrastructure
1. Frontend Tasks:
   - Implement basic audio chunking (~2MB)
   - Add chunk metadata (uuid, index, total_chunks, original_filename)
   - Split HTML, CSS, and JS into separate files for better maintainability
   - Implement animated progress bar with detailed status messages

2. Backend Tasks:
   - Create AnalysisWork class with proper chunk tracking
   - Implement organized directory structure for file storage
   - Implement chunk merging logic
   - Add comprehensive logging with traceback information

#### Stage 5.2: Multi-Chunk Processing
1. Frontend Tasks:
   - Enable multiple chunk transmission
   - Implement parallel POST requests
   - Add detailed progress tracking with status messages
   - Implement robust error handling and retry mechanisms

2. Backend Tasks:
   - Implement Voice Activity Detection (VAD) for intelligent audio splitting
   - Process audio segments in parallel for faster transcription
   - Implement detailed progress tracking with percentage thresholds
   - Add graceful fallback to time-based splitting when VAD fails

### Testing Requirements
1. Single Chunk:
   - Verify correct metadata handling
   - Confirm STT processing
   - Check result storage

2. Multiple Chunks:
   - Test parallel uploads
   - Verify chunk ordering
   - Validate complete result assembly
   - Test VAD-based splitting
   - Verify parallel processing of segments

---

## Stage 6: Speaker Diarization (Basic)

### Objective
- Implement basic speaker diarization using **GPT-4o-mini**.

### Tasks
- Analyze transcriptions for speaker roles.
- Assign speakers (`0` = counselor, `1` = client, `2+` = others).
- Return structured transcript data to the front end.

### Deliverables
- The UI displays speaker-labeled transcripts.

### Success Criteria
- Transcriptions are grouped by speaker.

---

## Stage 7: Speaker Diarization (Advanced)

### Objective
- Improve diarization accuracy with **audio-based** methods.

### Tasks
- Research and integrate a dedicated **speaker diarization** library.
- Improve Whisper's diarization by combining text and audio analysis.

### Deliverables
- Improved accuracy in speaker identification.

### Success Criteria
- Speaker labels are more accurate than in **Stage 6**.

---

## Stage 8: Performance Optimization

### Objective
- Reduce processing time and improve responsiveness.

### Tasks
- Implement **background tasks** for STT processing.
- Optimize Whisper's inference speed (e.g., GPU acceleration).
- Further optimize parallel processing of audio segments.

### Deliverables
- Faster audio analysis.

### Success Criteria
- Reduced processing time compared to **Stage 4**.

---

## Stage 9: Future Enhancements

### Objective
- Prepare the system for future scalability and advanced features.

### Possible Enhancements
- Emotional analysis (e.g., ElevenLabs integration).
- Summarization of transcribed conversations.
- Cloud deployment options.
- Database integration for persistent storage.

### Success Criteria
- System is extensible and stable.

---

## Completed Enhancements

### Frontend Improvements
- Split HTML, CSS, and JS into separate files for better maintainability
- Implemented animated progress bar with detailed status messages
- Added robust error handling and retry mechanisms
- Improved user feedback with detailed progress information

### Backend Improvements
- Implemented organized directory structure for file storage:
  - Temporary chunks: `temp/uploading/{uuid}/id[{uuid}]_ch[{idx}].part`
  - Merged file: `uploads/{uuid}/id[{uuid}]_merged.{ext}`
  - Split files: `uploads/splits/{uuid}/id[{uuid}]_split[{idx}].{ext}`
- Added Voice Activity Detection (VAD) for intelligent audio splitting
- Implemented parallel processing of audio segments for faster transcription
- Added detailed progress tracking with percentage thresholds
- Implemented comprehensive logging with traceback information
- Added graceful fallback mechanisms when primary methods fail

---

## Stage 10: Multi-Pass Results API (v2)

## Description

In this stage, we'll enhance the user experience by delivering intermediate results as they become available. This creates a more responsive feeling application, as users will see transcriptions earlier in the process before speaker identification is complete.

## Features

1. New API v2 endpoints that provide step-specific results
2. Display transcription immediately when available (before speaker identification)
3. Update the transcript with speaker information when diarization is complete
4. Progressively enhance the user interface as more processing steps complete

## Technical Approach

1. Create a new API route `/api/v2/upload/result/{step_name}/{uuid}` that returns the results of a specific processing step
2. Store intermediate results for each completed step in the AnalysisWork object
3. Update the client to poll for completed steps and request their intermediate results
4. Implement progressive display of results on the client side:
   - Initial display: Transcription with "undecided" speakers
   - Final display: Complete transcript with proper speaker identification

## Benefits

- Improved perceived performance: Users see results faster
- More interactive experience: The application feels more responsive
- Better feedback: Users can start reviewing content before processing is complete

---

**End of stage.md**
