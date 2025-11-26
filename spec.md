# Project Specification

This document outlines the requirements and specifications for a local server-based voice analysis solution designed to replace (or supplement) the current Supabase edge-function approach. The goal is to provide faster and more accurate speaker analysis while maintaining an efficient development process. Below is an overview of the project, its functional and non-functional requirements, as well as the architecture and technology stack.

---

## 1. Summary

- **Objective:** Build a local server solution for voice analysis and diarization (speaker separation).
- **Motivation:** The existing solution is slow and has low accuracy; thus, a faster and more accurate on-premise solution is needed.
- **Development Process:** The development will be phased into multiple stages (at least five). Each stage must deliver a deployable version of the project and incrementally add functionality.

---

## 2. Functional Requirements

1. **Backend Accessibility**
   - Must be callable from a test web page served as a **static public page** within the FastAPI server.
   - The API endpoints (including Swagger docs) must be available externally for testing.

2. **Front-End User Interface**
   - A **single HTML page** with **Vanilla JS** for testing.
   - The page must allow file uploads (drag & drop or button-based upload).
   - It should display progress while files are being analyzed (progress bar or simple indicator).
   - Analysis results (transcribed dialogue with speaker labels) must be displayed.
   - Frontend code should be organized into separate HTML, CSS, and JS files for maintainability.

3. **Audio Analysis and Speaker Diarization**
   - Implement a Python-based service to:
     - Transcribe audio (using Whisper or similar library).
     - Identify speakers (role classification: e.g., `0` for counselor, `1` for client, `2+` for others).
   - In later stages, improve accuracy with advanced local libraries or third-party tools.
   - Must handle compressed audio (e.g., M4A) by decoding on the front end or server side.
   - Implement Voice Activity Detection (VAD) for intelligent audio splitting.
   - Process audio segments in parallel for faster transcription.

4. **Chunked Upload Workflow**
   - Upload large audio files in chunks (approx. 2MB each).
   - Each chunk is sent via a REST API POST request with metadata:
     - `audio_uuid`
     - `chunk_index`
     - `total_chunks`
     - `original_filename`
   - Back end stores chunks in organized directory structure:
     - Temporary chunks: `temp/uploading/{uuid}/id[{uuid}]_ch[{idx}].part`
     - Merged file: `uploads/{uuid}/id[{uuid}]_merged.{ext}`
     - Split files: `uploads/splits/{uuid}/id[{uuid}]_split[{idx}].{ext}`
   - Back end assembles chunks, performs parallel STT and speaker segmentation, and combines results.

5. **Polling and Status Checks**
   - After uploading all chunks, the front end polls the server with the `audio_uuid` to retrieve analysis status.
   - Status updates include detailed progress information:
     - Current stage (uploading, processing, diarizing, transcribing, improving, completed)
     - Percentage complete with stage-specific thresholds
     - Detailed status message
   - When analysis is complete, final results are returned.

6. **Error Handling and Logging**
   - Comprehensive error handling with detailed error messages.
   - Extensive logging with traceback information for debugging.
   - Graceful fallback mechanisms when primary methods fail.

---

## 3. Non-Functional Requirements

1. **Performance**
   - The solution must be significantly faster than the current Supabase-based approach.
   - Must support concurrency (e.g., background tasks or workers for parallel processing).
   - Implement parallel processing of audio segments for faster transcription.

2. **Accuracy**
   - Must provide high-quality transcription and reliable speaker diarization.
   - Explore the use of advanced voice analysis libraries in future stages.
   - Use Voice Activity Detection (VAD) for more accurate audio segmentation.

3. **Timeline**
   - An initial working prototype is expected as quickly as possible (within days).
   - Subsequent improvements can be made in later stages.

4. **Scalability**
   - While initially targeted for local use, the design should permit scaling or cloud deployment if needed.
   - Directory structure and file organization should support scaling.

5. **Maintainability**
   - The FastAPI server should serve the static test page from a **public** directory.
   - Frontend code should be split into separate HTML, CSS, and JS files.
   - Backend code should be organized into logical components (controllers, models, logic).
   - Comprehensive logging for easier debugging and maintenance.

---

## 4. High-Level Architecture

1. **Front End (Vanilla JS + HTML + CSS)**
   - Separate files for HTML structure, CSS styling, and JS functionality.
   - Handles file uploads, chunking, progress display, and final result presentation.
   - Performs minimal preprocessing (e.g., decoding compressed files if needed).
   - Implements robust error handling and retry mechanisms.

2. **Back End (Python + FastAPI)**
   - Provides REST endpoints for uploading chunks and polling analysis results.
   - Uses a lightweight in-memory model (`AnalysisWork`) to store chunk data, track progress, and combine final results.
   - Integrates with STT and speaker diarization libraries (e.g., Whisper, PyTorch, etc.).
   - Implements Voice Activity Detection (VAD) for intelligent audio splitting.
   - Processes audio segments in parallel for faster transcription.
   - Exposes Swagger documentation for all endpoints.

3. **Workflow**
   - User uploads audio → chunked → posted to API → back end stores chunks → back end merges chunks → back end splits audio using VAD → back end processes splits in parallel → back end combines results → final result is returned on polling.

---

## 5. Technology Stack

- **Front End:**
  - **Vanilla JS**
  - **HTML + CSS**
  - Served as static files from FastAPI

- **Back End:**
  - Python 3.x
  - FastAPI
  - Whisper for speech-to-text
  - WebRTC VAD for Voice Activity Detection
  - PyDub for audio processing
  - (Later stages) Additional speaker-diarization libraries

- **Tools & Infrastructure:**
  - Local machine or any server with Python runtime
  - Possible open port on a personal router
  - CORS configurations to allow front-end to communicate

---

## 6. Deliverables

- **Deployable Application** at each stage:
  - Multiple stages with incremental feature additions.
  - Separate HTML, CSS, and JS files served from FastAPI.
  - Swagger documentation for API endpoints.
  - A working proof-of-concept in the shortest time.

---

## 7. Future Considerations

- **Emotion Analysis**: Potentially integrate advanced libraries for emotional or tone detection in later stages.
- **Authentication/Authorization**: Currently out of scope, can be added if the project needs user account control.
- **DB Integration**: If session storage or historical logs are needed, database usage could be introduced in a future stage.

---

# Audio Chunking and Parallel Processing Specification

## Audio Upload Process
1. Frontend splits audio files into chunks (~2MB each)
   - For compressed formats (m4a, etc.):
     - Use Web Audio API or similar libraries to decompress
     - Generate appropriate headers for each chunk

2. REST API Endpoint
   - Method: POST
   - Required Parameters:
     - audio_uuid: Unique identifier for the complete audio
     - chunk_index: Index of current chunk
     - total_chunks: Total number of chunks
     - original_filename: Original filename of the complete file
     - file: Binary audio chunk data

3. Backend Processing
   - Create/Update AnalysisWork instance:
     - Constructor parameters: audio_uuid, filename, total_chunks
     - Store audio chunks in a dictionary within AnalysisWork
     - Index chunks based on chunk_index
   - Immediate response after chunk storage
   - When all chunks are received:
     - Merge chunks into a complete file
     - Split audio using Voice Activity Detection (VAD)
     - Process each split with Whisper STT independently
     - Combine results with proper timestamp adjustments
   - Store results until client polls for completion

## File Organization
1. Temporary Chunks:
   - Path: `temp/uploading/{uuid}/id[{uuid}]_ch[{idx}].part`

2. Merged File:
   - Path: `uploads/{uuid}/id[{uuid}]_merged.{ext}`

3. Split Files:
   - Path: `uploads/splits/{uuid}/id[{uuid}]_split[{idx}].{ext}`

## Progress Tracking
1. Upload Phase (0-30%):
   - Progress based on chunks received

2. Processing Phase (30-50%):
   - Audio merging and VAD splitting

3. Transcription Phase (50-80%):
   - Progress based on splits transcribed

4. Diarization Phase (80-95%):
   - Speaker identification

5. Completion (100%):
   - Final result assembly

---

# API v2 Specification

## Endpoints

### GET /api/v2/upload/result/{step_name}/{audio_uuid}

Get the intermediate result for a specific processing step.

#### Path Parameters

- `step_name` (string, required): Name of the processing step. Valid values:
  - `transcription`: Returns transcript without speaker information
  - `diarization`: Returns diarization information only
  - `improving`: Returns improved transcript with speaker identification

- `audio_uuid` (string, required): Unique identifier for the audio file

#### Response

- Status Code: 200 OK
- Content Type: application/json
- Body:
  ```json
  {
    "audio_uuid": "string",
    "step": "string",
    "result": {
      // Result object depends on the step_name
    }
  }
  ```

#### Transcription Step Result
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Hello, how are you today?",
      "speaker": "undecided"
    },
    // ...
  ]
}
```

#### Diarization Step Result
```json
{
  "speakers": [
    {
      "id": 0,
      "role": "counselor"
    },
    {
      "id": 1,
      "role": "client"
    }
  ],
  "segments": [
    {
      "start": 0.0, 
      "end": 2.5,
      "speaker": 0
    },
    // ...
  ]
}
```

#### Improving Step Result (Same as Final Result)
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Hello, how are you today?",
      "speaker": 0
    },
    // ...
  ]
}
```

### Error Responses

- **404 Not Found**: Step not found or audio UUID not found
- **400 Bad Request**: Step has not been completed yet

## Implementation Notes

1. The client should continue polling the status endpoint to track overall progress
2. When a step is marked as "completed" in the status response, the client can immediately request the intermediate results for that step
3. The UI should update progressively:
   - After transcription: Show all text with "undecided" speakers
   - After improving: Update the UI with proper speaker identification
4. The multi-pass approach allows users to start reviewing content significantly earlier in the process

---

**End of spec.md**
