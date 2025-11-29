import asyncio
import threading
import traceback
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    status,
    Query,
    Path as FastAPIPath,
    Body,
)
from typing import Dict, Any, Optional
import sys
import os
import json
from fastapi.responses import FileResponse
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logic.models import (
    AnalysisWork,
    AudioStatus,
    analysis_jobs,
    load_analysis_work_from_json,
    save_analysis_work_to_json,
    get_saved_job_ids,
    get_job_metadata,
)
from logic.voice_analysis.process import process_audio_file
import config
from pathlib import Path
import shutil
from util.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/upload",
    tags=["Upload"],
)


# Helper function to save uploaded file
async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """
    Save an uploaded file to the specified destination.

    Args:
        upload_file: The uploaded file
        destination: The destination path

    Returns:
        Path: The path to the saved file
    """
    try:
        # Ensure the directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Save the file
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        return destination
    except Exception as e:
        err_msg = f"ERROR in save_upload_file: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise
    finally:
        upload_file.file.close()


def merge_chunks_and_process(job, total_chunks, file_ext, audio_uuid, background_tasks):
    """Merge uploaded chunks and start audio processing."""
    logger.info(f"Merging chunks for {audio_uuid}")
    try:
        # Create directory for the merged file
        merged_dir = config.UPLOADS_DIR / str(audio_uuid)
        merged_dir.mkdir(parents=True, exist_ok=True)

        # Create merged file path using original file extension
        merged_filename = f"id[{audio_uuid}]_merged.{file_ext}"
        merged_path = merged_dir / merged_filename

        # Merge chunks in order
        with open(merged_path, "wb") as outfile:
            for i in range(total_chunks):
                chunk = job.get_chunk(i)
                if chunk is None:
                    raise ValueError(f"Missing chunk {i}")
                with open(chunk.file_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)

        # Mark uploading as completed and update job
        job.complete_step("uploading")
        job.file_path = str(merged_path)
        save_analysis_work_to_json(job)

        # Process the file in the background
        asyncio.run(process_audio_file(job, background_tasks))

        logger.info(f"All chunks merged and processing started for {audio_uuid}")

    except Exception as merge_error:
        logger.error(f"Error merging chunks for {audio_uuid}: {merge_error}")
        job.error = str(merge_error)
        job.update_status(AudioStatus.FAILED)
        job.cleanup()


# Implement the upload_chunk endpoint for Stage 5
@router.post(
    "/chunk",
    summary="Upload Audio Chunk",
    description="""
    Upload a chunk of an audio file for processing.
    
    This endpoint accepts a chunk of an audio file along with metadata about the chunk.
    The server will store the chunk and, when all chunks are received, will assemble them
    into a complete file for processing.
    
    - The `audio_uuid` should be a unique identifier for the audio file.
    - The `chunk_index` is the 0-based index of the current chunk.
    - The `total_chunks` is the total number of chunks that will be uploaded.
    - The `options` is a JSON string containing processing options.
    
    The server will respond with the status of the upload and the number of chunks received so far.
    """,
    status_code=status.HTTP_202_ACCEPTED,
    response_description="Chunk upload status",
)
async def upload_chunk(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The audio file chunk to upload"),
    audio_uuid: str = Form(..., description="Unique identifier for the audio file"),
    chunk_index: int = Form(..., description="Index of the current chunk (0-based)"),
    total_chunks: int = Form(..., description="Total number of chunks"),
    original_filename: str = Form(
        ..., description="Original filename of the complete file"
    ),
    options: str = Form(None, description="JSON string containing processing options"),
) -> Dict[str, Any]:
    """
    Upload a chunk of an audio file for processing.

    Args:
        background_tasks: FastAPI background tasks
        file: The audio file chunk
        audio_uuid: Unique identifier for the audio file
        chunk_index: Index of the current chunk (0-based)
        total_chunks: Total number of chunks
        original_filename: Original filename of the complete file
        options: JSON string containing processing options

    Returns:
        Dict[str, Any]: Status of the chunk upload
    """
    # Check if chunk_index is valid
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chunk_index {chunk_index}, must be between 0 and {total_chunks - 1}",
        )

    processing_options = None
    if options:
        try:
            processing_options = json.loads(options)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid options JSON: {str(e)}",
            )

    # Get or create the analysis job
    job = analysis_jobs.get(audio_uuid)
    if job is None:
        # Create a new analysis job
        job = AnalysisWork(
            id=audio_uuid,
            filename=original_filename,
            total_chunks=total_chunks,
            status=AudioStatus.UPLOADING,
        )
        # Update options if provided
        if processing_options:
            job.update_options(processing_options)
        analysis_jobs[audio_uuid] = job

        save_analysis_work_to_json(job)
    elif job.total_chunks != total_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mismatched total_chunks: expected {job.total_chunks}, got {total_chunks}",
        )
    elif chunk_index in job.chunks:
        # Check if this chunk was already uploaded
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chunk {chunk_index} was already uploaded",
        )

    try:
        # Create a directory for this UUID's chunks
        uuid_dir = config.TEMP_DIR / "uploading" / str(audio_uuid)
        uuid_dir.mkdir(parents=True, exist_ok=True)

        # Get file extension from original filename
        file_ext = (
            os.path.splitext(original_filename)[1][1:] if original_filename else "bin"
        )

        # Create chunk filename with specified format
        chunk_filename = f"id[{audio_uuid}]_ch[{chunk_index}].part"
        chunk_path = uuid_dir / chunk_filename

        # Save the chunk
        await save_upload_file(file, chunk_path)

        # Add the chunk to the job
        job.add_chunk(chunk_index, str(chunk_path), os.path.getsize(chunk_path))

        # Save job state after adding chunk
        save_analysis_work_to_json(job)

        # If all chunks are received, merge them and start processing
        if job.all_chunks_received():
            merge_thread = threading.Thread(
                target=merge_chunks_and_process,
                args=(job, total_chunks, file_ext, audio_uuid, background_tasks),
                daemon=True,  # Daemonize thread if desired
            )
            merge_thread.start()
            return {
                "message": "All chunks received, merging started in background",
                "status": job.status,
                "audio_uuid": audio_uuid,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "received_chunks": len(job.chunks),
            }

        # Return the job status
        return {
            "message": "Chunk received",
            "status": job.status,
            "audio_uuid": audio_uuid,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "received_chunks": len(job.chunks),
        }
    except Exception as e:
        err_msg = (
            f"ERROR in upload_chunk: {e}\n with traceback:\n{traceback.format_exc()}"
        )
        logger.error(err_msg)

        # Update status to failed
        job.error = str(e)
        job.update_status(AudioStatus.FAILED)

        # Clean up chunks
        job.cleanup()

        # Remove the job from the in-memory storage
        analysis_jobs.pop(audio_uuid, None)

        # Raise an exception
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}",
        )


# Implement the get_status endpoint for Stage 3
@router.get(
    "/status/{audio_uuid}",
    summary="Get Upload Status",
    description="""
    Get the status of an audio file upload and processing.
    
    This endpoint returns the current status of an audio file that has been uploaded
    for processing. The status can be one of the following:
    
    - `pending`: The file is waiting to be processed.
    - `uploading`: The file is being uploaded (chunks are being received).
    - `processing`: The file is being processed.
    - `transcribing`: The file is being transcribed.
    - `improving`: Transcript improvement is being performed.
    - `completed`: Processing is complete.
    - `failed`: Processing failed.
    
    The response includes information about the number of chunks received and the
    total number of chunks expected.
    """,
    response_description="Upload and processing status",
)
async def get_status(
    audio_uuid: str = FastAPIPath(
        ..., description="Unique identifier for the audio file"
    )
) -> Dict[str, Any]:
    """
    Get the status of an audio file upload and processing.

    Args:
        audio_uuid: Unique identifier for the audio file

    Returns:
        Dict[str, Any]: Status of the upload and processing
    """
    try:
        job = load_analysis_work_from_json(audio_uuid)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis job found for audio_uuid {audio_uuid}",
            )

        # Create a safe copy of job data to avoid race conditions during reprocessing
        job_snapshot = {
            "audio_uuid": audio_uuid,
            "status": job.status,
            "progress": {
                "status": job.status,
                "received_chunks": len(job.chunks),
                "total_chunks": job.total_chunks,
                "percent_complete": 0,
                "steps": copy.deepcopy(
                    job.steps
                ),  # Use deepcopy to avoid modification during reading
            },
            "options": copy.deepcopy(job.options),
            "file_path": job.file_path,
            "error": job.error if job.status == AudioStatus.FAILED else None,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
        }

        # Calculate progress percentage
        progress_percent = 0
        if job.total_chunks > 0:
            # If uploading, progress is based on chunks received
            if job.status == AudioStatus.UPLOADING:
                progress_percent = int((len(job.chunks) / job.total_chunks) * 100)
            # If processing, progress is estimated based on status
            elif job.status == AudioStatus.SPLITTING:
                progress_percent = 40
            elif job.status == AudioStatus.DIARIZING:
                progress_percent = 60
            elif job.status == AudioStatus.TRANSCRIBING:
                progress_percent = 80
            elif job.status == AudioStatus.IMPROVING:
                progress_percent = 90
            elif job.status == AudioStatus.COMPLETING:
                progress_percent = 100

        job_snapshot["progress"]["percent_complete"] = progress_percent

        # Return the job status snapshot
        return job_snapshot

    except Exception as e:
        logger.error(f"Error in get_status for {audio_uuid}: {e}")
        # Return a minimal response in case of error
        return {
            "audio_uuid": audio_uuid,
            "status": "pending",
            "progress": {"status": "pending", "steps": [], "percent_complete": 0},
            "error": f"Error retrieving status: {str(e)}",
        }


# Implement the get_result endpoint for Stage 3
@router.get(
    "/result/{audio_uuid}",
    summary="Get Analysis Result",
    description="""
    Get the result of audio analysis.
    
    This endpoint returns the result of the audio analysis, including the transcription
    and speaker diarization. The result includes:
    
    - The full text transcription.
    - Segments of the transcription with start and end times.
    - Speaker information for each segment.
    
    The result is only available if the status of the audio file is `completed`.
    """,
    response_description="Analysis result",
)
async def get_result(
    audio_uuid: str = FastAPIPath(
        ..., description="Unique identifier for the audio file"
    )
) -> Dict[str, Any]:
    """
    Get the result of audio analysis.

    Args:
        audio_uuid: Unique identifier for the audio file

    Returns:
        Dict[str, Any]: Result of the analysis
    """
    # Check if the analysis job exists in memory
    job = analysis_jobs.get(audio_uuid)

    # If not in memory, try to load from disk
    if job is None:
        job = load_analysis_work_from_json(audio_uuid)
        if job:
            # Add to memory
            analysis_jobs[audio_uuid] = job
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis job found for audio_uuid {audio_uuid}",
            )

    # Check if the job is completed
    if job.status != AudioStatus.COMPLETING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis job is not completed (current status: {job.status})",
        )

    # Check if the result exists, load from disk if needed
    if not job.result:
        # Try to load result from disk (lazy loading)
        if not job.load_result_from_disk():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Analysis job is completed but no result is available",
            )

    # Return the result
    return {"audio_uuid": audio_uuid, "status": job.status, "result": job.result.dict()}


@router.get(
    "/result/{step_name}/{audio_uuid}",
    summary="Get Step Result",
    description="""
    Get the intermediate result for a specific processing step.
    
    This endpoint returns the result of a specific processing step for an audio file.
    The step_name can be one of:
    
    - 'transcription': Returns transcript without speaker information
    - 'diarization': Returns diarization information only
    - 'improving': Returns improved transcript with speaker identification
    
    The result is only available if the step has been completed.
    """,
    response_description="Step result",
)
async def get_step_result(
    step_name: str = FastAPIPath(..., description="Name of the processing step"),
    audio_uuid: str = FastAPIPath(
        ..., description="Unique identifier for the audio file"
    ),
) -> Dict[str, Any]:
    """
    Get intermediate results for a specific processing step.

    Args:
        step_name: Name of the processing step ('transcription', 'diarization', or 'improving')
        audio_uuid: Unique identifier for the audio file

    Returns:
        Dict[str, Any]: Result of the processing step
    """
    # Check if valid step name
    valid_steps = ["transcription", "diarization", "improving"]
    if step_name not in valid_steps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid step name: {step_name}. Valid values are: {', '.join(valid_steps)}",
        )

    # Check if the job exists in memory
    job = analysis_jobs.get(audio_uuid)

    # If not in memory, try to load from disk
    if job is None:
        job = load_analysis_work_from_json(audio_uuid)
        if job:
            # Add to memory
            analysis_jobs[audio_uuid] = job
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis job found for audio_uuid {audio_uuid}",
            )

    result = None
    # For improving step, use the final result if available
    if step_name == "improving":
        if job.result:
            result = job.result.dict()
        elif job.status == AudioStatus.COMPLETING:
            # Try to load result from disk for completed jobs
            if job.load_result_from_disk():
                result = job.result.dict()
    else:
        # For other steps, get the step-specific result
        result = job.get_step_result(step_name)

    if not result:
        # Check if the step has been completed based on job status
        if step_name == "transcription" and job.status.value not in [
            "transcribing",
            "improving",
            "completing",
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Step '{step_name}' has not been completed yet",
            )
        elif step_name == "diarization" and job.status.value not in [
            "improving",
            "completing",
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Step '{step_name}' has not been completed yet",
            )
        elif step_name == "improving" and job.status.value not in [
            "improving",
            "completing",
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Step '{step_name}' has not been completed yet",
            )

        # If the step should be completed but no result, return empty result
        if step_name == "transcription":
            result = {"segments": []}
        elif step_name == "diarization":
            result = {"speakers": [], "segments": []}
        elif step_name == "improving":
            result = {"segments": []}

    return {"audio_uuid": audio_uuid, "step": step_name, "result": result}


@router.get(
    "/audio/{audio_uuid}",
    summary="Get Audio File",
    description="Get the audio file for a specific analysis job",
    response_class=FileResponse,
)
async def get_audio_file(
    audio_uuid: str = FastAPIPath(
        ..., description="Unique identifier for the audio file"
    )
):
    """Get the audio file for playback"""
    # Check if the job exists in memory
    job = analysis_jobs.get(audio_uuid)

    # If not in memory, try to load from disk
    if job is None:
        job = load_analysis_work_from_json(audio_uuid)
        if job:
            # Add to memory
            analysis_jobs[audio_uuid] = job
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis job found for audio_uuid {audio_uuid}",
            )

    if not job.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found"
        )

    return FileResponse(
        job.file_path, media_type="audio/wav", filename=f"audio_{audio_uuid}.wav"
    )


@router.post(
    "/reprocess/{audio_uuid}",
    summary="Reprocess Audio",
    description="Restart processing for an existing audio analysis job",
    status_code=status.HTTP_202_ACCEPTED,
    response_description="Reprocessing status",
)
async def reprocess_audio(
    background_tasks: BackgroundTasks,
    audio_uuid: str = FastAPIPath(
        ..., description="Unique identifier for the audio file"
    ),
    request_data: Optional[Dict[str, Any]] = Body({}, description="Processing options"),
) -> Dict[str, Any]:
    """
    Restart processing for an existing audio analysis job.

    Args:
        background_tasks: FastAPI background tasks
        audio_uuid: Unique identifier for the audio file
        request_data: Optional processing options

    Returns:
        Dict[str, Any]: Status of the reprocessing request
    """
    # Check if the job exists in memory
    job = analysis_jobs.get(audio_uuid)

    # If not in memory, try to load from disk
    if job is None:
        logger.info(
            f"Reprocessing job: {audio_uuid} not found in memory, loading from disk"
        )
        job = load_analysis_work_from_json(audio_uuid)
        if job:
            # Add to memory
            analysis_jobs[audio_uuid] = job
            logger.info(f"Reprocessing job: {audio_uuid} loaded from disk")
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analysis job found for audio_uuid {audio_uuid}",
            )

    # Check if the file exists
    if not job.file_path or not os.path.exists(job.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found"
        )

    # Update options if provided
    if request_data and "options" in request_data:
        job.update_options(request_data["options"])

    try:
        # Reset job state for reprocessing
        job.status = AudioStatus.SPLITTING
        job.error = None
        job.result = None
        job.steps = []  # Clear previous steps
        job.step_results = {
            "transcription": None,
            "diarization": None,
            "improving": None,
        }

        # Add initial step
        job.update_step({"step_name": "splitting", "status": "in_progress"})

        logger.info(f"Reprocessing job: {job}")
        logger.info(f"Reprocessing job: {job.steps}")

        # Save the updated job state
        save_analysis_work_to_json(job)

        # Return success response immediately
        response = {
            "message": "Reprocessing started",
            "audio_uuid": audio_uuid,
            "status": job.status,
        }

        # Start processing in the background (after response is sent)
        background_tasks.add_task(process_audio_file, job, background_tasks)

        return response

    except Exception as e:
        logger.error(f"Error preparing job for reprocessing: {e}")
        # Mark job as failed if we couldn't prepare it for reprocessing
        job.error = str(e)
        job.update_status(AudioStatus.FAILED)
        save_analysis_work_to_json(job)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reprocessing audio: {str(e)}",
        )


@router.get(
    "/jobs",
    summary="List All Jobs",
    description="Get a list of all saved analysis jobs with their metadata",
    response_description="List of jobs",
)
async def list_jobs() -> Dict[str, Any]:
    """
    Get a list of all saved analysis jobs.

    Returns:
        Dict[str, Any]: List of job metadata
    """
    try:
        job_ids = get_saved_job_ids()
        jobs_metadata = []

        for job_id in job_ids:
            metadata = get_job_metadata(job_id)
            if metadata:
                jobs_metadata.append(metadata)

        return {"total_jobs": len(jobs_metadata), "jobs": jobs_metadata}

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing jobs: {str(e)}",
        )
