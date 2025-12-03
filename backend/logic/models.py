from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union, Literal
from enum import Enum
from datetime import datetime
import os
import backend.config as config
import json


class AudioStatus(str, Enum):
    """Status of audio processing."""

    PENDING = "pending"
    UPLOADING = "uploading"
    SPLITTING = "splitting"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    IMPROVING = "improving"
    COMPLETING = "completing"
    FAILED = "failed"


class ChunkInfo(BaseModel):
    """Information about a chunk of audio data."""

    index: int
    file_path: str
    size: int
    status: AudioStatus = AudioStatus.PENDING
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class Speaker(BaseModel):
    """Speaker information."""

    id: int
    role: Optional[str] = None  # e.g., "counselor", "client", etc.


class Segment(BaseModel):
    """A segment of transcribed audio."""

    id: int
    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str
    speaker: Optional[Union[int, Literal["undecided"]]] = None  # Speaker ID
    speaker_diarized: Optional[str] = None  # Speaker ID

    def __str__(self):
        return f"Segment(id={self.id}, start={self.start}, end={self.end}, text={self.text}, speaker={self.speaker}, speaker_diarized={self.speaker_diarized})"


class TranscriptionResult(BaseModel):
    """Result of audio transcription and diarization."""

    segments: List[Segment] = []
    speakers: List[Speaker] = []
    text: str = ""  # Full text


class AnalysisWork:
    """Class to track and manage audio analysis work."""

    def __init__(self, id: str, filename: str, total_chunks: int, status: AudioStatus):
        self.id = id
        self.filename = filename
        self.total_chunks = total_chunks
        self.status = status
        self.error = None
        self.chunks = {}  # Dict[int, ChunkInfo]
        self.file_path = None
        self.split_paths = []
        self.result = None
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.audio_uuid = id
        self.steps = []
        # Add options with default values
        self.options = {
            "diarization_method": "",
            "is_limit_time": True,
            "limit_time_sec": 600,
            "is_merge_segments": True,
        }
        # Add storage for intermediate results
        self.step_results = {
            "transcription": None,
            "diarization": None,
            "improving": None,
        }
        self.update_step(
            {
                "step_name": "uploading",
                "status": "in_progress",
                "total_chunks": total_chunks,
                "processed_chunks": 0,
            }
        )

        # New attributes for chunk management
        self.chunk_dir = config.TEMP_DIR / "uploading" / id
        self.chunk_dir.mkdir(exist_ok=True, parents=True)

    def update_step(self, step_info: dict) -> None:
        """Add a new step to the progress tracking or update existing one"""
        step_info["timestamp"] = datetime.now().isoformat()

        # Normalize status to one of the three allowed values
        if "status" in step_info:
            # If status is an AudioStatus enum, convert it to "in_progress"
            if isinstance(step_info["status"], AudioStatus):
                # Convert AudioStatus to the proper status string
                step_info["status"] = "in_progress"

        # Never use status as a step_name (this is the problem)
        # Instead, we need to have a valid step_name or reject the update
        if "step_name" not in step_info:
            # If we don't have a valid step name, log an error and add a default
            print(f"WARNING: Missing step_name in update_step: {step_info}")
            # DO NOT copy status to step_name
            # Instead, use the current job status to infer the step name
            step_info["step_name"] = (
                self.status.name.lower() if self.status else "unknown"
            )

        # Find existing step with the same step_name
        for step in self.steps:
            # Fix any existing steps that have status as step_name
            if step.get("step_name") == "in_progress":
                # Replace with current job status
                step["step_name"] = (
                    self.status.name.lower() if self.status else "unknown"
                )

            if step.get("step_name") == step_info.get("step_name"):
                # Update existing step
                step.update(step_info)
                return

        # If no matching step found, append new one
        self.steps.append(step_info)

    def update_current_step(self, update_dict: dict) -> None:
        """Update the latest step with new information"""
        if self.steps:
            self.steps[-1].update(update_dict)

    def add_chunk(self, chunk_index: int, chunk_path: str, chunk_size: int) -> None:
        """
        Add a new chunk to the analysis job.

        Args:
            chunk_index: Index of the chunk
            chunk_path: Path to the chunk file
            chunk_size: Size of the chunk in bytes
        """
        self.chunks[chunk_index] = ChunkInfo(
            index=chunk_index,
            file_path=chunk_path,
            size=chunk_size,
            status=AudioStatus.PENDING,
        )
        self.updated_at = datetime.now()
        self.update_current_step({"processed_chunks": len(self.chunks)})

    def get_chunk(self, chunk_index: int) -> Optional[ChunkInfo]:
        """
        Get information about a specific chunk.

        Args:
            chunk_index: Index of the chunk

        Returns:
            ChunkInfo or None if the chunk doesn't exist
        """
        return self.chunks.get(chunk_index)

    def update_chunk_status(
        self, chunk_index: int, status: AudioStatus, error: Optional[str] = None
    ) -> None:
        """
        Update the status of a specific chunk.

        Args:
            chunk_index: Index of the chunk
            status: New status
            error: Optional error message
        """
        if chunk_index in self.chunks:
            self.chunks[chunk_index].status = status
            if error:
                self.chunks[chunk_index].error = error
            self.updated_at = datetime.now()

    def update_chunk_result(self, chunk_index: int, result: Dict[str, Any]) -> None:
        """
        Update the result of a specific chunk.

        Args:
            chunk_index: Index of the chunk
            result: Result data
        """
        if chunk_index in self.chunks:
            self.chunks[chunk_index].result = result
            self.updated_at = datetime.now()

    def all_chunks_received(self) -> bool:
        """
        Check if all chunks have been received.

        Returns:
            bool: True if all chunks have been received
        """
        return len(self.chunks) == self.total_chunks

    def all_chunks_processed(self) -> bool:
        """
        Check if all chunks have been processed.

        Returns:
            bool: True if all chunks have been processed
        """
        return all(
            chunk.status == AudioStatus.COMPLETING for chunk in self.chunks.values()
        )

    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            # Remove all chunk files
            for chunk in self.chunks.values():
                try:
                    if os.path.exists(chunk.file_path):
                        os.unlink(chunk.file_path)
                except Exception as e:
                    print(f"Error removing chunk file {chunk.file_path}: {e}")

            # Remove chunk directory
            if os.path.exists(self.chunk_dir):
                os.rmdir(self.chunk_dir)

            # Remove final file if it exists
            if self.file_path and os.path.exists(self.file_path):
                os.unlink(self.file_path)
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def update_status(self, status: AudioStatus) -> None:
        """Update the status and add a new step"""
        self.status = status  # This is fine as the overall job status
        self.updated_at = datetime.now()

        # Get a readable name from the status enum
        step_name = status.name.lower()  # Convert SPLITTING to "splitting", etc.

        # Add appropriate step based on status - consistently use string values for status
        if status == AudioStatus.SPLITTING:
            self.update_step({"step_name": step_name, "status": "in_progress"})
        elif status == AudioStatus.DIARIZING:
            self.update_step({"step_name": step_name, "status": "in_progress"})
        elif status == AudioStatus.TRANSCRIBING:
            self.update_step(
                {
                    "step_name": step_name,
                    "status": "in_progress",
                    "total_splits": len(self.split_paths) if self.split_paths else 0,
                    "processed_splits": 0,
                }
            )
        elif status == AudioStatus.IMPROVING:
            self.update_step({"step_name": step_name, "status": "in_progress"})
        elif status == AudioStatus.COMPLETING:
            # Mark the current step as completed first
            self.complete_current_step()
            # Then add a completed step
            self.update_step({"step_name": step_name, "status": "completed"})
        elif status == AudioStatus.FAILED:
            self.update_step(
                {"step_name": step_name, "status": "failed", "error": self.error}
            )
        elif status == AudioStatus.UPLOADING:
            self.update_step(
                {
                    "step_name": step_name,
                    "status": "in_progress",
                    "total_chunks": self.total_chunks,
                    "processed_chunks": len(self.chunks),
                }
            )

        # Save job state
        save_analysis_work_to_json(self)

    def update_options(self, options: Dict):
        """Update analysis options."""
        if options:
            self.options.update(options)
            self.updated_at = datetime.now()

    def store_step_result(self, step_name: str, result: Dict[str, Any]) -> None:
        """
        Store intermediate result for a specific processing step.

        Args:
            step_name: Name of the processing step
            result: Result data
        """
        if step_name in self.step_results:
            self.step_results[step_name] = result
            self.updated_at = datetime.now()

    def get_step_result(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the result for a specific processing step.

        Args:
            step_name: Name of the processing step

        Returns:
            Dict[str, Any] or None if the result doesn't exist
        """
        # For 'improving' step, return the final result
        if step_name == "improving":
            if self.result:
                return self.result.dict()
            return None

        # For other steps, return from step_results if available
        if step_name in self.step_results:
            return self.step_results[step_name]

        return None

    def complete_current_step(self) -> None:
        """Mark the current step as completed"""
        # Find the current in-progress step and mark it as completed
        for step in reversed(self.steps):  # Check from newest to oldest
            if step.get("status") == "in_progress":
                step["status"] = "completed"
                step["timestamp"] = datetime.now().isoformat()
                return

    def complete_step(self, step_name: str) -> None:
        """Mark a specific step as completed"""
        for step in self.steps:
            if (
                step.get("step_name") == step_name
                and step.get("status") == "in_progress"
            ):
                step["status"] = "completed"
                step["timestamp"] = datetime.now().isoformat()
                return


# Dictionary to store analysis jobs
analysis_jobs: Dict[str, AnalysisWork] = {}


def save_analysis_work_to_json(job: AnalysisWork, save_path: str = None) -> None:
    """
    Save an analysis job to a JSON file.

    Args:
        job: The analysis job to save
    """
    # Create directory if it doesn't exist
    save_dir = config.DATA_DIR / "saved_jobs"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create a serializable dict
    job_dict = {
        "id": job.id,
        "filename": job.filename,
        "total_chunks": job.total_chunks,
        "status": job.status.value,
        "error": job.error,
        "file_path": job.file_path,
        "split_paths": job.split_paths,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "options": job.options,
        "steps": job.steps,
        "step_results": job.step_results,
    }

    # If result exists, save it
    if job.result:
        job_dict["result"] = job.result.dict()

    # Save to file
    if save_path is None:
        file_path = save_dir / f"{job.id}.json"
    else:
        file_path = save_path
    with open(file_path, "w") as f:
        json.dump(job_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved analysis job {job.id} to {file_path}")


def load_analysis_work_from_json(job_id: str) -> AnalysisWork:
    """
    Load an analysis job from a JSON file.

    Args:
        job_id: ID of the job to load

    Returns:
        AnalysisWork: The loaded job or None if it doesn't exist
    """
    file_path = config.DATA_DIR / "saved_jobs" / f"{job_id}.json"

    if not file_path.exists():
        return None

    try:
        with open(file_path, "r") as f:
            job_dict = json.load(f)

        # Create a new job
        job = AnalysisWork(
            id=job_dict["id"],
            filename=job_dict["filename"],
            total_chunks=job_dict["total_chunks"],
            status=AudioStatus(job_dict["status"]),
        )

        # Restore attributes
        job.error = job_dict.get("error")
        job.file_path = job_dict.get("file_path")
        job.split_paths = job_dict.get("split_paths", [])
        job.created_at = datetime.fromisoformat(job_dict.get("created_at"))
        job.updated_at = datetime.fromisoformat(job_dict.get("updated_at"))
        job.options = job_dict.get("options", {})
        job.steps = job_dict.get("steps", [])
        job.step_results = job_dict.get("step_results", {})

        # Restore result if it exists
        if "result" in job_dict and job_dict["result"]:
            job.result = TranscriptionResult(**job_dict["result"])

        print(f"Loaded analysis job {job_id} from {file_path}")
        return job
    except Exception as e:
        print(f"Error loading analysis job {job_id}: {e}")
        return None


def load_all_saved_jobs() -> Dict[str, AnalysisWork]:
    """
    Load all saved jobs from disk.

    Returns:
        Dict[str, AnalysisWork]: Dictionary of job ID to job
    """
    jobs = {}
    save_dir = config.DATA_DIR / "saved_jobs"

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        return jobs

    for file_path in save_dir.glob("*.json"):
        job_id = file_path.stem
        job = load_analysis_work_from_json(job_id)
        if job:
            jobs[job_id] = job

    print(f"Loaded {len(jobs)} saved jobs")
    return jobs


# Make sure these methods are called when updating jobs
def update_analysis_job(job: AnalysisWork) -> None:
    """
    Update an analysis job and save it to disk.

    Args:
        job: The job to update
    """
    analysis_jobs[job.id] = job
    save_analysis_work_to_json(job)
