import asyncio
import json
import os
import tempfile
import whisper
import torch
import numpy as np
from pydub import AudioSegment
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Any, Tuple
from fastapi import BackgroundTasks
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback
import webrtcvad
import wave
import array
import struct
from pyannote.audio import Pipeline
import torchaudio
import copy
import re
import matplotlib.pyplot as plt
import librosa
import librosa.display
import uuid
from backend.logic.stt_utils import get_improved_lines_with_ts
from backend.logic.models import AnalysisWork, AudioStatus, analysis_jobs, TranscriptionResult, Segment, Speaker
import backend.config as config
from backend.logic.voice_analysis.ai_utils import postprocess_segments, get_openai_client, improve_transcription_lines
from backend.util.logger import get_logger
from fuzzywuzzy import fuzz

# Get logger
logger = get_logger(__name__)

def get_seg_ts_with_diar_analytic(trans_segs_with_ts, diarization_segments):
    """
    - get transcription segments with diarization segments
    """
    pass

def convert_audio_for_whisper(file_path: Path) -> str:
    """
    Convert audio file to format compatible with Whisper (WAV, 16kHz, mono).
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        str: Path to the converted audio file
    """
    try:
        # Create a temporary file for the converted audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()
        
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono
        audio = audio.set_channels(1)
        
        # Convert to 16kHz
        audio = audio.set_frame_rate(16000)
        
        # Export to WAV
        audio.export(temp_path, format="wav")
        
        return temp_path
    except Exception as e:
        err_msg = f"ERROR in convert_audio_for_whisper: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise




async def process_audio_with_assembly_ai(file_path: str, idx_split: int, total_splits: int) -> Dict:
    """
    Process audio file with Assembly AI API.
    
    Args:
        file_path: Path to the audio file
        idx_split: Index of the current split
        total_splits: Total number of splits
        
    Returns:
        Dict: Transcription result
    """
    try:
        # Get analysis job from file path
        audio_uuid = os.path.basename(os.path.dirname(file_path))
        analysis_job = analysis_jobs.get(audio_uuid)
        
        if not analysis_job:
            raise ValueError(f"No analysis job found for {audio_uuid}")

        # Initialize AssemblyAI client
        api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")
        
        import assemblyai as aai
        aai.settings.api_key = api_key
        
        # Configure transcriber with nano model and speaker diarization
        transcription_config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.nano,
            punctuate=True,
            format_text=True,
            speaker_labels=True,  # Enable speaker diarization
            speakers_expected=2,  # Expected number of speakers
            language_code="ko",
            sentiment_analysis=False,  # Enable sentiment analysis -> not available for korean.
        )
        transcriber = aai.Transcriber(config=transcription_config)
        
        # Transcribe the audio file
        transcript = transcriber.transcribe(file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise ValueError(f"AssemblyAI transcription error: {transcript.error}")
        
        # ## save transcript to file
        # # Create the uploads directory if it doesn't exist
        uploads_dir = config.UPLOADS_DIR / audio_uuid
        uploads_dir.mkdir(parents=True, exist_ok=True)
        # # Create the transcript file path
        transcript_path = uploads_dir / f"id[{audio_uuid}]_assemblyai_transcript.json"
        ## TODO: save assemblyai transcript to json file
        # with transcript_path.open("w", encoding="utf-8") as f:
        #     json.dump(transcript, f, default=lambda o: o.__dict__, ensure_ascii=False, indent=2)
        # with transcript_path.open("w", encoding="utf-8") as f:
        #     json.dump(transcript, f, default=custom_json_encoder, ensure_ascii=False, indent=2)

        print("Transcript asemblyai", transcript, f"for file {str(file_path)}")

        if transcript.utterances is None:
            logger.warning(f"No utterances found in transcript for {audio_uuid} for file {str(file_path)} maybe audio too short or empty.")
            return {
                "text": "",
                "segments": []
            }

        
        # Process utterances (speaker segments)
        segments = []
        for i, utterance in enumerate(transcript.utterances):
            segments.append({
                "id": i,
                "start": utterance.start / 1000.0,  # Convert to seconds
                "end": utterance.end / 1000.0,
                "text": utterance.text,
                "speaker": utterance.speaker,
                "confidence": utterance.confidence,
            })
        
        # Get full text
        text = transcript.text
        
        # Handle segments based on diarization method
        # print(f"Diarization method: {analysis_job.options['diarization_method']}")
        # is_do_postprocess = analysis_job.options["diarization_method"] == "stt_apigpt_diar_apigpt"
        
        # For GPT-based splitting, we need to process segments through GPT
        for seg in segments:
            raw_text = seg["text"]
            if raw_text:
                raw_text = raw_text.strip()
            seg["text_raw"] = raw_text
            del seg["text"]
        
        logger.info(f"Postprocessing segments with GPT for filepath: {Path(file_path).name}")
        segments_with_speakers = postprocess_segments(segments)
        text = " ".join([seg["text"] for seg in segments_with_speakers])
        
        return {
            "text": text,
            "segments": segments_with_speakers
        }
        # else:
        #     # For ML-based diarization, we keep the original segments
        #     # and let the diarization step handle speaker assignment
        #     for seg in segments:
        #         seg["speaker"] = -1  # Initialize with -1 for ML diarization

        #     return {
        #         "text": text,
        #         "segments": segments
        #     }

    except Exception as e:
        err_msg = f"ERROR in process_audio_with_assembly_ai: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise ValueError(f"Error transcribing audio with AssemblyAI API: {str(e)}")

async def process_audio_with_openai_api(file_path: str, idx_split: int, total_splits: int, options: dict = None) -> Dict:
    client = get_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
    
    logger.info(f"Transcribing audio file with OpenAI API: {file_path}")
    
    try:
        # Get analysis job from file path
        audio_uuid = os.path.basename(os.path.dirname(file_path))
        analysis_job = analysis_jobs.get(audio_uuid)
        
        if not analysis_job:
            # Create a new analysis job if none exists
            logger.info(f"No analysis job found for {audio_uuid}, creating a new one")
            job_id = audio_uuid  # Use same ID
            analysis_job = AnalysisWork(
                id=job_id,
                filename=os.path.basename(file_path),
                total_chunks=total_splits,
                status=AudioStatus.PENDING
            )
            analysis_job.file_path = file_path
            analysis_job.audio_uuid = audio_uuid
            
            # Use options provided from frontend if available, otherwise use defaults
            if options is not None:
                analysis_job.options = options
            else:
                # Only use defaults if no options were provided
                analysis_job.options = {
                    "diarization_method": "stt_apigpt_diar_mlpyan",
                    "is_limit_time": False,
                    "limit_time_sec": 600,
                    "is_merge_segments": True
                }
                logger.warning(f"No options provided for {audio_uuid}, using defaults")
            
            # Register job with both IDs
            analysis_jobs[job_id] = analysis_job  # Register by ID
            if job_id != audio_uuid:  # Register by audio_uuid if different
                analysis_jobs[audio_uuid] = analysis_job
            
            logger.info(f"Created new analysis job for {audio_uuid} with options: {analysis_job.options}")

        # Open the audio file
        with open(file_path, "rb") as audio_file:
            try:
                # Try the newer API format with word timestamps
                response = client.audio.transcriptions.create(
                    model=config.OPENAI_API_STT_MODEL,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
                
                # Convert each word to a segment
                segments = [
                    {
                        "id": i,
                        "start": word.start,
                        "end": word.end,
                        "text": word.word,
                        "speaker": -1  # Initialize with -1 for ML diarization
                    }
                    for i, word in enumerate(response.words)
                ]
                
                text = response.text
                
            except (AttributeError, TypeError) as e:
                err_msg = f"ERROR in process_audio_with_openai_api (newer API): {e}\n with traceback:\n{traceback.format_exc()}"
                logger.error(err_msg)
                # If the newer API fails, try the older format
                logger.warning(f"Newer API format failed, trying older format: {e}")
                # Reset file pointer
                audio_file.seek(0)
                
                # Try older API format
                response = client.audio.transcribe(
                    model=config.OPENAI_API_STT_MODEL,
                    file=audio_file
                )
                
                # For older API that might not return segments
                if hasattr(response, 'segments'):
                    segments = []
                    for i, segment in enumerate(response.segments):
                        segments.append({
                            "id": i,
                            "start": getattr(segment, 'start', i),
                            "end": getattr(segment, 'end', i+1),
                            "text": segment.text,
                        })
                else:
                    # Create a single segment if no segments are returned
                    text = response.text
                    segments = [{
                        "id": 0,
                        "start": 0.0,
                        "end": 30.0,  # Arbitrary end time
                        "text": text,
                    }]

        # Handle segments based on diarization method
        print(f"Diarization method: {analysis_job.options['diarization_method']}")
        is_do_post1 = analysis_job.options["diarization_method"] in ["stt_apigpt_diar_apigpt", "stt_apigpt_diar_apigpt2"]
        # is_do_post2 = analysis_job.options["diarization_method"] in ["stt_apigpt_diar_mlpyan", "stt_apigpt_diar_apigpt3", "stt_apigpt_diar_mlpyan2"]
        is_do_post2 = analysis_job.options["diarization_method"] in ["stt_apigpt_diar_mlpyan", "stt_apigpt_diar_apigpt3"]
        # is_do_postprocess = False
        if is_do_post1:
            # For GPT-based splitting, we need to process segments through GPT
            for seg in segments:
                raw_text = seg["text"]
                if raw_text:
                    raw_text = raw_text.strip()
                seg["text_raw"] = raw_text
                del seg["text"]
            
            logger.info(f"Postprocessing segments with GPT for filepath: {Path(file_path).name}")
            segments_with_speakers = postprocess_segments(segments)
            text = " ".join([seg["text"] for seg in segments_with_speakers])
            
            return {
                "text": text,
                "segments": segments_with_speakers
            }
        elif is_do_post2:
            trans_lines = improve_transcription_lines(text)
            trans_segs_with_ts = get_improved_lines_with_ts(trans_lines, segments)
            for seg in trans_segs_with_ts:
                seg["speaker"] = -1  # Initialize with -1 for ML diarization

            return {
                "text": text,
                "segments": trans_segs_with_ts
            }

        else:
            # For ML-based diarization, we keep the original segments
            # and let the diarization step handle speaker assignment
            for seg in segments:
                seg["speaker"] = -1  # Initialize with -1 for ML diarization

            return {
                "text": text,
                "segments": segments
            }

    except Exception as e:
        err_msg = f"ERROR in process_audio_with_openai_api: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise

def split_audio_with_vad(file_path: str, analysis_job: AnalysisWork, target_duration: int = 60) -> List[str]:
    """
    Split audio file into chunks using VAD, aiming for chunks of target_duration seconds.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Apply time limit if enabled
        if analysis_job.options["is_limit_time"]:
            limit_ms = analysis_job.options["limit_time_sec"] * 1000
            if len(audio) > limit_ms:
                logger.info(f"Limiting audio to {analysis_job.options['limit_time_sec']} seconds")
                audio = audio[:limit_ms]

        # Get audio UUID from directory name
        audio_uuid = os.path.basename(os.path.dirname(file_path))
        
        # Create directory for splits
        splits_dir = config.TEMP_DIR / "splits" / audio_uuid
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to mono if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Convert to 16kHz for VAD
        audio = audio.set_frame_rate(16000).set_sample_width(2)
        
        # Create a temporary WAV file for VAD
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # Export to WAV with correct parameters
        audio.export(temp_wav_path, format='wav')

        try:
            # Initialize VAD
            vad = webrtcvad.Vad(3)  # Aggressiveness (1~3) 3 being most aggressive
            
            # Read WAV file
            with wave.open(temp_wav_path, 'rb') as wav:
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                
                if sample_rate not in [8000, 16000, 32000, 48000] or sample_width != 2:
                    logger.warning("Unsupported sample rate or width. Falling back to simple splitting.")
                    return split_audio_by_time(file_path, analysis_job, target_duration)
                
                # Calculate frame duration and size
                frame_duration_ms = 30
                frame_size = int(sample_rate * frame_duration_ms / 1000) * sample_width
                frames = wav.readframes(wav.getnframes())

        except Exception as e:
            logger.warning(f"Error using WebRTC VAD: {e}. Falling back to simple splitting.")
            return split_audio_by_time(file_path, analysis_job, target_duration)
        finally:
            os.unlink(temp_wav_path)

        # Parallel VAD processing
        def process_vad_frame(index):
            """ Process a single frame with VAD. """
            start = index * frame_size
            frame = frames[start: start + frame_size]
            if len(frame) != frame_size:
                return False
            try:
                return vad.is_speech(frame, sample_rate)
            except Exception as e:
                logger.warning(f"Error processing frame {index}: {e}.")
                return False

        num_frames = (len(frames) - frame_size + 1) // frame_size
        voice_frames = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_vad_frame, range(num_frames))
        
        voice_frames = list(results)

        # Detect continuous speech segments
        segments = []
        in_speech = False
        segment_start = 0
        segment_frames = 0
        target_frames = int((target_duration * 1000) / frame_duration_ms)

        # Calculate total duration in seconds
        total_duration = len(voice_frames) * frame_duration_ms / 1000

        for i, is_speech in enumerate(voice_frames):
            if is_speech and not in_speech:
                segment_start = i
                in_speech = True
                segment_frames = 0
            
            if in_speech:
                segment_frames += 1
                
                if (not is_speech and segment_frames >= target_frames * 0.8) or \
                   segment_frames >= target_frames * 1.2:
                    segments.append((segment_start, i))
                    in_speech = False

        if in_speech:
            segments.append((segment_start, len(voice_frames) - 1))

        # print(f"segments: {segments}")

        ## for all segments, make tuple to list
        segments = [list(segment) for segment in segments]

        # Create VAD segments info for frontend visualization
        vad_segments = []
        for idx_segment, (start, end) in enumerate(segments):
            start_time = start * frame_duration_ms / 1000
            end_time = end * frame_duration_ms / 1000
            vad_segments.append({
                "idx": idx_segment,
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time
            })

        ## set first segment is prev segment's last segment
        prev_end = 0
        segments_with_nonactive = []
        for segment in segments:
            # segment[0] = prev_end
            segments_with_nonactive.append([prev_end, segment[1]])
            prev_end = segment[1]


        ## fix start and 
        vad_segments_with_nonactive = []
        for idx_segment, (start, end) in enumerate(segments_with_nonactive):
            start_time = start * frame_duration_ms / 1000
            end_time = end * frame_duration_ms / 1000
            vad_segments_with_nonactive.append({
                "idx": idx_segment,
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time
            })
            
        

        print(f"vad_segments: {vad_segments}")
        print(f"vad_segments_with_nonactive: {vad_segments_with_nonactive}")

        # Update the job step with VAD segments info
        analysis_job.update_step({
            "step_name": "splitting",  # Always include explicit step_name
            "status": "in_progress",
            "vad_segments": vad_segments,
            "num_segments": len(segments),
            "total_duration": total_duration
        })

        # Add step tracking for transcription
        analysis_job.update_step({
            "step_name": "transcribing",  # Always include explicit step_name
            "status": "in_progress",
            "total_splits": len(analysis_job.split_paths),
            "processed_splits": 0
        })

        # Fallback if no speech segments found
        if not segments:
            logger.warning("No speech segments found. Falling back to simple time-based splitting.")
            return split_audio_by_time(file_path, analysis_job, target_duration)

        # Parallel processing for exporting audio segments
        original_audio = AudioSegment.from_file(file_path)
        split_segments = [None] * len(segments)

        def export_audio_segment(idx, start, end):
            """ Export a detected VAD segment to a WAV file. """
            start_ms = start * frame_duration_ms
            end_ms = end * frame_duration_ms
            print(f"Split {idx}: start at {start}, end at {end}")
            segment_audio = original_audio[start_ms:end_ms]
            
            # Convert to mono 16kHz WAV
            segment_audio = segment_audio.set_channels(1).set_frame_rate(16000)
            
            output_path = splits_dir / f"id[{audio_uuid}]_split[{idx}].wav"
            segment_audio.export(str(output_path), format='wav')
            logger.info(f"Created WAV split {idx}: {output_path} ({(end_ms - start_ms) / 1000:.2f} seconds)")
            
            # Get the corresponding vad_segment for this index
            vad_segment = vad_segments_with_nonactive[idx]
            return {
                "path": str(output_path),
                "idx": idx,
                "start": vad_segment["start"],  # Use start time in seconds from vad_segments
                "end": vad_segment["end"],      # Use end time in seconds from vad_segments
                "duration": vad_segment["duration"]
            }

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {
                executor.submit(export_audio_segment, idx, prev_end, end): idx
                for idx, (prev_end, end) in enumerate(segments_with_nonactive)
            }
            
            for future in as_completed(future_to_index):
                split_segments[future.result()["idx"]] = future.result()

        return split_segments
        
    except Exception as e:
        logger.error(f"ERROR in split_audio_with_vad: {e}\n{traceback.format_exc()}")
        logger.info("Falling back to simple time-based splitting due to error.")
        return split_audio_by_time(file_path, analysis_job, target_duration)

def split_audio_by_time(file_path: str, analysis_job: AnalysisWork, target_duration: int = 60) -> List[str]:
    """
    Split audio file into chunks based on time, without using VAD.
    Saves splits as WAV files with 16kHz sample rate and mono channel.
    
    Args:
        file_path: Path to the audio file
        analysis_job: The analysis job instance for updating progress
        target_duration: Target duration for each split in seconds (default: 60)
        
    Returns:
        List[str]: List of paths to the split audio files
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Get audio UUID from directory name
        audio_uuid = os.path.basename(os.path.dirname(file_path))
        
        # Create directory for splits
        splits_dir = config.TEMP_DIR / "splits" / audio_uuid
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of splits
        audio_length_ms = len(audio)
        target_duration_ms = target_duration * 1000
        
        # Generate split metadata
        split_metadata = []
        split_paths = []
        
        # Create VAD-like segments for visualization
        total_duration = audio_length_ms / 1000  # Convert to seconds

        for idx, start_ms in enumerate(range(0, audio_length_ms, target_duration_ms)):
            end_ms = min(start_ms + target_duration_ms, audio_length_ms)
            output_path = splits_dir / f"id[{audio_uuid}]_split[{idx}].wav"
            
            split_metadata.append({
                "index": idx,
                "start_time": start_ms / 1000,
                "end_time": end_ms / 1000,
                "file_path": str(output_path)
            })

        split_meta_wo_filepath = copy.deepcopy(split_metadata)
        for seg in split_meta_wo_filepath:
            del seg["file_path"]
        
        # Update the job step with VAD segments info
        analysis_job.update_step({
            "step_name": "splitting",  # Always include explicit step_name
            "status": "in_progress",
            "vad_segments": split_meta_wo_filepath,
            "num_splits": len(split_meta_wo_filepath),
            "total_duration": total_duration
        })
        
        # Save metadata before processing audio chunks
        meta_path = splits_dir / "split_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(split_metadata, f, indent=4)
        
        logger.info(f"Saved split metadata: {meta_path}")

        # Function to export audio segment
        def export_segment(segment_info):
            start_ms = int(segment_info["start_time"] * 1000)
            end_ms = int(segment_info["end_time"] * 1000)
            output_path = Path(segment_info["file_path"])

            segment_audio = audio[start_ms:end_ms]
            # Convert to mono 16kHz WAV
            segment_audio = segment_audio.set_channels(1).set_frame_rate(16000)
            segment_audio.export(str(output_path), format='wav')
            
            logger.info(f"Created WAV split {segment_info['index']}: {output_path} ({(end_ms - start_ms) / 1000:.2f} seconds)")
            return str(output_path)

        # Use multithreading to process audio export
        is_use_thread = True
        futures = []
        with ThreadPoolExecutor() as executor:
            for segment in split_metadata:
                if is_use_thread:
                    futures.append(executor.submit(export_segment, segment))
                else:
                    split_paths.append(export_segment(segment))
            
            for future in futures:
                if future is not None:
                    split_paths.append(future.result())

        return split_paths
        
    except Exception as e:
        err_msg = f"ERROR in split_audio_by_time: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise