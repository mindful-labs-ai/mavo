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
from backend.util.logger import get_logger
from fuzzywuzzy import fuzz

# Get logger
logger = get_logger(__name__)

# OpenAI client (lazy loading - will be loaded when needed)
_openai_client = None

def create_openai_client(api_key):
    """
    Create an OpenAI client with proper error handling.
    
    Args:
        api_key: The OpenAI API key
        
    Returns:
        The OpenAI client or None if initialization fails
    """
    try:
        # Try to create the client with just the API key
        return OpenAI(api_key=api_key)
    except TypeError as e:
        err_msg = f"ERROR in create_openai_client: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        if "unexpected keyword argument 'proxies'" in str(e):
            # If the error is about proxies, try without http_client
            logger.warning("Detected 'proxies' error, trying alternative initialization")
            try:
                # Import the specific HTTP client to customize it
                import httpx
                # Create a client without proxies
                http_client = httpx.Client()
                return OpenAI(api_key=api_key, http_client=http_client)
            except Exception as e2:
                err_msg = f"ERROR in create_openai_client (alternative init): {e2}\n with traceback:\n{traceback.format_exc()}"
                logger.error(err_msg)
                return None
        else:
            logger.error(f"TypeError initializing OpenAI client: {e}")
            return None
    except Exception as e:
        err_msg = f"ERROR in create_openai_client: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        return None

def get_openai_client():
    """
    Get the OpenAI client, initializing it if necessary.
    
    Returns:
        The OpenAI client
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return None
        
        logger.info("Initializing OpenAI client")
        _openai_client = create_openai_client(api_key)
        if _openai_client:
            logger.info("OpenAI client initialized successfully")
        else:
            logger.error("Failed to initialize OpenAI client")
            
    return _openai_client

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


def postprocess_segments(segments: List[dict]) -> List[dict]:
    """
    Send the segments to ChatGPT for text improvement and assignment of 'speaker'.
    Returns a list of improved segments with keys [id, start, end, text, speaker].
    """

    # 1. Prepare a JSON schema that ChatGPT must adhere to.
    json_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "text_raw": {"type": "string"},
                        "text": {"type": "string"},
                        "speaker": {"type": "integer"}
                    },
                    "required": ["id", "start", "end", "text", "text_raw", "speaker"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["segments"],
        "additionalProperties": False
    }

    # 2. Construct our system and user messages.
    #    - System message: Tells ChatGPT to be a helpful assistant, keep meaning, correct grammar.
    #    - User message:  Passes the original segments as JSON,
    #                     instructs ChatGPT to add a "speaker" field for each segment.

            # "content": (
            #     "You are a helpful assistant that improves transcription text from a psychological counseling session. "
            #     "Read the whole dialogue of the counseling session, then think how to improve the text. "
            #     "Only correct clear errors such as spelling, or misheard words, and do not rephrase or paraphrase the original content. "
            #     "The text should be in Korean."
            #     "Assign a 'speaker' value (0, 1, 2, ...) for each segment. 0 is counselor, 1 is client1, 2 is client2, etc. Use 'text' field to determine the speaker."
            #     "Counselor tends to start the conversation more, ask more questions, cut-in more, use professional, empathetic, and clear language, often asking reflective and open-ended questions, "
            #     "providing guidance in a calm and supportive manner. "
            #     "If the context of the dialogue changes, it is likely that the counselor has intervened. "
            #     "Client tends to express personal emotions and experiences, sometimes in an informal or hesitant tone, "
            #     "and may ask for analysis or express uncertainty. "
            #     "Please process the text accordingly and return the improved transcription with the assigned speaker values."
            # )


    segments_cpy = segments.copy()
    ## trim text to 25 characters
    for seg in segments_cpy:
        seg["text"] = seg["text"][:25]


    messages = [
        {
            "role": "system",
            "content": config.transcript_system_prompt
            
        },
            # "Assign a 'speaker' value (0, 1, 2) for each segment, where 0 is counselor, 1 is client, and 2 is others."
        {
            "role": "user",
            "content": (
                "Here is the JSON input:\n\n"
                + json.dumps(segments_cpy, ensure_ascii=False)
                + "\n\n"
                "Please return a valid JSON object following this exact schema:\n"
                + json.dumps(json_schema, ensure_ascii=False)
                + "\n\n"
                "The output must be strictly valid JSON and must only contain the 'segments' array of objects, "
                "where each object has 'id', 'start', 'end', 'text', and 'speaker'."
            )
        }
    ]

    # 3. Call ChatGPT with the response format set to our JSON schema.
    #    This ensures ChatGPT's response is strictly valid JSON.
    completion = get_openai_client().chat.completions.create(
        # model=config.OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL,  # or whichever model you prefer
        model="gpt-4.1-mini",  # or whichever model you prefer
        temperature=0.2,        # Adjust as needed
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": json_schema
            }
        }
    )

    # 4. Extract the improved segments from the response. 
    #    The property name here (completion.structured_response["segments"]) 
    #    corresponds to the root key in our JSON schema ("segments").
    # improved_segments = completion.structured_response["segments"]


    raw_response = completion.choices[0].message.content
    structured_response = json.loads(raw_response)
    improved_segments = structured_response["segments"]

    return improved_segments


def simple_diarization(segments: List[dict]) -> List[dict]:
    """
    Simple speaker diarization based on pauses.
    
    This is a placeholder for Stage 6 implementation.
    
    Args:
        segments: List of segments from Whisper
        
    Returns:
        List[dict]: List of segments with speaker labels
    """
    current_speaker = 0
    for i, segment in enumerate(segments):
        # If this is the first segment, assign speaker 0
        if i == 0:
            segment["speaker"] = current_speaker
            continue
            
        # Get the previous segment
        prev_segment = segments[i-1]
        
        # Calculate the pause between segments
        pause = segment["start"] - prev_segment["end"]
        
        # If the pause is long enough, switch speakers
        if pause > 0.75:  # 750ms pause threshold
            current_speaker = 1 - current_speaker  # Toggle between 0 and 1
            
        segment["speaker"] = current_speaker
        
    return segments


def custom_json_encoder(obj):
    """Custom JSON encoder that ignores non-serializable objects"""
    try:
        # Try to get __dict__ first
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}  # Filter out private attributes
        # For other types, just ignore them
        return "<non_serializable_data>"
    except Exception:
        return "<non_serializable_data>"


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

async def process_audio_with_openai_api(file_path: str, idx_split: int, total_splits: int, options=None) -> Dict:
    client = get_openai_client()
    if not client:
        raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
    
    logger.info(f"Transcribing audio file with OpenAI API: {file_path}")
    
    try:
        # Get analysis job from file path
        audio_uuid = os.path.basename(os.path.dirname(file_path))
        analysis_job = analysis_jobs.get(audio_uuid)
        
        if not analysis_job:
            raise ValueError(f"No analysis job found for {audio_uuid}")

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
        is_do_postprocess = analysis_job.options["diarization_method"] == "stt_apigpt_diar_apigpt"
        # is_do_postprocess = False
        if is_do_postprocess:
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
        raise ValueError(f"Error transcribing audio with OpenAI API: {str(e)}")

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
        for start, end in segments:
            start_time = start * frame_duration_ms / 1000
            end_time = end * frame_duration_ms / 1000
            vad_segments.append({
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
        

        print(f"vad_segments: {vad_segments}")

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
            return {
                "path": str(output_path),
                "start": start,
                "end": end,
                "idx": idx
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

def save_transcription_result_json(audio_uuid: str, transcription_result, suffix: str = "transcript") -> None:
    """
    Save the final transcription result as a JSON file.
    
    Args:
        audio_uuid: UUID of the audio file
        transcription_result: TranscriptionResult object to save
    """
    try:
        # Create the uploads directory if it doesn't exist
        uploads_dir = Path(config.UPLOADS_DIR) / audio_uuid
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the transcript file path
        transcript_path = uploads_dir / f"id[{audio_uuid}]_{suffix}.json"
        
        # Convert TranscriptionResult to dictionary
        result_dict = {
            "text": transcription_result.text,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker
                } for seg in transcription_result.segments
            ],
            "speakers": [
                {
                    "id": spk.id,
                    "role": spk.role
                } for spk in transcription_result.speakers
            ]
        }
        
        # Save as JSON
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved final transcription result to {transcript_path}")
        
    except Exception as e:
        logger.error(f"Error saving final transcription result: {e}\n{traceback.format_exc()}")
        # Continue even if saving fails

async def diarize_audio(file_path: str) -> List[Dict]:
    """
    Perform speaker diarization on an audio file.
    """
    try:
        logger.info(f"Starting diarization for {file_path}")
        
        # Get analysis job from file path
        audio_uuid = os.path.basename(os.path.dirname(file_path))
        analysis_job = analysis_jobs.get(audio_uuid)
        
        if not analysis_job:
            raise ValueError(f"No analysis job found for {audio_uuid}")
        
        # Convert audio to required format
        audio = AudioSegment.from_file(file_path)
        
        # Apply time limit if enabled
        if analysis_job.options["is_limit_time"]:
            limit_ms = analysis_job.options["limit_time_sec"] * 1000
            if len(audio) > limit_ms:
                logger.info(f"Limiting audio to {analysis_job.options['limit_time_sec']} seconds")
                audio = audio[:limit_ms]
        
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
            audio.export(tmp_wav_path, format='wav')
        
        try:
            if analysis_job.options["diarization_method"] == "stt_apigpt_diar_mlpyan":
                # Use pyannote.audio for diarization
                HF_TOKEN = os.getenv("HF_TOKEN")
                if not HF_TOKEN:
                    raise ValueError("HF_TOKEN not found in environment variables")
                    
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
                
                # Use MPS if available, otherwise CPU
                logger.info(f"Diarizing using device: {config.DEVICE}")
                device = config.DEVICE
                pipeline.to(torch.device(device))
                
                # Load audio and run diarization
                waveform, sample_rate = torchaudio.load(tmp_wav_path)
                audio_file = {"waveform": waveform, "sample_rate": sample_rate}
                
                # Ensure max_speakers is an integer
                max_speakers = int(config.MAX_SPEAKERS) if isinstance(config.MAX_SPEAKERS, str) else config.MAX_SPEAKERS
                diarization = pipeline(audio_file, min_speakers=2, max_speakers=max_speakers)
                
                # Get total duration from waveform
                total_duration = waveform.size(1) / sample_rate
                
                # Convert diarization result to list of segments
                diarization_segments = []
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_num = speaker.split("_")[1]
                    speaker_num_int = int(speaker_num) if speaker_num.isdigit() else -1

                    diarization_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.end - segment.start,
                        "speaker": speaker_num_int
                    })

                # sort diarization segments by start time
                diarization_segments.sort(key=lambda x: x["start"])

                # Mark the diarization step as completed
                analysis_job.complete_step("diarizing")
                
                # Save diarization results to JSON
                uploads_dir = config.UPLOADS_DIR / str(audio_uuid)
                uploads_dir.mkdir(parents=True, exist_ok=True)
                diarization_path = uploads_dir / f"id[{audio_uuid}]_ml_diarized.json"
                
                diarization_result = {
                    "total_duration": total_duration,
                    "num_speakers": len(set(seg["speaker"] for seg in diarization_segments)),
                    "segments": diarization_segments
                }
                
                with open(diarization_path, "w", encoding="utf-8") as f:
                    json.dump(diarization_result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved diarization result to {diarization_path}")
                return diarization_segments
            
            else:
                # Use GPT-based diarization
                # This will be handled in the transcription improvement step
                total_duration = len(audio) / 1000
                diarization_segments = [{
                    "start": 0,
                    "end": total_duration,
                    "duration": total_duration,
                    "speaker": -1
                }]
            
            # Update job step with diarization visualization data
            if analysis_job:
                analysis_job.update_step({
                    "status": AudioStatus.DIARIZING,
                    "diarization_segments": diarization_segments,
                    "total_duration": total_duration,
                    "num_speakers": len(set(seg["speaker"] for seg in diarization_segments))
                })
            
            logger.info(f"Diarization completed for {file_path}")

            
            return diarization_segments
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
                
    except Exception as e:
        err_msg = f"ERROR in diarize_audio: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise

def improve_transcription(segments: List[Segment]) -> List[Segment]:
    """
    Improve transcription segments using OpenAI API.
    """
    # 1. Prepare a JSON schema that ChatGPT must adhere to.
    json_schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "text": {"type": "string"},
                        "speaker": {"type": "integer"}
                    },
                    "required": ["id", "start", "end", "text", "speaker"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["segments"],
        "additionalProperties": False
    }

    # 2. Construct our system and user messages.
    #    - System message: Tells ChatGPT to be a helpful assistant, keep meaning, correct grammar.
    #    - User message:  Passes the original segments as JSON,
    #                     instructs ChatGPT to add a "speaker" field for each segment.
    segments_dict = [seg.__dict__ for seg in segments]
    messages = [
        {
            "role": "system",
            "content": config.transcript_system_prompt
        },
        {
            "role": "user",
            "content": (
                "Here is the JSON input:\n\n"
                + json.dumps(segments_dict, ensure_ascii=False)
                + "\n\n"
                "Please return a valid JSON object following this exact schema:\n"
                + json.dumps(json_schema, ensure_ascii=False)
                + "\n\n"
                "The output must be strictly valid JSON and must only contain the 'segments' array of objects, "
                "where each object has 'id', 'start', 'end', 'text', and 'speaker'."
            )
        }
    ]

    print("imporve_transcription messages", messages)

    # 3. Call ChatGPT with the response format set to our JSON schema.
    #    This ensures ChatGPT's response is strictly valid JSON.
    completion = get_openai_client().chat.completions.create(
        model=config.OPENAI_API_TRANSCRIPT_IMPROVEMENT_MODEL,  # or whichever model you prefer
        temperature=0.2,        # Adjust as needed
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": json_schema
            }
        }
    )

    # 4. Extract the improved segments from the response. 
    #    The property name here (completion.structured_response["segments"]) 
    #    corresponds to the root key in our JSON schema ("segments").
    # improved_segments = completion.structured_response["segments"]


    raw_response = completion.choices[0].message.content
    structured_response = json.loads(raw_response)
    improved_segments = structured_response["segments"]

    return improved_segments
    pass

## adhoc for async job
"""
To wrap asynchronous functions in a thread with a new event loop, we can define helper functions that use asyncio.run() to call the async functions synchronously.

example:
def run_async_function(coro):
    return asyncio.run(coro)
"""
def run_diarize_audio(file_path):
    return asyncio.run(diarize_audio(file_path))
def run_process_transcription(analysis_job, split_segments):
    return asyncio.run(process_transcription(analysis_job, split_segments))


def save_analysis_work_json(analysis_job: AnalysisWork) -> None:
    """
    Save the AnalysisWork state as JSON.
    
    Args:
        analysis_job: The AnalysisWork instance to save
    """
    try:
        # Get audio UUID from file path
        audio_uuid = os.path.basename(os.path.dirname(analysis_job.file_path))
        
        # Create uploads directory if it doesn't exist
        save_dir = Path(config.UPLOADS_DIR) / audio_uuid
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the save path
        save_path = save_dir / f"id[{audio_uuid}]_analysiswork.json"
        
        # Convert AnalysisWork to dict, handling datetime objects
        analysis_dict = {
            "id": analysis_job.id,
            "filename": analysis_job.filename,
            "file_path": analysis_job.file_path,
            "status": analysis_job.status,
            "error": analysis_job.error,
            "options": analysis_job.options,
            "total_chunks": analysis_job.total_chunks,
            "chunks": {str(k): v.dict() for k, v in analysis_job.chunks.items()},
            "steps": analysis_job.steps,
            "created_at": analysis_job.created_at.isoformat(),
            "updated_at": analysis_job.updated_at.isoformat(),
            "result": analysis_job.result.dict() if analysis_job.result else None
        }
        
        # Save as JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved analysis work state to {save_path}")
        
    except Exception as e:
        err_msg = f"ERROR in save_analysis_work_json: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)


async def do_vad_split(analysis_job: AnalysisWork) -> None:
    """Split audio file with VAD."""
    # Update status to splitting
    analysis_job.update_status(AudioStatus.SPLITTING)
    
    # Split audio using VAD
    logger.info("Splitting audio using VAD...")
    split_segments = split_audio_with_vad(analysis_job.file_path, analysis_job, target_duration=60*2)
    logger.info(f"Created {len(split_segments)} splits")
    analysis_job.split_segments = split_segments ## cautious - this may lead dump error.
    
    # Store split paths in the job
    analysis_job.split_paths = split_segments if isinstance(split_segments[0], str) else [seg["path"] for seg in split_segments]

    # Save state after status update
    save_analysis_work_json(analysis_job)
    
    # At the end of the function, mark the splitting step as completed
    analysis_job.complete_step("splitting")
    
    # Log completion
    logger.info(f"VAD splitting completed for {analysis_job.id}")

async def process_analysis_job_gpt_diar_gpt(analysis_job: AnalysisWork) -> None:
    logger.info(f"Processing analysis job for {analysis_job.audio_uuid} with method: {analysis_job.options['diarization_method']}")
    await do_vad_split(analysis_job)

    ## add tasks
    tasks = []
    analysis_job.update_status(AudioStatus.TRANSCRIBING)
    transcription_task = asyncio.create_task(asyncio.to_thread(run_process_transcription, analysis_job, analysis_job.split_segments), name="transcription")
    tasks.append(transcription_task)

    # Wait for tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("asyncio results", results)

    result_transcription = results[0]
    transcription_segments = result_transcription["segments"]

    # Store diarization result (in this method, the diarization is done in the transcription)
    # We still need to store an intermediate result for the front end
    speaker_segments = []
    speaker_ids = set()
    for i, seg in enumerate(transcription_segments):
        speaker_id = seg.get("speaker", 0)
        # Handle "undecided" speaker ID - map it to a special integer or keep as is
        if speaker_id != "undecided":
            speaker_ids.add(speaker_id)
        speaker_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker_id
        })
    
    # Create speakers list - only for actual numeric speakers
    speakers = [
        {"id": speaker_id, "role": "counselor" if speaker_id == 0 else "client"}
        for speaker_id in sorted(speaker_ids) if speaker_id != "undecided"
    ]
    
    # Store diarization result
    analysis_job.store_step_result("diarization", {
        "speakers": speakers,
        "segments": speaker_segments
    })
    
    # Update status to improving
    analysis_job.update_status(AudioStatus.IMPROVING)
    
    # Create segments with proper handling of "undecided" speaker
    segments = [
        Segment(
            id=i,
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            # Keep "undecided" as is, it's now supported by the Segment model
            speaker=seg["speaker"],
            speaker_diarized=""
        ) for i, seg in enumerate(transcription_segments)
    ]
        
    # Sort segments by start time
    segments.sort(key=lambda x: x.start)

    ## save the result
    analysis_job.result = TranscriptionResult(
        text=result_transcription["text"],
        segments=segments,
        speakers=[]
    )
    
    # Store improving result (complete result with speaker identification)
    improving_segments = []
    for i, seg in enumerate(segments):
        improving_segments.append({
            "id": i,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker": seg.speaker
        })
    
    # Store the improving result
    analysis_job.store_step_result("improving", {
        "segments": improving_segments
    })

    # Mark the improving step as completed
    analysis_job.complete_step("improving")

    print("analysis_job", analysis_job)
    print("analysis_job.uuid", analysis_job.audio_uuid)
    print("analysis_job.result", analysis_job.result)

    save_transcription_result_json(analysis_job.audio_uuid, analysis_job.result, suffix="transcript")
    save_analysis_work_json(analysis_job)

    # Mark the job as completed
    analysis_job.update_status(AudioStatus.COMPLETING)

    

    pass


async def identify_speakers_from_conversation(trans_segs_with_ts, unique_speakers):
    """
    대화 내용을 분석하여 상담사와 내담자를 식별합니다.
    
    Args:
        trans_segs_with_ts: 시간 정보가 포함된 전사 세그먼트
        unique_speakers: 고유한 화자 ID 목록
    
    Returns:
        tuple: (counselor_ids, client_ids) - 상담사와 내담자로 식별된 ID 목록
    """
    # 각 화자별 대화 내용 수집
    speaker_texts = {}
    for seg in trans_segs_with_ts:
        if seg["speaker"] not in speaker_texts:
            speaker_texts[seg["speaker"]] = []
        speaker_texts[seg["speaker"]].append(seg["text"])
    
    # 텍스트가 충분히 긴 화자만 분석 대상으로 선정
    valid_speakers = {
        speaker: texts for speaker, texts in speaker_texts.items() 
        if speaker >= 0 and texts and "".join(texts).strip()
    }
    
    if not valid_speakers:
        logger.warning("No valid speakers found with text content")
        return [0] if 0 in unique_speakers else [unique_speakers[0] if unique_speakers else 0], \
               [s for s in unique_speakers if s != 0 and s != (unique_speakers[0] if unique_speakers else 0)]
    
    # ChatGPT에 화자별 특징 분석 요청
    speaker_analysis_prompt = """
    아래는 상담 대화에서 각 화자가 말한 내용입니다. 각 화자가 상담사인지 내담자인지 분석해주세요.
    
    상담사의 특징:
    - 전문적인 용어 사용
    - 반영적 경청, 개방형 질문
    - 공감적이고 구조화된 대화 방식
    - 내담자의 말에 대한 요약과 명확화
    - 상담 초기와 마무리 과정 주도
    
    내담자의 특징:
    - 개인적 경험과 감정 표현
    - 도움을 요청하는 어투
    - 불확실성이나 고민 표현
    - 자신의 문제에 대한 설명
    
    각 화자에 대해 '상담사', '내담자', 또는 '불확실'로 분류하고, 그 이유를 간략히 설명해주세요.
    정확한 JSON 형식으로 응답해주세요.
    """

    # JSON 스키마 정의
    json_schema = {
        "type": "object",
        "properties": {
            "speaker_roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker_id": {"type": "integer"},
                        "role": {"type": "string", "enum": ["counselor", "client"]},
                        "confidence_norm": {"type": "number"},
                        "reason": {"type": "string"}
                    },
                    "required": ["speaker_id", "role", "confidence_norm", "reason"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["speaker_roles"],
        "additionalProperties": False
    }
    
    speakers_content = ""
    for speaker_id, texts in valid_speakers.items():
        speakers_content += f"\n화자 {speaker_id} 발언:\n"
        # 각 화자당 최대 10개의 발언만 포함 (너무 길지 않게)
        for i, text in enumerate(texts[:10]):
            speakers_content += f"- {text}\n"
        if len(texts) > 10:
            speakers_content += f"... 외 {len(texts)-10}개 발언\n"
    print("speakers_content", speakers_content)
    
    messages = [
        {"role": "system", "content": speaker_analysis_prompt},
        {"role": "user", "content": speakers_content}
    ]
    print("speaker_analysis_prompt", speaker_analysis_prompt)
    print("messages", messages)

    try:
        response = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema
                }
            }
        )
        
        analysis = json.loads(response.choices[0].message.content)
        print("identify speaker response", analysis)
        
        # 결과 정렬 (신뢰도 기준)
        speaker_roles = sorted(
            analysis["speaker_roles"], 
            key=lambda x: (x["role"], -x["confidence_norm"])
        )
        
        # 상담사와 내담자 ID 추출
        # counselor_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "counselor" and s["confidence_norm"] > 0.6]
        # client_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "client" and s["confidence_norm"] > 0.6]
        # uncertain_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "uncertain" or s["confidence_norm"] <= 0.6]
        counselor_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "counselor"]
        client_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "client"]
        uncertain_ids = [s["speaker_id"] for s in speaker_roles if s["role"] == "uncertain"]
        
        # 결과 후처리
        if not counselor_ids and client_ids:
            # 상담사가 없으면 불확실한 화자 중 첫 번째를 상담사로 간주
            if uncertain_ids:
                counselor_ids = [uncertain_ids[0]]
                uncertain_ids.pop(0)
            # 여전히 없으면 첫 번째 화자를 상담사로 간주
            elif unique_speakers:
                counselor_ids = [unique_speakers[0]]
        
        # 나머지 불확실한 화자는 내담자로 간주
        client_ids.extend(uncertain_ids)
        
        # 중복 제거
        counselor_ids = list(set(counselor_ids))
        client_ids = list(set(client_ids) - set(counselor_ids))
        
        # 로깅
        logger.info(f"Identified counselor IDs: {counselor_ids}")
        logger.info(f"Identified client IDs: {client_ids}")
        
        return counselor_ids, client_ids
        
    except Exception as e:
        logger.error(f"Error in speaker identification: {e}, {traceback.format_exc()}")
        # 오류 발생 시 기본값 반환 (첫 번째 화자를 상담사로 간주)
        return [0] if 0 in unique_speakers else [unique_speakers[0] if unique_speakers else 0], \
               [s for s in unique_speakers if s != 0 and s != (unique_speakers[0] if unique_speakers else 0)]


# def get_cleared_diarization_segments(diarization_segments):
#     """
#     - remove diarization segments that overlaps.
#     - if overlaps, make split, and keep the shorer one.
#     """
#     cleared_diarization_segments = []

#     ## TODO: do something
#     return cleared_diarization_segments

def get_cleared_diarization_segments(diarization_segments):
    """
    Ensures no overlapping segments remain. If two segments overlap,
    we 'keep' whichever segment has the smaller total duration (shorter).
    
    Steps:
      1. Sort segments by start time.
      2. Iterate over them, merging/resolving overlaps one by one.
      3. If new seg overlaps with last seg in cleared list:
         - Compare durations.
         - Split so that only the shorter one occupies the overlap region.
         - Leftover portions (non-overlapping) are kept with the original speaker.
    """
    
    # Sort the segments by start time
    diarization_segments = sorted(diarization_segments, key=lambda x: x['start'])

    cleared = []

    for seg in diarization_segments:
        # Make sure duration is consistent (in case it's missing or off)
        seg['duration'] = seg['end'] - seg['start']
        
        # If no segments in cleared yet, just add
        if not cleared:
            cleared.append(seg)
            continue
        
        # Compare with the last segment we added to cleared
        prev = cleared[-1]
        
        # If no overlap, just add it
        if seg['start'] >= prev['end']:
            cleared.append(seg)
            continue
        
        # There *is* an overlap
        #
        # Overlap region: [overlap_start, overlap_end]
        overlap_start = seg['start']
        overlap_end   = min(seg['end'], prev['end'])
        
        # Compare the *entire* durations
        # (not just the overlap portion)
        if seg['duration'] < prev['duration']:
            # -------------------------------
            #  The *new* segment is shorter
            #  => it 'wins' in the overlap
            # -------------------------------
            
            old_prev_end = prev['end']  # remember old end
            
            # 1. Trim or remove old segment's overlap
            if overlap_start > prev['start']:
                # Just end the old segment where the new one begins
                prev['end'] = overlap_start
                prev['duration'] = prev['end'] - prev['start']
            else:
                # The new seg starts before or exactly at the prev segment's start;
                # so basically the old seg is fully overshadowed at the beginning.
                cleared.pop()  # remove old entirely for now
            
            # 2. If the old segment originally extended beyond seg['end'],
            #    we keep that leftover region for the old speaker
            if seg['end'] < old_prev_end:
                leftover_seg = {
                    'start': seg['end'],
                    'end': old_prev_end,
                    'speaker': prev['speaker'],
                    'duration': old_prev_end - seg['end']
                }
                # If we had popped `prev`, we can just append the leftover
                cleared.append(leftover_seg)
            
            # 3. Now add the new (shorter) segment
            cleared.append(seg)
        
        else:
            # ---------------------------------
            #  The *old* segment is shorter (or equal),
            #  => it 'wins' in the overlap
            # ---------------------------------
            #
            # So remove the overlap from the new segment
            # i.e. push the new segment's start to the old segment's end
            if seg['end'] > prev['end']:
                # There's at least some portion of the new seg that extends beyond prev
                seg['start'] = prev['end']
                seg['duration'] = seg['end'] - seg['start']
                
                # If there's still leftover time after trimming the overlap, we keep it
                if seg['duration'] > 0:
                    cleared.append(seg)
                # If seg fits completely inside `prev`, it gets fully discarded
            # If seg['end'] <= prev['end'], that means the new segment
            # was fully inside the old segment, so we discard it entirely
            # (do nothing here)
            pass
    
    # Finally, filter out any invalid segments (duration <= 0, etc.)
    final_segments = []
    for c in cleared:
        d = c['end'] - c['start']
        if d > 0:
            c['duration'] = d
            final_segments.append(c)

    return final_segments

def assign_speaker_to_lines_with_gpt(lines):
#     prompt_text_improvement = """
# This is psychological counseling session transcript.
# Read whole text and guess how many speakers are there.
# Counselor tends to start the conversation more, ask more questions, cut-in more, use professional, empathetic, and clear language, often asking reflective and open-ended questions, providing guidance in a calm and supportive manner.
# There should be one counsler at leaset. And there should be at least one client, and max is 4 clients.
# Assign speaker to each line, reading the lines.
# Give me 'speaker' as an interger. 0 for 'counsler'. 1, 2, 3 ... for different clients.
# """
    prompt_text_improvement = """
This is psychological counseling session transcript. There is one counselor and one client.
Counselor tends to start the conversation more, ask more questions, cut-in more, use professional, empathetic, and clear language, often asking reflective and open-ended questions, providing guidance in a calm and supportive manner.
Assign speaker to each line, reading the lines.
Give me 'speaker' as an interger. 0 for 'counsler'. 1 for 'client'.
"""

    text = ""
    for idx_line, line in enumerate(lines):
        text += f"IDX {idx_line}: {line}\n"

    messages = [
        {"role": "system", "content": prompt_text_improvement},
        {"role": "user", "content": text}
    ]

    json_schema = { ## string list
        "type": "object",
        "properties": {
            "improved_lines": {
                "type": "array",
                "items": {
                    "type": "object",
                     "properties": {
                        "idx": {"type": "number"},
                        # "text": {"type": "string"},
                        "speaker": {"type": "number"}
                    },
                    # "required": ["idx","text", "speaker"],
                    "required": ["idx","speaker"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["improved_lines"],
        "additionalProperties": False
    }

    improved_lines = None
    try:
        completion = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",  # or whichever model you prefer
            temperature=0.3,        # Adjust as needed
            # model="o3-mini",  # or whichever model you prefer
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema
                }
            }
        )
        # improved_lines_jsonstring = completion.choices[0].message['content']
        
        # improved_lines = json.loads(improved_lines_jsonstring)
        response_content = completion.choices[0].message.content
        # response_content = improved_lines_jsonstring
        response_data = json.loads(response_content)
        improved_lines = response_data.get("improved_lines", [])
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    if improved_lines is None:
        logger.error("Failed to improve transcription text somehow")
        return None

    # print("improved_lines", improved_lines)

    return improved_lines

def improve_transcription_lines_with_speaker(text):
    """
    - improve transcription text with gpt-4o
    - correct transcription words using improved text with cosine similarity
    """

    prompt_text_improvement = """
The following text is the result of speech-to-text (STT) transcription from a psychological counseling session.

First, improve the text.
The STT contains errors. Correct the content according to the context.
When guessing the best text improvment, be aware that text often include expressions of emotions, personal feelings, and discussions of sensitive topics such as self-harm, suicidal thoughts, or intent to harm others.
Preserve the natural flow of spoken language. Use proper spacing and punctuation. Do not include explanations. Do not paraphrase.
Break lines at each sentence, as much as possible. Break lines at natural pauses, sentence endings, question marks or periods. Preserve the original meaning and flow of speech.
You may change the text to make it more natural and correct.

Second, guess how many speakers are there.

Third, assign speaker to each line, reading the lines.
There might be one counsler and at lease one client.
Give me improved 'improved_lines' json adding 'speaker' field, 
and expected values for 'speaker' is interger, 0 for consultant, 1, 2, 3 ... for clients.
"""
            #     "Counselor tends to start the conversation more, ask more questions, cut-in more, use professional, empathetic, and clear language, often asking reflective and open-ended questions, "
            #     "providing guidance in a calm and supportive manner. "
            #     "If the context of the dialogue changes, it is likely that the counselor has intervened. "
            #     "Client tends to express personal emotions and experiences, sometimes in an informal or hesitant tone, "



    if text is None:
        text = ""

    text = text.strip()
    ## remove leading and trailing quotes
    text = text.strip("\"'")
    ## remove leading and trailing newlines
    text = text.strip("\n")
    ## remove all whitespace and special characters in the text
    text = text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

    messages = [
        {"role": "system", "content": prompt_text_improvement},
        {"role": "user", "content": text}
    ]

    json_schema = { ## string list
        "type": "object",
        "properties": {
            "improved_lines": {
                "type": "array",
                "items": {
                    "type": "object",
                     "properties": {
                        "text": {"type": "string"},
                        "speaker": {"type": "number"}
                    },
                    "required": ["text", "speaker"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["improved_lines"],
        "additionalProperties": False
    }

    improved_lines = None
    try:
        completion = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",  # or whichever model you prefer
            temperature=0.2,        # Adjust as needed
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema
                }
            }
        )
        # improved_lines_jsonstring = completion.choices[0].message['content']
        
        # improved_lines = json.loads(improved_lines_jsonstring)
        response_content = completion.choices[0].message.content
        # response_content = improved_lines_jsonstring
        response_data = json.loads(response_content)
        improved_lines = response_data.get("improved_lines", [])
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    if improved_lines is None:
        logger.error("Failed to improve transcription text somehow")
        return None

    # print("improved_lines", improved_lines)

    return improved_lines

def improve_transcription_lines(text):
    """
    - improve transcription text with gpt-4o
    - correct transcription words using improved text with cosine similarity
    """

    prompt_text_improvement = """
The following text is the result of speech-to-text (STT) transcription from a psychological counseling session.
The STT contains errors. Correct the content according to the context.
When guessing the best text improvment, be aware that text often include expressions of emotions, personal feelings, and discussions of sensitive topics such as self-harm, suicidal thoughts, or intent to harm others.
Preserve the natural flow of spoken language. Use proper spacing and punctuation. Do not include explanations. Do not paraphrase.
Break lines at each sentence aggressively, as much as possible. Break lines at natural pauses, sentence endings, question marks or periods. Preserve the original meaning and flow of speech.
You may change the text to make it more natural and correct.
"""

#     """
# The following text is the result of speech-to-text (STT) transcription from a psychological counseling session.
# The STT output may contain recognition errors. Correct the content while preserving the original meaning and tone of the speaker.
# This text may include expressions of emotions, personal feelings, and discussions of sensitive topics such as self-harm or suicidal thoughts.
# Make only minimal edits necessary for clarity. Do not paraphrase or rephrase.
# Break lines at the end of each sentence. Use appropriate spacing, punctuation, and line breaks.
# Do not add explanations or summaries. Preserve the natural flow of spoken language.
# """

    if text is None:
        text = ""

    text = text.strip()
    ## remove leading and trailing quotes
    text = text.strip("\"'")
    ## remove leading and trailing newlines
    text = text.strip("\n")
    ## remove all whitespace and special characters in the text
    text = text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

    messages = [
        {"role": "system", "content": prompt_text_improvement},
        {"role": "user", "content": text}
    ]

    json_schema = { ## string list
        "type": "object",
        "properties": {
            "improved_lines": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["improved_lines"],
        "additionalProperties": False
    }

    improved_lines = None
    try:
        completion = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",  # or whichever model you prefer
            temperature=0.2,        # Adjust as needed
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema
                }
            }
        )
        # improved_lines_jsonstring = completion.choices[0].message['content']
        
        # improved_lines = json.loads(improved_lines_jsonstring)
        response_content = completion.choices[0].message.content
        # response_content = improved_lines_jsonstring
        response_data = json.loads(response_content)
        improved_lines = response_data.get("improved_lines", [])
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    if improved_lines is None:
        logger.error("Failed to improve transcription text somehow")
        return None

    # print("improved_lines", improved_lines)
    # print("transcription_words", transcription_words)

    # result = []
    # used_indices = set()

    # for idx_line, line in enumerate(improved_lines):
    #     # Normalize whitespace for length checks + fuzzy matching
    #     line_ws = " ".join(line.split())

    #     best_score = 0
    #     best_range = None

    #     # Search in 10-word windows. Adjust if needed.
    #     for i in range(len(transcription_words)):
    #         for j in range(i, min(i+10, len(transcription_words))):
    #             # Check if any words in this window have already been used
    #             if any(k in used_indices for k in range(i, j + 1)):
    #                 continue

    #             # Build window text
    #             window_words = transcription_words[i:j+1]
    #             window_text = " ".join(w['text'] for w in window_words)
    #             window_ws = " ".join(window_text.split())

    #             # 1) Skip if length difference > 3
    #             if abs(len(line_ws) - len(window_ws)) > 3:
    #                 continue

    #             # 2) Compare fuzzy scores
    #             score = fuzz.token_set_ratio(line_ws, window_ws)
    #             if score > best_score:
    #                 best_score = score
    #                 best_range = (i, j)

    #     # Once we have the best match in that 10-word range
    #     if best_range is not None:
    #         i, j = best_range
    #         matched_words = transcription_words[i:j+1]
    #         start = matched_words[0]['start']
    #         end   = matched_words[-1]['end']
    #         word_ids = [w['id'] for w in matched_words]

    #         # Mark these indices as used
    #         for k in range(i, j + 1):
    #             used_indices.add(k)
    #     else:
    #         # No match found
    #         start = None
    #         end = None
    #         word_ids = []

    #     result.append({
    #         "text": line,
    #         "start": start,
    #         "end": end,
    #         # "line_id": idx_line,
    #         "word_ids": word_ids
    #     })

    # print("better text result with paragraphing", result)

    return improved_lines



def get_seg_ts_with_diar(trans_segs_with_ts, diarization_segments):
    """
    - get transcription segments with diarization segments
    """
    trans_segs_with_ts_filt = []
    allowed_fields = ['text', 'start', 'end']
    for idx_seg, seg in enumerate(trans_segs_with_ts):
        seg_filt = {k: v for k, v in seg.items() if k in allowed_fields}
        seg_filt['idx'] = idx_seg
        trans_segs_with_ts_filt.append(seg_filt)

    diar_segs_filt = []
    allowed_fields = ['start', 'end', 'speaker']

    diarization_segments_cp = copy.deepcopy(diarization_segments)
    for seg in diarization_segments_cp:
        seg_filt = {k: v for k, v in seg.items() if k in allowed_fields}
        diar_segs_filt.append(seg_filt)

    for seg in diarization_segments_cp:
        seg['diar_label'] = chr(ord('A') + seg['speaker'])
        del seg['speaker']


    
    prompt_text_improvement = """
im analyzing the counsling.
There might be one counsler and at lease one client.
given the following data of timestamp and diarization result.
but beware that diarization result have some error and may overlap speakers, so you should consider the text as well.
give me improved 'trans_segs_with_ts' json adding 'speaker' field, 
and expected values for 'speaker' is interger, 0 for consultant, 1, 2, 3 ... for clients.
"""

    text = ""

    text += "trans_segs_with_ts: " + json.dumps(trans_segs_with_ts_filt, ensure_ascii=False)
    text += "\n\n"
    text += "diarization_segments: " + json.dumps(diar_segs_filt, ensure_ascii=False)

    messages = [
        {"role": "system", "content": prompt_text_improvement},
        {"role": "user", "content": text}
    ]
    print(f"improving transcription with diarization result, text: {text}")

    json_schema = { ## string list
        "type": "object",
        "properties": {
            "trans_segs_with_ts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "idx": {"type": "number"},
                        "speaker": {"type": "number"}
                    },
                    "required": ["idx", "start", "end", "text", "speaker"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["trans_segs_with_ts"],
        "additionalProperties": False
    }

    trans_segs_with_ts = None
    try:
        completion = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",  # or whichever model you prefer
            temperature=0.2,        # Adjust as needed
            messages=messages,
            max_tokens=16384,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema
                }
            }
        )
        response_content = completion.choices[0].message.content
        print(f"respone of seg_ts_with_diar: {response_content}")
        response_data = json.loads(response_content)
        trans_segs_with_ts = response_data.get("trans_segs_with_ts", [])
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    if trans_segs_with_ts is None:
        logger.error("Failed to improve transcription text somehow")
        return None

    print("improved trans_segs_with_ts", trans_segs_with_ts)
    
    
    return trans_segs_with_ts
    
async def process_analysis_job_gpt_diar_gpt2(analysis_job: AnalysisWork) -> None:
    """Process the analysis job using GPT for transcription and MLP-YAN for diarization."""
    try:
        # Get the audio UUID from the analysis job
        audio_uuid = analysis_job.id
        if not audio_uuid:
            raise ValueError("Missing audio_uuid in analysis_job")
            
        # Set audio_uuid on the analysis_job object for consistency
        analysis_job.audio_uuid = audio_uuid
            
        # Now use audio_uuid in path construction
        diar_debug_path = config.TEMP_DIR / "transcripts" / audio_uuid / f"id[{audio_uuid}]_diarization_debug.json"
        
        logger.info(f"Processing analysis job for {audio_uuid} with method: {analysis_job.options['diarization_method']}")
        
        # Step 1: VAD splitting
        await do_vad_split(analysis_job)

        # Step 2 & 3: Run transcription and diarization in parallel
        tasks = []
        
        # Transcription task
        analysis_job.update_status(AudioStatus.TRANSCRIBING)
        transcription_task = asyncio.create_task(
            asyncio.to_thread(run_process_transcription, analysis_job, analysis_job.split_segments), 
            name="transcription"
        )
        tasks.append(transcription_task)
        
        # # Diarization task
        # analysis_job.update_status(AudioStatus.DIARIZING)
        # diarization_task = asyncio.create_task(
        #     asyncio.to_thread(run_diarize_audio, analysis_job.file_path), 
        #     name="diarization"
        # )
        # tasks.append(diarization_task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        result_transcription = results[0]
        # diarization_segments = results[1]
        
        # Check for exceptions
        for result in results:
            if isinstance(result, Exception):
                raise result
            

        # Save diarization results for frontend visualization
        audio_uuid = analysis_job.audio_uuid
        
        # Save frontend-friendly diarization data
        # frontend_diarization = {
        #     "status": AudioStatus.DIARIZING,
        #     "diarization_segments": diarization_segments,
        #     "total_duration": max([seg["end"] for seg in diarization_segments]) if diarization_segments else 0,
        #     "num_speakers": len(set(seg["speaker"] for seg in diarization_segments if "speaker" in seg))
        # }
        
        # Update job step with diarization visualization data for frontend
        # analysis_job.update_step(frontend_diarization)
        
        # # Save debug files for troubleshooting
        # diar_debug_path.parent.mkdir(parents=True, exist_ok=True)
        # with open(diar_debug_path, "w", encoding="utf-8") as f:
        #     json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

        # Also save in uploads directory for frontend access
        # uploads_dir = config.UPLOADS_DIR / audio_uuid
        # uploads_dir.mkdir(parents=True, exist_ok=True)
        # diar_frontend_path = uploads_dir / f"id[{audio_uuid}]_ml_diarized.json"
        # with open(diar_frontend_path, "w", encoding="utf-8") as f:
        #     json.dump(frontend_diarization, f, ensure_ascii=False, indent=2)

        # Apply speaker info from diarization to transcription segments
        transcription_words = result_transcription["segments"]
        
        # Save transcription segments before speaker assignment for debugging
        trans_debug_path = config.TEMP_DIR / "transcripts" / audio_uuid / f"id[{audio_uuid}]_transcription_words_debug.json"
        with open(trans_debug_path, "w", encoding="utf-8") as f:
            json.dump(transcription_words, f, ensure_ascii=False, indent=2)

        ## update script with text improvement
        analysis_job.update_status(AudioStatus.IMPROVING)
        trans_lines = improve_transcription_lines(result_transcription["text"])
        print("improved trans_lines", trans_lines)
        # trans_lines_with_speaker = improve_transcription_lines_with_speaker(result_transcription["text"])

        trans_lines_with_speaker = assign_speaker_to_lines_with_gpt(trans_lines)
        print("improved trans_lines_with_speaker", trans_lines_with_speaker)

        print(f"len trans_lines_with_speaker: {len(trans_lines_with_speaker)}, len trans_lines: {len(trans_lines)}")
        # trans_lines = [line['text'] for line in trans_lines_with_speaker]
        trans_speakers = [line['speaker'] for line in trans_lines_with_speaker]

        ## num of different speakers
        num_speakers = len(set(trans_speakers))
        print(f"num of different speakers: {num_speakers}")

        ## (DBG) save 'improved_lines' to json file
        save_path = config.TEMP_DIR / f"tmp_improved_lines.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(trans_lines, f, ensure_ascii=False, indent=2)
        save_path = config.TEMP_DIR / f"tmp_transcription_words.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(transcription_words, f, ensure_ascii=False, indent=2)
        # save_path = config.TEMP_DIR / f"tmp_diarization_segments.json"
        # with open(save_path, "w", encoding="utf-8") as f:
        #     json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

        print("\ntrans_lines", trans_lines)
        print("\ntranscription_words", transcription_words)
        # print("\ndiarization_segments", diarization_segments)

        trans_segs_with_ts = get_improved_lines_with_ts(trans_lines, transcription_words)
        # print("\ntrans_segs_with_ts", trans_segs_with_ts)
        print(f"len speaker: {len(trans_speakers)}, len trans_lines: {len(trans_lines)}, len transcription_words: {len(transcription_words)} len trans_segs_with_ts: {len(trans_segs_with_ts)}")
        for idx_seg, seg in enumerate(trans_segs_with_ts):
            seg['speaker'] = trans_speakers[idx_seg]


        # diarization_segments = get_cleared_diarization_segments(diarization_segments)
        # save_path = config.TEMP_DIR / f"tmp_cleared_diarization_segments.json"
        # with open(save_path, "w", encoding="utf-8") as f:
        #     json.dump(diarization_segments, f, ensure_ascii=False, indent=2)
        # print("\ncleared_diarization_segments", diarization_segments)

        # trans_segs_with_ts_with_diar = get_seg_ts_with_diar(trans_segs_with_ts, diarization_segments)
        

        print("transcription_segments", trans_segs_with_ts)
        # print("transcription_segments_with_diar", trans_segs_with_ts_with_diar)

        ## merge segments with same speaker continues speaking
        trans_segs_with_ts_conti = []
        prev_seg = None
        for idx_seg, seg in enumerate(trans_segs_with_ts):
            if prev_seg is None:
                prev_seg = seg
                continue
            if prev_seg['speaker'] == seg['speaker']:
                prev_seg['end'] = seg['end']
                prev_seg['text'] += " " + seg['text'].strip()
            else:
                trans_segs_with_ts_conti.append(prev_seg)
                prev_seg = seg
        # AFTER the loop, do this:
        if prev_seg is not None:
            trans_segs_with_ts_conti.append(prev_seg)


        print("transcription_segments_conti", trans_segs_with_ts_conti)

        # Create segments from transcription result with assigned speakers
        segments = []
        speakers = []
        current_speaker_id = 0
        speaker_id_map = {}
        
        # Process segments to create Segment objects with proper speaker IDs
        # for seg in trans_segs_with_ts:
        for seg in trans_segs_with_ts_conti:
        # for seg in segments:
            speaker_id = seg.get("speaker")
            # Skip "undecided" speaker ID in the mapping
            if speaker_id != "undecided" and speaker_id not in speaker_id_map:
                speaker_id_map[speaker_id] = current_speaker_id
                current_speaker_id += 1
            
            segment = Segment(
                id=len(segments),
                start=seg.get("start"),
                end=seg.get("end"),
                text=seg.get("text"),
                # If speaker_id is "undecided", keep it as is, otherwise map it
                speaker="undecided" if speaker_id == "undecided" else speaker_id_map.get(speaker_id, 0)
            )
            segments.append(segment)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        
        # Reassign IDs after sorting
        for i, segment in enumerate(segments):
            segment.id = i
        
        # Create speaker list, making sure the first speaker is the counselor
        speakers = [
            Speaker(id=id, role="counselor" if id == 0 else f"client {id}")
            for id in range(len(speaker_id_map))
        ]
        
        # Create the final transcription result with segments and speakers
        analysis_job.result = TranscriptionResult(
            segments=segments,
            speakers=speakers
        )
        
        # Mark the improving step as completed
        analysis_job.complete_step("improving")

        print("will return result:", analysis_job.result)
        
        # Mark the job as completed
        analysis_job.update_status(AudioStatus.COMPLETING)
    
    except Exception as e:
        # Handle errors
        err_msg = f"ERROR in process_analysis_job_gpt_diar_mlpyan: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        analysis_job.error = str(e)
        analysis_job.update_status(AudioStatus.FAILED)

async def process_analysis_job_gpt_diar_mlpyan(analysis_job: AnalysisWork, is_display: bool = False) -> None:
    
    """Process the analysis job using GPT for transcription and MLP-YAN for diarization."""
    try:
        # Get the audio UUID from the analysis job
        audio_uuid = analysis_job.id
        if not audio_uuid:
            raise ValueError("Missing audio_uuid in analysis_job")
            
        # Set audio_uuid on the analysis_job object for consistency
        analysis_job.audio_uuid = audio_uuid
            
        # Now use audio_uuid in path construction
        diar_debug_path = config.TEMP_DIR / "transcripts" / audio_uuid / f"id[{audio_uuid}]_diarization_debug.json"
        
        logger.info(f"Processing analysis job for {audio_uuid} with method: {analysis_job.options['diarization_method']}")
        
        # Step 1: VAD splitting
        await do_vad_split(analysis_job)

        # Step 2 & 3: Run transcription and diarization in parallel
        tasks = []
        
        # Transcription task
        analysis_job.update_status(AudioStatus.TRANSCRIBING)
        transcription_task = asyncio.create_task(
            asyncio.to_thread(run_process_transcription, analysis_job, analysis_job.split_segments), 
            name="transcription"
        )
        tasks.append(transcription_task)
        
        # Diarization task
        analysis_job.update_status(AudioStatus.DIARIZING)
        diarization_task = asyncio.create_task(
            asyncio.to_thread(run_diarize_audio, analysis_job.file_path), 
            name="diarization"
        )
        tasks.append(diarization_task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        result_transcription = results[0]
        diarization_segments = results[1]
        
        # Check for exceptions
        for result in results:
            if isinstance(result, Exception):
                raise result
            
        # Save diarization results for frontend visualization
        audio_uuid = analysis_job.audio_uuid
        
        # Save frontend-friendly diarization data
        frontend_diarization = {
            "status": AudioStatus.DIARIZING,
            "diarization_segments": diarization_segments,
            "total_duration": max([seg["end"] for seg in diarization_segments]) if diarization_segments else 0,
            "num_speakers": len(set(seg["speaker"] for seg in diarization_segments if "speaker" in seg))
        }
        
        # Update job step with diarization visualization data for frontend
        analysis_job.update_step(frontend_diarization)
        
        # Save debug files for troubleshooting
        diar_debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diar_debug_path, "w", encoding="utf-8") as f:
            json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

        # Also save in uploads directory for frontend access
        uploads_dir = config.UPLOADS_DIR / audio_uuid
        uploads_dir.mkdir(parents=True, exist_ok=True)
        diar_frontend_path = uploads_dir / f"id[{audio_uuid}]_ml_diarized.json"
        with open(diar_frontend_path, "w", encoding="utf-8") as f:
            json.dump(frontend_diarization, f, ensure_ascii=False, indent=2)

        # Apply speaker info from diarization to transcription segments
        transcription_words = result_transcription["segments"]
        
        # Save transcription segments before speaker assignment for debugging
        trans_debug_path = config.TEMP_DIR / "transcripts" / audio_uuid / f"id[{audio_uuid}]_transcription_words_debug.json"
        with open(trans_debug_path, "w", encoding="utf-8") as f:
            json.dump(transcription_words, f, ensure_ascii=False, indent=2)

        ## update script with text improvement
        analysis_job.update_status(AudioStatus.IMPROVING)
        trans_lines = improve_transcription_lines(result_transcription["text"])

        ## (DBG) save 'improved_lines' to json file
        save_path = config.TEMP_DIR / f"tmp_improved_lines.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(trans_lines, f, ensure_ascii=False, indent=2)
        save_path = config.TEMP_DIR / f"tmp_transcription_words.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(transcription_words, f, ensure_ascii=False, indent=2)
        save_path = config.TEMP_DIR / f"tmp_diarization_segments.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

        print("\ntrans_lines", trans_lines)
        print("\ntranscription_words", transcription_words)
        print("\ndiarization_segments", diarization_segments)

        trans_segs_with_ts = get_improved_lines_with_ts(trans_lines, transcription_words)
        print("\ntrans_segs_with_ts", trans_segs_with_ts)


        if is_display:
            # with matplotlib, display to timeline.
            # - 1. polot the diarization segments, along x axis is time, y axis is speaker.
            # - 2. plot the transcription segments as text labels. 'start' time is label position. rotate 90 degrees.

            pass


        diarization_segments = get_cleared_diarization_segments(diarization_segments)
        save_path = config.TEMP_DIR / f"tmp_cleared_diarization_segments.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(diarization_segments, f, ensure_ascii=False, indent=2)
        print("\ncleared_diarization_segments", diarization_segments)

        trans_segs_with_ts_with_diar = get_seg_ts_with_diar(trans_segs_with_ts, diarization_segments)
        

        print("transcription_segments", trans_segs_with_ts)
        print("transcription_segments_with_diar", trans_segs_with_ts_with_diar)

        # Create segments from transcription result with assigned speakers
        segments = []
        speakers = []
        current_speaker_id = 0
        speaker_id_map = {}
        
        # Process segments to create Segment objects with proper speaker IDs
        for seg in trans_segs_with_ts_with_diar:
            speaker_id = seg.get("speaker")
            # Skip "undecided" speaker ID in the mapping
            if speaker_id != "undecided" and speaker_id not in speaker_id_map:
                speaker_id_map[speaker_id] = current_speaker_id
                current_speaker_id += 1
            
            segment = Segment(
                id=len(segments),
                start=seg.get("start"),
                end=seg.get("end"),
                text=seg.get("text"),
                # If speaker_id is "undecided", keep it as is, otherwise map it
                speaker="undecided" if speaker_id == "undecided" else speaker_id_map.get(speaker_id, 0)
            )
            segments.append(segment)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        
        # Reassign IDs after sorting
        for i, segment in enumerate(segments):
            segment.id = i
        
        # Create speaker list, making sure the first speaker is the counselor
        speakers = [
            Speaker(id=id, role="counselor" if id == 0 else "client")
            for id in range(len(speaker_id_map))
        ]
        
        # Create the final transcription result with segments and speakers
        analysis_job.result = TranscriptionResult(
            segments=segments,
            speakers=speakers
        )
        
        # Mark the improving step as completed
        analysis_job.complete_step("improving")
        
        # Mark the job as completed
        analysis_job.update_status(AudioStatus.COMPLETING)
    
    except Exception as e:
        # Handle errors
        err_msg = f"ERROR in process_analysis_job_gpt_diar_mlpyan: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        analysis_job.error = str(e)
        analysis_job.update_status(AudioStatus.FAILED)

async def process_analysis_job_asm_diar_asm(analysis_job: AnalysisWork) -> None:
    logger.info(f"Processing analysis job for {analysis_job.audio_uuid} with method: {analysis_job.options['diarization_method']}")
    await do_vad_split(analysis_job)

    ## add tasks
    tasks = []
    analysis_job.update_status(AudioStatus.TRANSCRIBING)
    transcription_task = asyncio.create_task(asyncio.to_thread(run_process_transcription, analysis_job, analysis_job.split_segments), name="transcription")
    tasks.append(transcription_task)

    # Wait for tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("asyncio results", results)

    result_transcription = results[0]
    transcription_segments = result_transcription["segments"]

    # Store diarization result (in this method, the diarization is done in the transcription)
    # We still need to store an intermediate result for the front end
    speaker_segments = []
    speaker_ids = set()
    for i, seg in enumerate(transcription_segments):
        speaker_id = seg.get("speaker", 0)
        speaker_ids.add(speaker_id)
        speaker_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker_id
        })
    
    # Create speakers list
    speakers = [
        {"id": speaker_id, "role": "counselor" if speaker_id == 0 else "client"}
        for speaker_id in sorted(speaker_ids)
    ]
    
    # Store diarization result
    analysis_job.store_step_result("diarization", {
        "speakers": speakers,
        "segments": speaker_segments
    })
    
    # Update status to improving
    analysis_job.update_status(AudioStatus.IMPROVING)

    # Create segments
    segments = [
        Segment(
            id=i,
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            # Handle the case where speaker might be "undecided"
            speaker=seg["speaker"],
            speaker_diarized=""
        ) for i, seg in enumerate(transcription_segments)
    ]
        
    # Sort segments by start time
    segments.sort(key=lambda x: x.start)

    ## save the result
    analysis_job.result = TranscriptionResult(
        text=result_transcription["text"],
        segments=segments,
        speakers=[]
    )
    
    # Store improving result (complete result with speaker identification)
    improving_segments = []
    for i, seg in enumerate(segments):
        improving_segments.append({
            "id": i,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker": seg.speaker
        })
    
    # Store the improving result
    analysis_job.store_step_result("improving", {
        "segments": improving_segments
    })
    
    save_transcription_result_json(analysis_job.audio_uuid, analysis_job.result, suffix="transcript")
    save_analysis_work_json(analysis_job)

    pass

async def process_audio_file(analysis_job: AnalysisWork, background_tasks: BackgroundTasks) -> None:
    """Process the audio file for the analysis job."""
    try:

        print(f"Processing audio file: {analysis_job.id} {analysis_job.status} {analysis_job.steps}")
        # Don't set a generic PROCESSING status
        # Just go directly to the first real step - VAD splitting
        await do_vad_split(analysis_job)
        
        # Get the diarization method from options
        diarization_method = analysis_job.options.get("diarization_method", "")
        
        # Process the audio file based on the diarization method
        if diarization_method == "stt_apigpt_diar_apigpt":
            await process_analysis_job_gpt_diar_gpt(analysis_job)
        elif diarization_method == "stt_apigpt_diar_apigpt2":
            await process_analysis_job_gpt_diar_gpt2(analysis_job)
        elif diarization_method == "stt_apiasm_diar_apiasm":
            await process_analysis_job_asm_diar_asm(analysis_job)
        elif diarization_method == "stt_apigpt_diar_mlpyan":
            await process_analysis_job_gpt_diar_mlpyan(analysis_job)
        else:
            # Default to GPT-MLPYAN
            await process_analysis_job_gpt_diar_mlpyan(analysis_job)
            
    except Exception as e:
        # Handle errors
        analysis_job.error = f"Error processing audio file: {e}"
        analysis_job.update_status(AudioStatus.FAILED)
        # Log the error
        err_msg = f"ERROR in process_audio_file: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)

async def process_transcription(analysis_job: AnalysisWork, split_segments: List[Dict]) -> Dict:
    """Process transcription for all split segments."""
    total_splits = len(split_segments)
    segments_list = [None] * total_splits
    text_list = [None] * total_splits
    # full_text = [None] * total_splits
    processed_splits = 0
    diarization_method = analysis_job.options["diarization_method"]
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_index = {
            executor.submit(
                transcribe_segment, 
                i, 
                seg["path"] if isinstance(seg, dict) else seg,  # Handle both formats
                total_splits, 
                diarization_method,
                analysis_job.options  # Pass options to transcribe_segment
            ): i 
            for i, seg in enumerate(split_segments)
        }
        
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                if result:
                    if result.get("is_coroutine", False):
                        try:
                            api_result = await result["coroutine"]
                            if os.path.exists(result["temp_path"]):
                                os.unlink(result["temp_path"])
                            result["segments"] = api_result["segments"]
                            result["text"] = api_result["text"]
                        except Exception as e:
                            logger.error(f"Error awaiting API result: {e}\n{traceback.format_exc()}")
                            continue
                    
                    # Calculate time offset based on segment format
                    time_offset = sum(
                        len(AudioSegment.from_file(s["path"] if isinstance(s, dict) else s)) / 1000 
                        for s in split_segments[:result["index"]]
                    )
                    
                    for segment in result["segments"]:
                        segment["start"] += time_offset
                        segment["end"] += time_offset
                        # all_segments.append(segment)
                        if segments_list[result["index"]] is None:
                            segments_list[result["index"]] = []
                        segments_list[result["index"]].append(segment)
                    
                    # full_text.append(result["text"])
                    text_list[result["index"]] = result["text"]
                    
                    # Update progress
                    processed_splits += 1
                    analysis_job.update_step({
                        "step_name": "transcribing",  # Always include explicit step_name
                        "status": "in_progress",
                        "total_splits": total_splits,
                        "processed_splits": processed_splits,
                        "percent_complete": (processed_splits / total_splits) * 100
                    })
                    
            except Exception as e:
                logger.error(f"Error processing future result: {e}\n{traceback.format_exc()}")
                continue
    
    all_segments = []
    full_text = []
    for i in range(total_splits):
        if segments_list[i] is not None:
            all_segments.extend(segments_list[i])
        if text_list[i] is not None:
            full_text.append(text_list[i])

    # Create transcription result
    transcription_result = {
        "segments": all_segments,
        "text": " ".join(full_text)
    }
    
    # Store transcription result as intermediate result
    transcription_segments = []
    for i, seg in enumerate(all_segments):
        # Add "undecided" as speaker for transcription step
        transcription_segments.append({
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": "undecided"
        })
    
    # Store the transcription result in the step_results
    analysis_job.store_step_result("transcription", {
        "segments": transcription_segments
    })
    
    # After all splits are processed, mark transcribing as completed
    analysis_job.complete_step("transcribing")
    
    # Create the full text from the segments
    full_text_str = " ".join(full_text)
    
    return {
        "segments": transcription_segments,
        "text": full_text_str  # Add the text key to the returned dictionary
    }

def transcribe_segment(i, split_path, total_splits, diarization_method, options=None):
    """ Transcribes a single audio segment and returns the result. """
    logger.info(f"Processing split {i+1}/{total_splits}: {split_path}")
    
    try:
        
        # Use the WAV file directly since it's already in the correct format
        # if diarization_method == "stt_apigpt_diar_apigpt":
            
        if diarization_method == "stt_apiasm_diar_apiasm":
            logger.info(f"Using Assembly AI API for transcription of split {i+1} / {total_splits}")
            api_result = asyncio.run(process_audio_with_assembly_ai(split_path, i, total_splits))
        else:
            logger.info(f"Using OpenAI API for transcription of split {i+1} / {total_splits}")
            # If options are not provided, try to get them from the analysis job
            if options is None:
                audio_uuid = os.path.basename(os.path.dirname(split_path))
                analysis_job = analysis_jobs.get(audio_uuid)
                if analysis_job and hasattr(analysis_job, 'options'):
                    options = analysis_job.options
            
            api_result = asyncio.run(process_audio_with_openai_api(split_path, i, total_splits, options))
        
        result = {
            "index": i,
            "segments": api_result["segments"],
            "text": api_result["text"],
            "is_coroutine": False
        }
        
        # Save transcription result as JSON
        save_transcription_result(split_path, result)
        
        return result
    except Exception as split_error:
        logger.error(f"ERROR processing split {i+1}: {split_error}\n{traceback.format_exc()}")
        return None  # Continue processing other segments

def save_transcription_result(split_path, result):
    """
    Save transcription result as JSON file.
    
    Args:
        split_path: Path to the original split audio file
        result: Transcription result dictionary
    """
    try:
        # Extract audio_uuid from split_path
        split_dir = os.path.dirname(split_path)
        audio_uuid = os.path.basename(split_dir)
        
        # Create directory for transcripts if it doesn't exist
        transcript_dir = config.TEMP_DIR / "transcripts" / audio_uuid
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSON filename based on split index
        split_idx = result["index"]
        json_filename = f"id[{audio_uuid}]_split[{split_idx}].json"
        json_path = transcript_dir / json_filename
        
        # Save result as JSON
        with open(json_path, "w", encoding="utf-8") as f:
            # Remove is_coroutine from the saved result
            save_result = {k: v for k, v in result.items() if k != "is_coroutine"}
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved transcription result to {json_path}")
    except Exception as e:
        err_msg = f"ERROR in save_transcription_result: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        # Continue even if saving fails

if __name__ == "__main__":
    import whisper
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import os
    from pathlib import Path
    
    print("Starting simplified audio analysis...")
    
    # Set file path
    file_path = '/Users/beaver.baek/Documents/audio_samples/audio-test_short_34sec.mp3'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        exit(1)
        
    print(f"Processing file: {file_path}")
    
    # Load audio file
    print("Loading audio file...")
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Transcribe with Whisper
    print("Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    segments = result["segments"]
    
    print("\nTranscription:")
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f} - {end:.2f}] {text}")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Plot transcription timeline
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 1)
    ax2.set_title('Transcription Timeline')
    ax2.set_xlabel('Time (s)')
    ax2.get_yaxis().set_visible(False)
    
    # Add text annotations for segments
    text_height = 0.8
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"][:50] + "..." if len(segment["text"]) > 50 else segment["text"]
        
        # Draw segment boundaries
        ax2.axvline(x=start, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(x=end, color='r', linestyle='--', alpha=0.5)
        
        # Add text annotation
        ax2.text(start, text_height, text, fontsize=8, 
                horizontalalignment='left', verticalalignment='center')
        
        # Draw line to indicate segment
        ax2.plot([start, end], [0.5, 0.5], linewidth=2, color='black')
        
        # Alternate text height for better readability
        text_height = 0.2 if text_height == 0.8 else 0.8
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"analysis_visualization_{os.path.basename(file_path)}.png"
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    
    # Show figure
    plt.show()
    
    print("Analysis complete.")
    
    pass
