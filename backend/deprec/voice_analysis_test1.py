#!/usr/bin/env python3
"""
Voice Analysis Test Module

This script tests the voice analysis capabilities by processing an audio file,
transcribing it using Whisper, and displaying the transcription results.
It uses the functions from the backend.logic.voice_analysis module and
visualizes the results using matplotlib.
"""

import os
import sys
import uuid
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import librosa
import librosa.display
import whisper
import soundfile as sf
import traceback  # Explicitly import traceback
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
import torch
from omegaconf import OmegaConf

# Import simple_diarizer
from simple_diarizer.diarizer import Diarizer

# Import pyannote for speaker diarization
try:
    from pyannote.audio import Pipeline
    has_pyannote = True
except ImportError:
    has_pyannote = False
    print("Pyannote not installed. To install: pip install pyannote.audio")

# Import NeMo for speaker diarization
try:
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    has_nemo = True
except ImportError:
    has_nemo = False
    print("NeMo not installed. To install: pip install nemo_toolkit[asr]")

# Setup Python path to include backend directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# switch
is_use_openai_whisper = True
is_use_first_some_min = True
is_use_pyannote = True  # Flag for pyannote
is_use_nemo = True      # Flag for NeMo
max_mins = 1.5
max_char_in_label = 15

# Import functions from voice_analysis module
from backend.logic.voice_analysis import convert_audio_for_whisper, postprocess_segments, get_openai_client
from backend.logic.models import AnalysisWork, AudioStatus, TranscriptionResult, Segment, Speaker
from backend.config import DEVICE

def create_analysis_job(audio_file):
    """
    Create an analysis job for the given audio file.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        AnalysisWork: The created analysis job
    """
    # Generate a unique ID for the job
    job_id = str(uuid.uuid4())
    
    # Create analysis job with initial status
    job = AnalysisWork(
        id=job_id,
        filename=os.path.basename(audio_file),
        total_chunks=1,  # We're not chunking the file for this test
        status=AudioStatus.PENDING
    )
    
    # Set the file path
    job.file_path = audio_file
    
    # Set options
    job.options = {
        "diarization_method": "whisper_diarize",  # Simple diarization method
        "is_limit_time": False,
        "limit_time_sec": 600,
        "is_merge_segments": True
    }
    
    return job

def run_simple_diarizer(file_path, num_speakers=2):
    """
    Run the simple_diarizer on the audio file
    
    Args:
        file_path: Path to the audio file
        num_speakers: Expected number of speakers
        
    Returns:
        List of diarization segments
    """
    print("\nRunning simple_diarizer...")
    diar = Diarizer(
        embed_model='xvec',  # 'xvec' or 'ecapa'
        cluster_method='sc'  # 'ahc' or 'sc'
    )
    
    # Ensure file path is a wav file
    if not file_path.endswith('.wav'):
        # Convert to WAV if needed
        print(f"Converting {file_path} to WAV for diarization...")
        audio = AudioSegment.from_file(file_path)
        temp_wav = "temp_diarization.wav"
        audio.export(temp_wav, format="wav")
        file_path = temp_wav
    
    # Run diarization
    segments = diar.diarize(file_path, num_speakers=num_speakers)
    
    print(f"Simple diarizer found {len(segments)} segments with {num_speakers} speakers.")
    return segments

def run_pyannote_diarizer(file_path):
    """
    Run the pyannote speaker diarization on the audio file
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        List of diarization segments
    """
    if not has_pyannote:
        print("Pyannote not available. Skipping pyannote diarization.")
        return []
    
    print(f"\nRunning pyannote speaker diarization on device: {DEVICE}...")
    
    try:
        # Validate and convert audio file
        print("Validating audio format for pyannote...")
        
        # Load audio file and check properties
        audio = AudioSegment.from_file(file_path)
        print(f"Original audio properties:")
        print(f"- Sample rate: {audio.frame_rate}Hz")
        print(f"- Channels: {audio.channels}")
        print(f"- Duration: {len(audio)/1000:.2f}s")
        
        needs_conversion = False
        if audio.frame_rate != 16000:
            print(f"Converting sample rate from {audio.frame_rate}Hz to 16000Hz")
            needs_conversion = True
        
        if audio.channels != 1:
            print(f"Converting from {audio.channels} channels to mono")
            needs_conversion = True
        
        if needs_conversion or not file_path.endswith('.wav'):
            print("Converting audio to required format...")
            temp_wav = "temp_pyannote_diarization.wav"
            
            # Convert to required format
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export as 16-bit PCM WAV
            audio.export(
                temp_wav,
                format="wav",
                parameters=["-acodec", "pcm_s16le"]  # Ensure 16-bit PCM
            )
            
            # Verify the converted file
            converted_audio = AudioSegment.from_file(temp_wav)
            print(f"Converted audio properties:")
            print(f"- Sample rate: {converted_audio.frame_rate}Hz")
            print(f"- Channels: {converted_audio.channels}")
            print(f"- Duration: {len(converted_audio)/1000:.2f}s")
            
            file_path = temp_wav
        
        # Load the pyannote speaker diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv('HF_TOKEN')
        ).to(torch.device(DEVICE))
        
        # Apply the pipeline to the audio file
        diarization = pipeline(file_path)
        
        # Convert pyannote output to a list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'label': int(speaker.split("_")[1]) if "_" in speaker else 0
            })
        
        print(f"Pyannote diarizer found {len(segments)} segments")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        # Print first few segments for debugging
        print("\nFirst few segments:")
        for i, seg in enumerate(segments[:5]):
            print(f"Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s (Speaker {seg['label']})")
        
        return segments
    
    except Exception as e:
        print(f"Error running pyannote diarization: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

def create_nemo_config(output_dir):
    """
    Create configuration for NeMo Neural Diarizer based on reference implementation
    
    Args:
        output_dir: Directory for NeMo outputs
        
    Returns:
        OmegaConf configuration for NeMo
    """
    from omegaconf import OmegaConf
    
    # Domain type can be meeting or telephonic
    DOMAIN_TYPE = "telephonic"
    
    cfg = {
        "device": DEVICE,
        "diarizer": {
            "manifest_filepath": os.path.join(output_dir, "input_manifest.json"),
            "out_dir": output_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "infer_overlap": False,
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": 1.5,
                    "shift_length_in_sec": 0.75,
                    "multiscale_weights": [0.33, 0.33, 0.33],
                    "save_embeddings": False
                }
            },
            "clustering": {
                "parameters": {
                    "method": "spectral",
                    "oracle_num_speakers": False,
                    "max_num_speakers": 8 if DOMAIN_TYPE == "meeting" else 2,
                    "enhancement": "asp",
                    "max_rp_threshold": 0.15,
                    "sparse_search_volume": 30
                }
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic" if DOMAIN_TYPE == "telephonic" else "diar_msdd_meeting",
                "parameters": {
                    "overlap_infer_spk_limit": 8 if DOMAIN_TYPE == "meeting" else 2,
                    "use_speaker_model_from_ckpt": True,
                    "fix_num_spks": False,
                    "soft_label_thres": 0.5
                }
            }
        }
    }
    
    return OmegaConf.create(cfg)

def run_nemo_diarizer(file_path):
    """
    Run the NeMo Neural Diarizer on the audio file
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        List of diarization segments
    """
    if not has_nemo:
        print("NeMo not available. Skipping NeMo diarization.")
        return []
    
    print(f"\nRunning NeMo speaker diarization on device: {DEVICE}...")
    
    try:
        # Import required modules for NeMo
        from omegaconf import OmegaConf
        import json
        
        # Validate and convert audio file
        print("Validating audio format for NeMo...")
        
        # Load audio file and check properties
        audio = AudioSegment.from_file(file_path)
        print(f"Original audio properties:")
        print(f"- Sample rate: {audio.frame_rate}Hz")
        print(f"- Channels: {audio.channels}")
        print(f"- Duration: {len(audio)/1000:.2f}s")
        
        # Create temp directory for NeMo outputs
        temp_path = os.path.join(os.getcwd(), "temp_nemo_outputs")
        os.makedirs(temp_path, exist_ok=True)
        
        # Convert to mono for NeMo compatibility
        if audio.channels != 1:
            print(f"Converting from {audio.channels} channels to mono for NeMo")
            audio = audio.set_channels(1)
        
        # Export the audio file for NeMo
        mono_file_path = os.path.join(temp_path, "mono_file.wav")
        audio.export(mono_file_path, format="wav")
        
        # Create input manifest for NeMo
        with open(os.path.join(temp_path, "input_manifest.json"), "w") as f:
            json.dump({"audio_filepath": mono_file_path, "offset": 0, "duration": len(audio) / 1000, "label": "infer", "text": "-"}, f)
        
        # Initialize NeMo diarization model
        config = create_nemo_config(temp_path)
        
        # Print out the config for debugging
        print("NeMo config:", OmegaConf.to_yaml(config))
        
        # Initialize the NeMo diarizer with proper config
        # msdd_model = NeuralDiarizer(cfg=config).to(torch.device(DEVICE))
        msdd_model = NeuralDiarizer(cfg=config)
        
        # Run diarization
        msdd_model.diarize()
        
        # Read the diarization result
        rttm_file = os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
        if not os.path.exists(rttm_file):
            print(f"RTTM file not found at {rttm_file}")
            return []
        
        # Parse RTTM file to extract diarization results
        segments = []
        with open(rttm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10 and parts[0] == 'SPEAKER':
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    speaker = int(parts[7])
                    segments.append({
                        'start': start_time,
                        'end': start_time + duration,
                        'label': speaker
                    })
        
        print(f"NeMo diarizer found {len(segments)} segments")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        # Print first few segments for debugging
        print("\nFirst few segments:")
        for i, seg in enumerate(segments[:5]):
            print(f"Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s (Speaker {seg['label']})")
        
        return segments
    
    except Exception as e:
        print(f"Error running NeMo diarization: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

def analyze_audio(file_path):
    """
    Process an audio file using the voice analysis pipeline.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        dict: Analysis results
    """
    print("Starting audio analysis...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Create analysis job
    job = create_analysis_job(file_path)
    print(f"Created analysis job with ID: {job.id}")
    
    try:
        # Update job status to processing
        job.update_status(AudioStatus.SPLITTING)
        print(f"Processing file: {file_path}")
        
        # Convert audio for Whisper if needed
        print("Converting audio for Whisper...")
        try:
            converted_path = convert_audio_for_whisper(Path(file_path))
            print(f"Converted audio to: {converted_path}")
        except Exception as e:
            print(f"Error converting audio: {e}")
            converted_path = file_path  # Fall back to original file
        
        # Load audio file
        print("Loading audio file...")
        audio_data = AudioSegment.from_file(converted_path)
        duration_ms = len(audio_data)
        duration = duration_ms / 1000  # Convert to seconds
        print(f"Original audio duration: {duration:.2f} seconds")
        
        # If is_use_first_some_min is True and audio exceeds 5 minutes, trim it
        
        if is_use_first_some_min and duration > max_mins * 60:
            print(f"Audio exceeds {max_mins} minutes and is_use_first_some_min is True. Trimming to first {max_mins} minutes.")
            audio_data = audio_data[:max_mins * 60 * 1000]  # Trim to 5 minutes (300,000 ms)
            trimmed_path = os.path.join("temp", f"trimmed_{os.path.basename(converted_path)}")
            audio_data.export(trimmed_path, format=os.path.splitext(converted_path)[1][1:])
            converted_path = trimmed_path
            duration = max_mins * 60  # Update duration to 5 minutes
            print(f"Trimmed audio saved to: {converted_path}")
        
        # Load audio for visualization with librosa
        audio, sr = librosa.load(converted_path, sr=16000)
        
        # Update job status to transcribing
        job.update_status(AudioStatus.TRANSCRIBING)
        
        # Transcribe with Whisper based on the flag
        print("Transcribing with Whisper (this may take a moment)...")
        
        # Store words for waveform annotation
        word_timestamps = []
        
        if is_use_openai_whisper:
            # Use OpenAI Whisper API
            print("Using OpenAI Whisper API with word-level timestamps...")
            try:
                # Get OpenAI client
                client = get_openai_client()
                
                # Use OpenAI's Whisper API with word-level timestamps
                with open(converted_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )
                
                # Save word timestamps for waveform annotation
                if hasattr(response, 'words') and response.words is not None:
                    word_timestamps = [
                        {"word": word.word, "start": word.start, "end": word.end - 0.01}
                        for word in response.words
                    ]
                
                # Format result to match local Whisper output structure
                result = {
                    "text": response.text,
                    "language": response.language,
                    "segments": []
                }
                
                # Check if words exist in the response
                if hasattr(response, 'words') and response.words is not None and len(response.words) > 0:
                    # Group words into segments based on pauses or punctuation
                    current_segment = {
                        "id": 0,
                        "start": response.words[0].start,
                        "end": None,
                        "text": ""
                    }
                    
                    for i, word in enumerate(response.words):
                        # Add word to current segment text
                        if current_segment["text"]:
                            current_segment["text"] += " " + word.word.strip()
                        else:
                            current_segment["text"] = word.word.strip()
                        
                        # Update end time
                        current_segment["end"] = word.end
                        
                        # Check if we should start a new segment
                        # Create a new segment if this word ends with sentence-ending punctuation
                        # or if there's a significant pause (e.g., > 0.5 seconds to next word)
                        is_end_of_sentence = word.word.strip().endswith(('.', '!', '?'))
                        is_long_pause = (i < len(response.words) - 1 and 
                                        response.words[i+1].start - word.end > 0.5)
                        
                        if is_end_of_sentence or is_long_pause or i == len(response.words) - 1:
                            # Add current segment to results and start a new one
                            result["segments"].append(current_segment)
                            
                            if i < len(response.words) - 1:
                                current_segment = {
                                    "id": len(result["segments"]),
                                    "start": response.words[i+1].start,
                                    "end": None,
                                    "text": ""
                                }
                else:
                    # If no words are provided, create a single segment with the full text
                    print("No words found in API response, creating a single segment")
                    result["segments"] = [{
                        "id": 0,
                        "start": 0,
                        "end": duration,  # Use the full duration
                        "text": response.text
                    }]
                
                print(f"Detected language: {result['language']}")
                print(f"Created {len(result['segments'])} segments from word timestamps")
            
            except Exception as e:
                print(f"Error using OpenAI Whisper API: {e}")
                print(f"Falling back to local Whisper model on device: {DEVICE}...")
                # Fall back to local Whisper if API fails
                model = whisper.load_model("base").to(torch.device(DEVICE))
                result = model.transcribe(converted_path)
        else:
            # Use local Whisper model with device configuration
            print(f"Using local Whisper model on device: {DEVICE}...")
            model = whisper.load_model("base").to(torch.device(DEVICE))
            result = model.transcribe(converted_path)
        
        # Display transcription results
        print("\nTranscription Results:")
        print(f"Detected language: {result.get('language', 'unknown')}")
        
        # Check if language_probability exists in the result
        if 'language_probability' in result:
            print(f"Language probability: {result['language_probability']:.4f}")
        
        # Format segments for further processing
        segments = []
        print("\nSegments:")
        for i, segment in enumerate(result["segments"]):
            # Format segment for postprocessing
            seg = {
                "id": i,
                "start": segment["start"],
                "end": segment["end"],
                "text_raw": segment["text"],  # Store original text as text_raw
            }
            segments.append(seg)
            print(f"Segment {i+1}: [{segment['start']:.2f}s - {segment['end']:.2f}s]  {segment['text']}")
        
        # Update job status to diarizing
        job.update_status(AudioStatus.DIARIZING)
        
        # Process segments with speaker diarization
        try:
            print("\nDiarizing segments (assigning speakers)...")
            processed_segments = postprocess_segments(segments)
            
            # Display diarized segments
            print("\nDiarized Segments:")
            for i, segment in enumerate(processed_segments):
                speaker_label = "Counselor" if segment["speaker"] == 0 else f"Client {segment['speaker']}"
                print(f"Segment {i+1}: [{segment['start']:.2f}s - {segment['end']:.2f}s]  Speaker: {speaker_label}")
                print(f"  Text: {segment['text']}")
                if 'text_raw' in segment and segment['text_raw'] != segment['text']:
                    print(f"  Original: {segment['text_raw']}")
            
            # Save the segments to the job
            job.step_results["transcription"] = {
                "segments": processed_segments,
                "text": " ".join([seg["text"] for seg in processed_segments])
            }
        except Exception as e:
            print(f"Error during diarization: {e}")
            # Fall back to using original segments without diarization
            processed_segments = []
            for i, segment in enumerate(result["segments"]):
                processed_segments.append({
                    "id": i,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": i % 2  # Alternate speakers as a simple fallback
                })
            job.step_results["transcription"] = {
                "segments": processed_segments,
                "text": result["text"]
            }
        
        # Run simple_diarizer on the same file
        try:
            # Estimate number of speakers from our processed segments
            num_speakers = len(set(seg["speaker"] for seg in processed_segments))
            # Ensure at least 2 speakers
            num_speakers = max(2, num_speakers)
            
            # Run the simple_diarizer
            simple_diarizer_segments = run_simple_diarizer(converted_path, num_speakers)
        except Exception as e:
            print(f"Error running simple_diarizer: {e}")
            simple_diarizer_segments = []
        
        # Run pyannote diarization if enabled
        pyannote_segments = []
        if is_use_pyannote:
            try:
                pyannote_segments = run_pyannote_diarizer(converted_path)
            except Exception as e:
                print(f"Error running pyannote diarization: {e}")
        
        # Run NeMo diarization if enabled
        nemo_segments = []
        if is_use_nemo:
            try:
                nemo_segments = run_nemo_diarizer(converted_path)
            except Exception as e:
                print(f"Error running NeMo diarization: {e}")
        
        # Create visualization
        print("\nCreating visualization...")
        
        # Setup fonts for visualization
        font_candidates = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Arial Unicode MS', 'NanumMyeongjo']
        font_found = False
        
        for font_name in font_candidates:
            if any(font_name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = font_name
                font_found = True
                print(f"Using font: {font_name}")
                break
        
        if not font_found:
            print("Warning: No suitable Korean font found. Some characters may not display correctly.")
            
        # Alternative method for Korean fonts
        try:
            # For macOS
            if sys.platform == 'darwin':
                mpl.rc('font', family='AppleGothic')
            # For Windows
            elif sys.platform == 'win32':
                mpl.rc('font', family='Malgun Gothic')
        except:
            pass
        
        # Create figure with 5 subplots and an additional subplot for the scrollbar
        fig = plt.figure(figsize=(14, 20))  # Increase height to accommodate new row
        gs = fig.add_gridspec(6, 1, height_ratios=[4, 4, 4, 4, 4, 1])  # 6 rows, last one for scrollbar

        # Create the main plotting axes
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])
        ax5 = fig.add_subplot(gs[4])  # New axis for NeMo
        ax_scroll = fig.add_subplot(gs[5])  # Scrollbar axis

        # Set the default view range
        view_window = 30  # seconds
        view_start = 0
        view_end = min(view_window, duration)

        # Function to update all plots when scrolling
        def update_view(val):
            view_start = val
            view_end = min(val + view_window, duration)
            
            # Update xlim for all plots
            ax1.set_xlim(view_start, view_end)
            ax2.set_xlim(view_start, view_end)
            ax3.set_xlim(view_start, view_end)
            ax4.set_xlim(view_start, view_end)
            ax5.set_xlim(view_start, view_end)  # Add NeMo axis
            
            fig.canvas.draw_idle()

        # Set initial view limits for all plots
        ax1.set_xlim(view_start, view_end)
        ax2.set_xlim(view_start, view_end)
        ax3.set_xlim(view_start, view_end)
        ax4.set_xlim(view_start, view_end)
        ax5.set_xlim(view_start, view_end)  # Add NeMo axis

        # Create a slider
        from matplotlib.widgets import Slider
        slider = Slider(ax_scroll, 'Time', 0, max(0, duration - view_window), 
                        valinit=0, valstep=1)
        slider.on_changed(update_view)

        # Plot audio waveform
        ax1.set_title("Audio Waveform with Transcription")
        librosa.display.waveshow(y=audio, sr=sr, ax=ax1)
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Amplitude")

        # Plot segments with different colors for speakers
        ax2.set_title("OpenAI-based Diarization (LLM only)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_yticks([])  # Hide Y-axis
        
        # Define colors for speakers
        speaker_colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01']
        
        # Plot segments with colors based on speaker
        for segment in processed_segments:
            speaker = segment.get("speaker", 0)
            color = speaker_colors[speaker % len(speaker_colors)]
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            
            # Draw segment
            ax2.barh(0, end-start, left=start, height=0.5, color=color, alpha=0.6)
            
            # Add speaker label and text
            speaker_label = "Counselor" if speaker == 0 else f"Client {speaker}"
            
            # Truncate text if too long
            display_text = text
            if len(text) > max_char_in_label:
                display_text = text[:max_char_in_label-3] + "..."
                
            # Add text label
            ax2.text(start + (end-start)/2, 0, 
                     f"{speaker_label}: {display_text}", 
                     ha='center', va='center', 
                     fontsize=8, rotation=90, 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add legend for speakers
        from matplotlib.patches import Patch
        legend_elements = []
        
        # Find all unique speakers
        speakers = set()
        for segment in processed_segments:
            speakers.add(segment.get("speaker", 0))
        
        for speaker in sorted(speakers):
            color = speaker_colors[speaker % len(speaker_colors)]
            label = "Counselor" if speaker == 0 else f"Client {speaker}"
            legend_elements.append(Patch(facecolor=color, label=label))
            
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Plot simple_diarizer results in the third subplot
        ax3.set_title("Simple Diarizer Results")
        ax3.set_xlabel("Time (seconds)")
        ax3.set_yticks([])  # Hide Y-axis
        ax3.set_xlim(view_start, view_end)
        
        # If we have simple_diarizer segments, plot them
        if simple_diarizer_segments:
            # Define different colors for simple_diarizer
            diarizer_colors = ['#9C27B0', '#FF9800', '#03A9F4', '#4CAF50', '#F44336']
            
            for segment in simple_diarizer_segments:
                speaker = segment['label']  # Using 'label' instead of 'speaker'
                color = diarizer_colors[speaker % len(diarizer_colors)]
                start = segment['start']
                end = segment['end']
                
                # Draw segment
                ax3.barh(0, end-start, left=start, height=0.5, color=color, alpha=0.6)
                
                # Find corresponding text from our segments
                text = "No text"
                for our_seg in processed_segments:
                    # If there's significant overlap, use that text
                    overlap_start = max(start, our_seg["start"])
                    overlap_end = min(end, our_seg["end"])
                    if overlap_end > overlap_start and (overlap_end - overlap_start) > 0.5:
                        text = our_seg["text"]
                        if len(text) > max_char_in_label:
                            text = text[:max_char_in_label-3] + "..."
                        break
                
                # Add speaker label and text
                ax3.text(start + (end-start)/2, 0, 
                         f"Speaker {speaker}: {text}", 
                         ha='center', va='center', 
                         fontsize=8, rotation=90, 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add legend for simple_diarizer speakers
            diarizer_legend = []
            for i in range(num_speakers):
                color = diarizer_colors[i % len(diarizer_colors)]
                diarizer_legend.append(Patch(facecolor=color, label=f"Speaker {i}"))
            
            ax3.legend(handles=diarizer_legend, loc='upper right')
        else:
            ax3.text(view_end/2, 0, "Simple Diarizer failed to process this file", 
                    ha='center', va='center', fontsize=12)
        
        # Plot pyannote diarization results in the fourth subplot
        ax4.set_title("Pyannote Speaker Diarization Results")
        ax4.set_xlabel("Time (seconds)")
        ax4.set_yticks([])  # Hide Y-axis
        ax4.set_xlim(view_start, view_end)
        
        # If we have pyannote segments, plot them
        if pyannote_segments:
            # Define different colors for pyannote
            pyannote_colors = ['#8BC34A', '#E91E63', '#00BCD4', '#FF5722', '#9E9E9E']
            
            for segment in pyannote_segments:
                speaker = segment['label']  # Using 'label' instead of 'speaker'
                color = pyannote_colors[speaker % len(pyannote_colors)]
                start = segment['start']
                end = segment['end']
                
                # Draw segment
                ax4.barh(0, end-start, left=start, height=0.5, color=color, alpha=0.6)
                
                # Find corresponding text from our segments
                text = "No text"
                for our_seg in processed_segments:
                    # If there's significant overlap, use that text
                    overlap_start = max(start, our_seg["start"])
                    overlap_end = min(end, our_seg["end"])
                    if overlap_end > overlap_start and (overlap_end - overlap_start) > 0.5:
                        text = our_seg["text"]
                        if len(text) > max_char_in_label:
                            text = text[:max_char_in_label-3] + "..."
                        break
                
                # Add speaker label and text
                ax4.text(start + (end-start)/2, 0, 
                         f"Speaker {speaker}: {text}", 
                         ha='center', va='center', 
                         fontsize=8, rotation=90, 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add legend for pyannote speakers
            pyannote_legend = []
            # Find all unique speakers
            unique_speakers = set(seg['label'] for seg in pyannote_segments)
            for speaker in sorted(unique_speakers):
                color = pyannote_colors[speaker % len(pyannote_colors)]
                pyannote_legend.append(Patch(facecolor=color, label=f"Speaker {speaker}"))
            
            ax4.legend(handles=pyannote_legend, loc='upper right')
        else:
            ax4.text(view_end/2, 0, "Pyannote diarization failed or not available", 
                    ha='center', va='center', fontsize=12)
        
        # Plot NeMo diarization results in the fifth subplot
        ax5.set_title("NeMo Neural Diarizer Results")
        ax5.set_xlabel("Time (seconds)")
        ax5.set_yticks([])  # Hide Y-axis
        ax5.set_xlim(view_start, view_end)

        # If we have NeMo segments, plot them
        if nemo_segments:
            # Define different colors for NeMo
            nemo_colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#33FFF5']
            
            for segment in nemo_segments:
                speaker = segment['label']  # Using 'label' for speaker ID
                color = nemo_colors[speaker % len(nemo_colors)]
                start = segment['start']
                end = segment['end']
                
                # Draw segment
                ax5.barh(0, end-start, left=start, height=0.5, color=color, alpha=0.6)
                
                # Find corresponding text from our segments
                text = "No text"
                for our_seg in processed_segments:
                    # If there's significant overlap, use that text
                    overlap_start = max(start, our_seg["start"])
                    overlap_end = min(end, our_seg["end"])
                    if overlap_end > overlap_start and (overlap_end - overlap_start) > 0.5:
                        text = our_seg["text"]
                        if len(text) > max_char_in_label:
                            text = text[:max_char_in_label-3] + "..."
                        break
                
                # Add speaker label and text
                ax5.text(start + (end-start)/2, 0, 
                         f"Speaker {speaker}: {text}", 
                         ha='center', va='center', 
                         fontsize=8, rotation=90, 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add legend for NeMo speakers
            nemo_legend = []
            # Find all unique speakers
            unique_speakers = set(seg['label'] for seg in nemo_segments)
            for speaker in sorted(unique_speakers):
                color = nemo_colors[speaker % len(nemo_colors)]
                nemo_legend.append(Patch(facecolor=color, label=f"Speaker {speaker}"))
            
            ax5.legend(handles=nemo_legend, loc='upper right')
        else:
            ax5.text(view_end/2, 0, "NeMo diarization failed or not available", 
                    ha='center', va='center', fontsize=12)
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(bottom=0.1, hspace=0.5)
        
        # Save the visualization
        output_path = f"temp/analysis_visualization_{os.path.basename(file_path)}.png"
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
        
        # Display the plot
        plt.show()
        
        # Create a JSON result object
        final_result = {
            "id": job.id,
            "filename": job.filename,
            "duration": duration,
            "language": result.get("language", "unknown"),
            "segments": processed_segments,
            "text": job.step_results["transcription"]["text"]
        }
        
        # Save the result to a JSON file
        result_path = f"temp/analysis_result_{os.path.basename(file_path)}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        print(f"Analysis result saved to: {result_path}")
        
        # Update job status to completed
        job.update_status(AudioStatus.COMPLETING)
        print("\nAnalysis complete.")
        
        return final_result
        
    except Exception as e:
        # Handle any errors
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        job.update_status(AudioStatus.FAILED)
        job.error = str(e)
        return None

def main():
    """Main function to run the audio analysis."""
    # Use command line argument if provided, otherwise use default path
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default file path
        audio_file = '/Users/beaver.baek/Documents/audio_samples/audio-test_short_34sec.mp3'
        # audio_file = '/Users/beaver.baek/Documents/audio_samples/김수현 1_250203_35min.wav'
        # audio_file = '/Users/beaver.baek/Documents/audio_samples/나00FT#1_250401.MP3'
        print(f"No file provided, using default: {audio_file}")
    
    # Analyze the audio file
    result = analyze_audio(audio_file)
    
    if result:
        print(f"Successfully analyzed {audio_file}")

if __name__ == "__main__":
    main() 