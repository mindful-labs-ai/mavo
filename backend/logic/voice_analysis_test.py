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
import traceback
import asyncio
from backend import config
from backend.deprec.voice_analysis import do_vad_split
from backend.logic.models import AnalysisWork, AudioStatus, analysis_jobs
from backend.logic.voice_analysis import process_analysis_job_gpt_diar_mlpyan
import shutil

from backend.logic.voice_analysis.process import process_analysis_job_gpt_diar_mlpyan2

def analyze_audio(audio_file):
    """Analyze the audio file and return the results."""
    # Create an analysis job
    job_id = str(uuid.uuid4())
    print(f"Creating analysis job with ID: {job_id}")
    
    # Create a directory for this analysis job
    job_dir = Path(config.UPLOADS_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the audio file to the job directory
    new_audio_path = job_dir / os.path.basename(audio_file)
    shutil.copy2(audio_file, new_audio_path)
    
    analysis_job = AnalysisWork(
        id=job_id,
        filename=os.path.basename(audio_file),
        total_chunks=1,
        status=AudioStatus.PENDING
    )
    
    # Set the file path to the new location
    analysis_job.file_path = str(new_audio_path)
    
    # Set the audio_uuid to match the job_id
    analysis_job.audio_uuid = job_id
    
    # Set options for GPT transcription and MLP-YAN diarization
    analysis_job.options = {
        "diarization_method": "stt_apigpt_diar_mlpyan2",
        "is_limit_time": False,
        "limit_time_sec": 600,
        "is_merge_segments": True
    }
    
    # Add the job to the global analysis_jobs dictionary
    analysis_jobs[job_id] = analysis_job

    # Run VAD split synchronously since we're in a non-async function
    asyncio.run(do_vad_split(analysis_job))
    
    # Process the analysis job
    asyncio.run(process_analysis_job_gpt_diar_mlpyan2(analysis_job, is_display=False))

    print(f"Analysis job {job_id} completed")
    
    return True

def main():
    audio_file = '/Users/beaver.baek/Documents/audio_samples/김수현 1_250203_35min.wav'
    # audio_file = '/Users/beaver.baek/Documents/audio_samples/audio-test_short_34sec_converted.wav'
    # audio_file = '/Users/beaver.baek/Documents/audio_samples/audio-test_short_34sec.mp3'
    # audio_file = '/Users/beaver.baek/Documents/audio_samples/커플#7_250203_1h20min 3자.MP3'

    if config.is_linux:
        print(f"changing path for linux")
        audio_file = audio_file.replace('/Users/beaver.baek/Documents/audio_samples', '/home/gq/workspace/simri/audio_samples')

    # get only few minute of audio
    tmp_audio_file = "log/tmp_audio.wav"
    mins = 4
    secs = mins * 60
    audio = AudioSegment.from_wav(audio_file)
    audio = audio[:secs * 1000]
    audio.export(tmp_audio_file, format="wav")
    
    # tmp_audio_file = audio_file
    result = analyze_audio(tmp_audio_file)
    if result:
        print(f"Successfully analyzed {tmp_audio_file}")

if __name__ == "__main__":
    main() 