import asyncio
import gc
import json
import os
import tempfile
from pydub import AudioSegment
from typing import Dict, List, Any
from fastapi import BackgroundTasks
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from pyannote.audio import Pipeline
import torchaudio
import matplotlib.pyplot as plt
import librosa
import librosa.display
from backend.logic.stt_utils import get_improved_lines_with_ts
from backend.logic.models import (
    AnalysisWork,
    AudioStatus,
    analysis_jobs,
    TranscriptionResult,
    Segment,
    Speaker,
    save_analysis_work_to_json,
)
import backend.config as config
from backend.logic.voice_analysis.ai_utils import (
    assign_speaker_roles,
    assign_speaker_to_lines_with_gpt,
    get_openai_client,
    get_seg_ts_with_diar_with_speaker_infer_wo_skip,
    get_seg_ts_with_speaker_infer_wo_skip,
    improve_transcription_lines,
    improve_transcription_lines_parallel,
)
from backend.logic.voice_analysis.audio_utils import (
    process_audio_with_assembly_ai,
    process_audio_with_openai_api,
    split_audio_with_vad,
)
from backend.logic.voice_analysis.common_utils import (
    create_consecutive_segments,
    get_cleared_diarization_segments,
    save_analysis_work_json,
    save_transcription_result,
    save_transcription_result_json,
)
from backend.util.logger import get_logger
import sys
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
import matplotlib as mpl

import torch

# This must be imported before pyannote.audio, torchaudio, etc.
import backend.util.setup_torch  # noqa: F401

if config.is_linux:
    import whisperx

logger = get_logger(__name__)


if config.DEVICE == "cuda":
    logger.info("Setting allow_tf32 for CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


async def diarize_audio_parallel(analysis_job, split_segments):
    """
    Perform speaker diarization on multiple audio segments in parallel.
    Unlike diarize_audio which processes a single file, this function
    processes multiple split segments and combines the results.
    """
    try:
        audio_uuid = analysis_job.audio_uuid
        logger.info(
            f"Starting parallel diarization for {len(split_segments)} segments of job {audio_uuid}"
        )

        # Ensure we have the HF_TOKEN for pyannote
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not found in environment variables")

        # Process segments in parallel using a thread pool
        # all_diarization_segments = []
        all_diarization_segments = [None] * len(split_segments)
        # total_duration = 0
        # time_offset = 0

        # Initialize the pipeline once for all segments
        logger.info(f"Initializing diarization pipeline on device: {config.DEVICE}")

        is_use_thread = True
        count_fin_segments = 0

        # Process each segment
        # for segment_idx, segment in enumerate(split_segments):
        def fn_diarize_segment(segment_idx, segment):
            nonlocal count_fin_segments

            ## @ref https://huggingface.co/pyannote/speaker-diarization-3.1
            # pipeline = Pipeline.from_pretrained("JSWOOK/pyannote_3_fine_tuning", use_auth_token=HF_TOKEN)

            ## @ref https://huggingface.co/JSWOOK/pyannote_3_fine_tuning (finetuned, requires onnxruntime or onnxruntime-gpu/onnxruntime-silicon)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
            )

            device = config.DEVICE
            pipeline.to(torch.device(device))

            # Ensure max_speakers is an integer
            max_speakers = (
                int(config.MAX_SPEAKERS)
                if isinstance(config.MAX_SPEAKERS, str)
                else config.MAX_SPEAKERS
            )

            segment_path = segment["path"] if isinstance(segment, dict) else segment
            logger.info(
                f"Processing segment {segment_idx+1}/{len(split_segments)}: {segment_path}"
            )
            print("segment", segment)

            tmp_wav_path = str(segment_path)

            time_offset = segment["start"]  # second
            print("time_offset for segment", segment_idx, time_offset)

            # try:
            # Load audio and run diarization
            waveform, sample_rate = torchaudio.load(tmp_wav_path)
            audio_file = {"waveform": waveform, "sample_rate": sample_rate}

            # Run diarization on this segment
            diarization = pipeline(
                audio_file, min_speakers=2, max_speakers=max_speakers
            )

            # Convert diarization result to list of segments and apply time offset
            segment_diarization = []
            for diar_segment, _, speaker in diarization.itertracks(yield_label=True):
                speaker_num = speaker.split("_")[1]
                speaker_num_int = int(speaker_num) if speaker_num.isdigit() else -1

                segment_diarization.append(
                    {
                        "start": diar_segment.start + time_offset,
                        "end": diar_segment.end + time_offset,
                        "duration": diar_segment.end - diar_segment.start,
                        "speaker": speaker_num_int,
                    }
                )

            # Add to overall results
            all_diarization_segments[segment_idx] = segment_diarization

            # # Update time offset for next segment
            # time_offset += segment_duration
            # total_duration += segment_duration

            # Update progress
            count_fin_segments += 1
            progress = (count_fin_segments) / len(split_segments) * 100
            analysis_job.update_step(
                {
                    "step_name": "diarizing",
                    "status": "in_progress",
                    "total_segments": len(split_segments),
                    "processed_segments": count_fin_segments,
                    "percent_complete": progress,
                }
            )

            # finally:
            #     # Clean up temporary file
            #     if os.path.exists(tmp_wav_path):
            #         os.unlink(tmp_wav_path)

        futures = []
        if is_use_thread:
            with ThreadPoolExecutor(max_workers=5) as executor:
                for segment_idx, segment in enumerate(split_segments):
                    futures.append(
                        executor.submit(fn_diarize_segment, segment_idx, segment)
                    )

                for future in as_completed(futures):
                    future.result()
        else:
            for segment_idx, segment in enumerate(split_segments):
                fn_diarize_segment(segment_idx, segment)

        # Sort all segments by start time
        all_diarization_segments.sort(key=lambda x: x[0]["start"])

        ## differentiate speakers for each segs
        num_unique_speakers_per_seg = []
        for idx_seg, seg in enumerate(all_diarization_segments):
            num_unique_speakers = len(set(seg["speaker"] for seg in seg))
            num_unique_speakers_per_seg.append(num_unique_speakers)
            print(f"at {idx_seg}th segment, num_unique_speakers: {num_unique_speakers}")
        offset_speaker_num = 0
        for i, seg in enumerate(all_diarization_segments):
            for seg_seg in seg:
                seg_seg["speaker"] = seg_seg["speaker"] + offset_speaker_num
            offset_speaker_num += num_unique_speakers_per_seg[i]

        # Mark the diarization step as completed
        analysis_job.complete_step("diarizing")

        # Save diarization results to JSON
        uploads_dir = config.UPLOADS_DIR / str(audio_uuid)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        diarization_path = uploads_dir / f"id[{audio_uuid}]_ml_diarized.json"

        # Flatten the segments list and get unique speakers
        flattened_segments = [
            seg for segment_list in all_diarization_segments for seg in segment_list
        ]
        unique_speakers = set(seg["speaker"] for seg in flattened_segments)

        diarization_result = {
            # "total_duration": total_duration,
            "total_duration": split_segments[-1]["end"],
            "num_speakers": len(unique_speakers),
            "segments": flattened_segments,
        }

        with open(diarization_path, "w", encoding="utf-8") as f:
            json.dump(diarization_result, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved parallel diarization result to {diarization_path}")
        return flattened_segments

    except Exception as e:
        err_msg = f"ERROR in diarize_audio_parallel: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise


async def diarize_audio_whisperx(analysis_job):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_language = "ko"  # 기본값 설정
    MAX_SPEAKERS_IN_SPLIT = 5
    custom_model = str(config.DATASET_DIR / "custom_diarize" / "config.yaml")
    HF_TOKEN = os.getenv("HF_TOKEN")

    try:
        print("custom_model: ", custom_model)
        print("Loading diarization model... token: ", HF_TOKEN)

        diarize_model = whisperx.DiarizationPipeline(
            model_name=custom_model, use_auth_token=HF_TOKEN, device=device
        )

        print("Diarizing segments...")

        audio = whisperx.load_audio(analysis_job.file_path)
        max_speaker_multiplier = 4
        diarize_segments = diarize_model(
            audio,
            num_speakers=None,
            max_speakers=MAX_SPEAKERS_IN_SPLIT * max_speaker_multiplier,
        )

        print("Diarizing segments done")

        ## memory cleanup
        del diarize_model
        torch.cuda.empty_cache()

        return diarize_segments
    except Exception as e:
        err_msg = f"ERROR in diarize_audio_whisperx: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        raise


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
            # 분석 작업을 찾지 못한 경우, 새로운 작업 생성
            logger.info(f"No analysis job found for {audio_uuid}, creating a new one")
            job_id = audio_uuid  # 동일한 ID 사용
            analysis_job = AnalysisWork(
                id=job_id,
                filename=os.path.basename(file_path),
                total_chunks=1,
                status=AudioStatus.PENDING,
            )
            analysis_job.file_path = file_path
            analysis_job.audio_uuid = audio_uuid
            analysis_job.options = {
                "diarization_method": "stt_apigpt_diar_mlpyan2",
                "is_limit_time": False,
                "limit_time_sec": 600,
                "is_merge_segments": True,
            }

            # 두 ID로 작업 등록
            analysis_jobs[job_id] = analysis_job  # id로 등록
            if (
                job_id != audio_uuid
            ):  # 다른 경우에만 중복 등록 (이 경우는 같지만 안전성을 위해)
                analysis_jobs[audio_uuid] = analysis_job  # audio_uuid로도 등록

            logger.info(f"Created new analysis job for {audio_uuid}")

        # Convert audio to required format
        audio = AudioSegment.from_file(file_path)

        # Apply time limit if enabled
        if analysis_job.options["is_limit_time"]:
            limit_ms = analysis_job.options["limit_time_sec"] * 1000
            if len(audio) > limit_ms:
                logger.info(
                    f"Limiting audio to {analysis_job.options['limit_time_sec']} seconds"
                )
                audio = audio[:limit_ms]

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
            audio.export(tmp_wav_path, format="wav")

        try:
            if analysis_job.options["diarization_method"] == "stt_apigpt_diar_mlpyan":
                # Use pyannote.audio for diarization
                HF_TOKEN = os.getenv("HF_TOKEN")
                if not HF_TOKEN:
                    raise ValueError("HF_TOKEN not found in environment variables")

                ## @ref https://huggingface.co/pyannote/speaker-diarization-3.1
                pipeline = Pipeline.from_pretrained(
                    "JSWOOK/pyannote_3_fine_tuning", use_auth_token=HF_TOKEN
                )

                ## @ref https://huggingface.co/JSWOOK/pyannote_3_fine_tuning (finetuned, requires onnxruntime or onnxruntime-gpu/onnxruntime-silicon)
                # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)

                # Use MPS if available, otherwise CPU
                logger.info(f"Diarizing using device: {config.DEVICE}")
                device = config.DEVICE
                pipeline.to(torch.device(device))

                # Load audio and run diarization
                waveform, sample_rate = torchaudio.load(tmp_wav_path)
                audio_file = {"waveform": waveform, "sample_rate": sample_rate}

                # Ensure max_speakers is an integer
                max_speakers = (
                    int(config.MAX_SPEAKERS)
                    if isinstance(config.MAX_SPEAKERS, str)
                    else config.MAX_SPEAKERS
                )
                diarization = pipeline(
                    audio_file, min_speakers=2, max_speakers=max_speakers
                )

                # Get total duration from waveform
                total_duration = waveform.size(1) / sample_rate

                # Convert diarization result to list of segments
                diarization_segments = []
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_num = speaker.split("_")[1]
                    speaker_num_int = int(speaker_num) if speaker_num.isdigit() else -1

                    diarization_segments.append(
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "duration": segment.end - segment.start,
                            "speaker": speaker_num_int,
                        }
                    )

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
                    "num_speakers": len(
                        set(seg["speaker"] for seg in diarization_segments)
                    ),
                    "segments": diarization_segments,
                }

                with open(diarization_path, "w", encoding="utf-8") as f:
                    json.dump(diarization_result, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved diarization result to {diarization_path}")
                return diarization_segments

            else:
                # Use GPT-based diarization
                # This will be handled in the transcription improvement step
                total_duration = len(audio) / 1000
                diarization_segments = [
                    {
                        "start": 0,
                        "end": total_duration,
                        "duration": total_duration,
                        "speaker": -1,
                    }
                ]

            # Update job step with diarization visualization data
            if analysis_job:
                analysis_job.update_step(
                    {
                        "status": AudioStatus.DIARIZING,
                        "diarization_segments": diarization_segments,
                        "total_duration": total_duration,
                        "num_speakers": len(
                            set(seg["speaker"] for seg in diarization_segments)
                        ),
                    }
                )

            logger.info(f"Diarization completed for {file_path}")

            return diarization_segments

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)

    except Exception as e:
        err_msg = (
            f"ERROR in diarize_audio: {e}\n with traceback:\n{traceback.format_exc()}"
        )
        logger.error(err_msg)
        raise


## adhoc for async job
"""
To wrap asynchronous functions in a thread with a new event loop, we can define helper functions that use asyncio.run() to call the async functions synchronously.

example:
def run_async_function(coro):
    return asyncio.run(coro)
"""


def run_diarize_audio(file_path):
    return asyncio.run(diarize_audio(file_path))


def run_diarize_audio_parallel(analysis_job, split_segments):
    return asyncio.run(diarize_audio_parallel(analysis_job, split_segments))


def run_process_transcription(analysis_job, split_segments):
    return asyncio.run(process_transcription(analysis_job, split_segments))


def run_diarize_audio_whisperx(analysis_job):
    return asyncio.run(diarize_audio_whisperx(analysis_job))


def run_whisperx_transcription_and_diarization(analysis_job, split_segments):
    return asyncio.run(
        process_whisperx_transcription_and_diarization(analysis_job, split_segments)
    )


async def do_vad_split(analysis_job: AnalysisWork) -> None:
    """Split audio file with VAD."""
    # Update status to splitting
    analysis_job.update_status(AudioStatus.SPLITTING)

    # Split audio using VAD
    logger.info("Splitting audio using VAD...")
    vad_mins = 2
    split_segments = split_audio_with_vad(
        analysis_job.file_path, analysis_job, target_duration=60 * vad_mins
    )
    logger.info(f"Created {len(split_segments)} splits")
    analysis_job.split_segments = (
        split_segments  ## cautious - this may lead dump error.
    )

    # Store split paths in the job
    analysis_job.split_paths = (
        split_segments
        if isinstance(split_segments[0], str)
        else [seg["path"] for seg in split_segments]
    )

    # Save state after status update
    save_analysis_work_json(analysis_job)

    # At the end of the function, mark the splitting step as completed
    analysis_job.complete_step("splitting")

    # Log completion
    logger.info(f"VAD splitting completed for {analysis_job.id}")


async def process_analysis_job_gpt_diar_gpt(analysis_job: AnalysisWork) -> None:
    logger.info(
        f"Processing analysis job for {analysis_job.audio_uuid} with method: {analysis_job.options['diarization_method']}"
    )
    await do_vad_split(analysis_job)

    ## add tasks
    tasks = []
    analysis_job.update_status(AudioStatus.TRANSCRIBING)
    transcription_task = asyncio.create_task(
        asyncio.to_thread(
            run_process_transcription, analysis_job, analysis_job.split_segments
        ),
        name="transcription",
    )
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
        speaker_segments.append(
            {"start": seg["start"], "end": seg["end"], "speaker": speaker_id}
        )

    # Create speakers list - only for actual numeric speakers
    speakers = [
        {"id": speaker_id, "role": "counselor" if speaker_id == 0 else "client"}
        for speaker_id in sorted(speaker_ids)
        if speaker_id != "undecided"
    ]

    # Store diarization result
    analysis_job.store_step_result(
        "diarization", {"speakers": speakers, "segments": speaker_segments}
    )

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
            speaker_diarized="",
        )
        for i, seg in enumerate(transcription_segments)
    ]

    # Sort segments by start time
    segments.sort(key=lambda x: x.start)

    ## save the result
    analysis_job.result = TranscriptionResult(
        text=result_transcription["text"], segments=segments, speakers=[]
    )

    # Store improving result (complete result with speaker identification)
    improving_segments = []
    for i, seg in enumerate(segments):
        improving_segments.append(
            {
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
            }
        )

    # Store the improving result
    analysis_job.store_step_result("improving", {"segments": improving_segments})

    # Mark the job as completed
    analysis_job.update_status(AudioStatus.COMPLETING)

    print("analysis_job", analysis_job)
    print("analysis_job.audio_uuid", analysis_job.audio_uuid)
    print("analysis_job.result", analysis_job.result)
    print("analysis_job.result.segments", analysis_job.result.segments)
    print("analysis_job.result.speakers", analysis_job.result.speakers)

    save_transcription_result_json(
        analysis_job.audio_uuid, analysis_job.result, suffix="transcript"
    )
    save_analysis_work_json(analysis_job)

    pass


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
        transcript_dir = config.TEMP_DIR / "transcripts" / audio_uuid
        transcript_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리 자동 생성
        diar_debug_path = transcript_dir / f"id[{audio_uuid}]_diarization_debug.json"

        logger.info(
            f"Processing analysis job for {audio_uuid} with method: {analysis_job.options['diarization_method']}"
        )

        # Step 1: VAD splitting
        await do_vad_split(analysis_job)

        # Step 2 & 3: Run transcription and diarization in parallel
        tasks = []

        # Transcription task
        analysis_job.update_status(AudioStatus.TRANSCRIBING)
        transcription_task = asyncio.create_task(
            asyncio.to_thread(
                run_process_transcription, analysis_job, analysis_job.split_segments
            ),
            name="transcription",
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

        # Save transcription segments before speaker assignment for debugging
        trans_debug_path = (
            transcript_dir / f"id[{audio_uuid}]_transcription_words_debug.json"
        )
        with open(trans_debug_path, "w", encoding="utf-8") as f:
            json.dump(result_transcription["segments"], f, ensure_ascii=False, indent=2)

        ## update script with text improvement
        analysis_job.update_status(AudioStatus.IMPROVING)
        trans_lines = improve_transcription_lines(result_transcription["text"])
        print("improved trans_lines", trans_lines)
        # trans_lines_with_speaker = improve_transcription_lines_with_speaker(result_transcription["text"])

        trans_lines_with_speaker = assign_speaker_to_lines_with_gpt(trans_lines)
        print("improved trans_lines_with_speaker", trans_lines_with_speaker)

        print(
            f"len trans_lines_with_speaker: {len(trans_lines_with_speaker)}, len trans_lines: {len(trans_lines)}"
        )
        # trans_lines = [line['text'] for line in trans_lines_with_speaker]
        trans_speakers = [line["speaker"] for line in trans_lines_with_speaker]

        ## num of different speakers
        num_speakers = len(set(trans_speakers))
        print(f"num of different speakers: {num_speakers}")

        ## (DBG) save 'improved_lines' to json file
        save_path = config.TEMP_DIR / f"tmp_improved_lines.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(trans_lines, f, ensure_ascii=False, indent=2)
        save_path = config.TEMP_DIR / f"tmp_transcription_words.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_transcription["segments"], f, ensure_ascii=False, indent=2)
        # save_path = config.TEMP_DIR / f"tmp_diarization_segments.json"
        # with open(save_path, "w", encoding="utf-8") as f:
        #     json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

        print("\ntrans_lines", trans_lines)
        print("\ntranscription_words", result_transcription["segments"])
        # print("\ndiarization_segments", diarization_segments)

        trans_segs_with_ts = get_improved_lines_with_ts(
            trans_lines, result_transcription["segments"]
        )
        # print("\ntrans_segs_with_ts", trans_segs_with_ts)
        print(
            f"len speaker: {len(trans_speakers)}, len trans_lines: {len(trans_lines)}, len transcription_words: {len(result_transcription['segments'])} len trans_segs_with_ts: {len(trans_segs_with_ts)}"
        )
        for idx_seg, seg in enumerate(trans_segs_with_ts):
            seg["speaker"] = trans_speakers[idx_seg]

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
            if prev_seg["speaker"] == seg["speaker"]:
                prev_seg["end"] = seg["end"]
                prev_seg["text"] += " " + seg["text"].strip()
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
                speaker=(
                    "undecided"
                    if speaker_id == "undecided"
                    else speaker_id_map.get(speaker_id, 0)
                ),
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
        analysis_job.result = TranscriptionResult(segments=segments, speakers=speakers)

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


async def process_analysis_job_gpt_diar_gpt3(
    analysis_job: AnalysisWork, is_display: bool = False
) -> None:
    """Process the analysis job using GPT for transcription and MLP-YAN for diarization."""
    try:
        # Get the audio UUID from the analysis job
        audio_uuid = analysis_job.id
        if not audio_uuid:
            raise ValueError("Missing audio_uuid in analysis_job")

        # Set audio_uuid on the analysis_job object for consistency
        analysis_job.audio_uuid = audio_uuid

        # Now use audio_uuid in path construction
        transcript_dir = config.TEMP_DIR / "transcripts" / audio_uuid
        transcript_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리 자동 생성
        diar_debug_path = transcript_dir / f"id[{audio_uuid}]_diarization_debug.json"

        logger.info(
            f"Processing analysis job for {audio_uuid} with method: {analysis_job.options['diarization_method']}"
        )

        # Step 1: VAD splitting
        await do_vad_split(analysis_job)

        # Step 2 & 3: Run transcription and diarization in parallel
        tasks = []

        # Transcription task
        analysis_job.update_status(AudioStatus.TRANSCRIBING)
        transcription_task = asyncio.create_task(
            asyncio.to_thread(
                run_process_transcription, analysis_job, analysis_job.split_segments
            ),
            name="transcription",
        )
        tasks.append(transcription_task)

        # # Diarization task
        # analysis_job.update_status(AudioStatus.DIARIZING)
        # diarization_task = asyncio.create_task(
        #     # asyncio.to_thread(run_diarize_audio, analysis_job.file_path),
        #     asyncio.to_thread(run_diarize_audio_parallel, analysis_job, analysis_job.split_segments),
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

        # # Save frontend-friendly diarization data
        # frontend_diarization = {
        #     "step_name": "diarizing",
        #     # "status": AudioStatus.DIARIZING,
        #     "diarization_segments": diarization_segments,
        #     "total_duration": max([seg["end"] for seg in diarization_segments]) if diarization_segments else 0,
        #     "num_speakers": len(set(seg["speaker"] for seg in diarization_segments if "speaker" in seg))
        # }

        # # Update job step with diarization visualization data for frontend
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
        transcription_words_splits = result_transcription["text_splits"]

        # Save transcription segments before speaker assignment for debugging
        trans_debug_path = (
            transcript_dir / f"id[{audio_uuid}]_transcription_words_debug.json"
        )
        with open(trans_debug_path, "w", encoding="utf-8") as f:
            json.dump(transcription_words, f, ensure_ascii=False, indent=2)

        is_do_text_improv = False
        if is_do_text_improv:
            ## update script with text improvement
            analysis_job.update_status(AudioStatus.IMPROVING)
            logger.info(f"improving transcription_lines with ai")
            logger.info("dbg11", result_transcription["text"])
            logger.info("dbg12", transcription_words_splits)
            # trans_lines = improve_transcription_lines(result_transcription["text"])
            trans_lines = improve_transcription_lines_parallel(
                transcription_words_splits
            )
            # logger.info("dbg22", trans_lines)

            ## (DBG) save 'improved_lines' to json file
            if config.is_save_temp_files:
                save_path = config.TEMP_DIR / f"tmp010_improved_lines.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(trans_lines, f, ensure_ascii=False, indent=2)
                save_path = config.TEMP_DIR / f"tmp020_transcription_words.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(transcription_words, f, ensure_ascii=False, indent=2)
                # save_path = config.TEMP_DIR / f"tmp030_diarization_segments.json"
                # with open(save_path, "w", encoding="utf-8") as f:
                #     json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

            # print("\ntrans_lines", trans_lines)
            # print("\ntranscription_words", transcription_words)
            # print("\ndiarization_segments", diarization_segments)

            logger.info(f"applying improved transcription_lines to transcription_words")
            trans_segs_with_ts = get_improved_lines_with_ts(
                trans_lines, transcription_words
            )
            # print("\ntrans_segs_with_ts", trans_segs_with_ts)

            if config.is_save_temp_files:
                ## save 'trans_segs_with_ts'
                save_path = config.TEMP_DIR / f"tmp040_trans_segs_with_ts.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(trans_segs_with_ts, f, ensure_ascii=False, indent=2)
        else:
            logger.info(f"skip text improvement")
            trans_segs_with_ts = (
                transcription_words  # this is already improved, segments.
            )

        logger.info(
            f"infer speaker from transcription_segments and diarization_segments"
        )

        trans_segs_with_ts_with_diar = get_seg_ts_with_speaker_infer_wo_skip(
            trans_segs_with_ts
        )

        ## merge segments with same speaker continues speaking
        trans_segs_with_ts_conti = []
        prev_seg = None
        for idx_seg, seg in enumerate(trans_segs_with_ts_with_diar):
            if prev_seg is None:
                prev_seg = seg
                continue
            if prev_seg["speaker"] == seg["speaker"]:
                prev_seg["end"] = seg["end"]
                prev_seg["text"] += " " + seg["text"].strip()
            else:
                trans_segs_with_ts_conti.append(prev_seg)
                prev_seg = seg
        # AFTER the loop, do this:
        if prev_seg is not None:
            trans_segs_with_ts_conti.append(prev_seg)

        if config.is_save_temp_files:
            ## save 'trans_segs_with_ts_conti'
            save_path = config.TEMP_DIR / f"tmp060_trans_segs_with_ts_conti.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(trans_segs_with_ts_conti, f, ensure_ascii=False, indent=2)

        # print("transcription_segments", trans_segs_with_ts)
        # print("transcription_segments_with_diar", trans_segs_with_ts_with_diar)

        # Create segments from transcription result with assigned speakers
        segments = []
        speakers = []
        current_speaker_id = 0
        speaker_id_map = {}

        # Process segments to create Segment objects with proper speaker IDs
        for seg in trans_segs_with_ts_conti:
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
                speaker=(
                    "undecided"
                    if speaker_id == "undecided"
                    else speaker_id_map.get(speaker_id, 0)
                ),
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
        analysis_job.result = TranscriptionResult(segments=segments, speakers=speakers)

        if config.is_save_temp_files:
            save_path = config.TEMP_DIR / f"tmp080_analysis_job_result.json"
            save_analysis_work_to_json(analysis_job, save_path)

        if is_display:
            # with matplotlib, display to timeline.
            # 첫 번째 행 :
            #     제목: "Audio Waveform with Transcription"
            #     내용: librosa를 사용한 오디오 파형 시각화
            #     표시 항목: 시간에 따른 오디오 진폭 그래프
            # 두 번째 행 :
            #     제목: "Pyannote Speaker Diarization Results"
            #     내용: pyannote 라이브러리 화자 분할 결과
            #     표시 항목: 시간별 화자 구분 색상 바, 화자 번호, 연결된 텍스트
            # 특징
            # 모든 행에서 시간 동기화된 뷰를 제공함
            # 각 행마다 다른 화자 분할 기술의 결과 비교 가능
            # 화자별 다른 색상 코드 사용 (각 방식마다 다른 색상표 적용)
            # 텍스트가 너무 긴 경우 max_char_in_label(15자) 제한으로 표시
            # 한글 폰트 지원을 위해 여러 폰트 후보 시도 (NanumGothic, Malgun Gothic 등)
            # 결과물은 PNG 이미지로 저장되고 plt.show()로 표시됨
            try:
                # Load audio for visualization
                audio, sr = librosa.load(analysis_job.file_path, sr=16000)
                duration = len(audio) / sr

                # Setup fonts for visualization
                font_candidates = [
                    "NanumGothic",
                    "Malgun Gothic",
                    "AppleGothic",
                    "Arial Unicode MS",
                    "NanumMyeongjo",
                ]
                font_found = False

                for font_name in font_candidates:
                    if any(
                        font_name.lower() in f.name.lower()
                        for f in mpl.font_manager.fontManager.ttflist
                    ):
                        mpl.rcParams["font.family"] = font_name
                        font_found = True
                        logger.info(f"Using font: {font_name}")
                        break

                if not font_found:
                    logger.warning(
                        "No suitable Korean font found. Some characters may not display correctly."
                    )

                # Alternative method for Korean fonts
                try:
                    # For macOS
                    if sys.platform == "darwin":
                        mpl.rc("font", family="AppleGothic")
                    # For Windows
                    elif sys.platform == "win32":
                        mpl.rc("font", family="Malgun Gothic")
                except:
                    pass

                # Create figure with 3 subplots and an additional subplot for the scrollbar
                fig = plt.figure(figsize=(14, 18))  # Increase height for 5 rows
                gs = fig.add_gridspec(
                    5, 1, height_ratios=[4, 4, 4, 4, 1]
                )  # 5 rows, last one for scrollbar

                # Create the main plotting axes
                ax1 = fig.add_subplot(gs[0])  # Audio waveform
                ax2 = fig.add_subplot(gs[1])  # Speaker diarization results
                ax3 = fig.add_subplot(gs[2])  # Raw diarization
                ax4 = fig.add_subplot(gs[3])  # Cleared diarization
                ax_scroll = fig.add_subplot(gs[4])  # Scrollbar axis

                # Set the default view range
                view_window = 30  # seconds
                view_start = 0
                view_end = min(view_window, duration)
                max_char_in_label = 15

                # Function to update all plots when scrolling
                def update_view(val):
                    view_start = val
                    view_end = min(val + view_window, duration)

                    # Update xlim for all plots
                    ax1.set_xlim(view_start, view_end)
                    ax2.set_xlim(view_start, view_end)
                    ax3.set_xlim(view_start, view_end)
                    ax4.set_xlim(view_start, view_end)

                    fig.canvas.draw_idle()

                # Set initial view limits for all plots
                ax1.set_xlim(view_start, view_end)
                ax2.set_xlim(view_start, view_end)
                ax3.set_xlim(view_start, view_end)
                ax4.set_xlim(view_start, view_end)

                # Create a slider
                slider = Slider(
                    ax_scroll,
                    "Time",
                    0,
                    max(0, duration - view_window),
                    valinit=0,
                    valstep=1,
                )
                slider.on_changed(update_view)

                # Plot audio waveform
                ax1.set_title("Audio Waveform with Transcription")
                librosa.display.waveshow(y=audio, sr=sr, ax=ax1)
                ax1.set_xlabel("Time (seconds)")
                ax1.set_ylabel("Amplitude")

                # Add word-level timestamps if available in the transcription
                # First, try to extract word-level timestamps from the transcription results
                word_timestamps = []

                # Check if we have any word-level timestamp information
                try:
                    logger.info("Looking for word-level timestamps for visualization")

                    # Try different sources for word-level transcription
                    for word_data in transcription_words:
                        if (
                            "text" in word_data
                            and "start" in word_data
                            and "end" in word_data
                        ):
                            word_timestamps.append(
                                {
                                    "word": word_data["text"],
                                    "start": word_data["start"],
                                    "end": word_data["end"],
                                }
                            )

                except Exception as word_error:
                    logger.warning(
                        f"Could not add word markers to waveform: {word_error}"
                    )
                    logger.warning(traceback.format_exc())
                    # Continue without word markers

                # Plot segments with different colors for speakers
                ax2.set_title("Speaker Diarization Results")
                ax2.set_xlabel("Time (seconds)")
                ax2.set_yticks([])  # Hide Y-axis

                # Define colors for speakers
                speaker_colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01"]

                unique_speakers_in_result = list(
                    set(segment.speaker for segment in segments)
                )
                print(f"len of unique_speakers: {len(unique_speakers_in_result)}")

                # Plot segments with colors based on speaker
                for segment in segments:
                    # Get speaker ID, handling "undecided" case
                    if segment.speaker == "undecided":
                        speaker = -1  # Special value for undecided
                    else:
                        speaker = segment.speaker

                    color = (
                        speaker_colors[speaker % len(speaker_colors)]
                        if speaker >= 0
                        else "#CCCCCC"
                    )
                    start = segment.start
                    end = segment.end
                    text = segment.text

                    # Draw segment
                    ax2.barh(
                        0, end - start, left=start, height=0.5, color=color, alpha=0.6
                    )

                    # Add speaker label and text
                    if speaker >= 0:
                        speaker_label = (
                            "상담사" if speaker == 0 else f"내담자 {speaker}"
                        )
                    else:
                        speaker_label = "미확인"

                    # Truncate text if too long
                    display_text = text
                    if len(text) > max_char_in_label:
                        display_text = text[: max_char_in_label - 3] + "..."

                    # Add text label
                    ax2.text(
                        start + (end - start) / 2,
                        0,
                        f"{speaker_label}: {display_text}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        rotation=90,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )

                # Add legend for speakers
                legend_elements = []

                # Find all unique speakers
                unique_speakers_in_result = set()
                for segment in segments:
                    if segment.speaker != "undecided":
                        unique_speakers_in_result.add(segment.speaker)

                # Add undecided speaker if needed
                has_undecided = any(
                    segment.speaker == "undecided" for segment in segments
                )

                for speaker in sorted(unique_speakers_in_result):
                    color = speaker_colors[speaker % len(speaker_colors)]
                    label = "상담사" if speaker == 0 else f"내담자 {speaker}"
                    legend_elements.append(Patch(facecolor=color, label=label))

                # Add undecided to legend if present
                if has_undecided:
                    legend_elements.append(Patch(facecolor="#CCCCCC", label="미확인"))

                ax2.legend(handles=legend_elements, loc="upper right")

                # Plot original diarization segments in the third subplot
                ax3.set_title("Pyannote Raw Diarization")
                ax3.set_xlabel("Time (seconds)")
                ax3.set_yticks([])  # Hide Y-axis

                # Define different colors for diarization visualization
                diar_colors = ["#8BC34A", "#E91E63", "#00BCD4", "#FF5722", "#9E9E9E"]

                # Get original diarization segments from the step results if available
                diarization_segments = []
                try:
                    # Try to get original diarization information
                    if "diarizing" in analysis_job.steps:
                        diar_step = analysis_job.steps["diarizing"]
                        if hasattr(diar_step, "get") and diar_step.get(
                            "diarization_segments"
                        ):
                            diarization_segments = diar_step["diarization_segments"]
                            logger.info(
                                f"Found {len(diarization_segments)} diarization segments to visualize"
                            )

                    # Fallback to frontend-friendly diarization data
                    if not diarization_segments and analysis_job.step_results.get(
                        "diarization"
                    ):
                        diarization_segments = analysis_job.step_results[
                            "diarization"
                        ].get("segments", [])
                        logger.info(
                            f"Using {len(diarization_segments)} segments from step_results"
                        )

                    # See if we can load from file as last resort
                    if not diarization_segments:
                        audio_uuid = analysis_job.audio_uuid
                        diar_path = (
                            config.UPLOADS_DIR
                            / str(audio_uuid)
                            / f"id[{audio_uuid}]_ml_diarized.json"
                        )
                        if os.path.exists(diar_path):
                            with open(diar_path, "r") as f:
                                diar_data = json.load(f)
                                if "diarization_segments" in diar_data:
                                    diarization_segments = diar_data[
                                        "diarization_segments"
                                    ]
                                    logger.info(
                                        f"Loaded {len(diarization_segments)} segments from file"
                                    )

                    unique_speakers_in_diar = list(
                        set(
                            segment.get("speaker", 0)
                            for segment in diarization_segments
                        )
                    )

                    # If we have diarization segments, plot them
                    if diarization_segments:
                        # Create multiple rows for visualization (up to 5)
                        row_height = 0.15
                        row_spacing = 0.2
                        max_rows = len(
                            unique_speakers_in_diar
                        )  # Use all unique speakers instead of fixed max
                        speaker_rows = {}

                        # First pass to assign rows to speakers
                        for speaker_id in unique_speakers_in_diar:
                            # Assign row position (0 at bottom, increasing upward)
                            speaker_rows[speaker_id] = len(speaker_rows)

                        # Plot segments with colors based on speaker in their assigned row
                        for segment in diarization_segments:
                            speaker = segment.get("speaker", 0)
                            # Only plot speakers that have assigned rows
                            if speaker in speaker_rows:
                                row_idx = speaker_rows[speaker]
                                row_pos = row_idx * row_spacing

                                color = diar_colors[speaker % len(diar_colors)]
                                start = segment.get("start", 0)
                                end = segment.get("end", 0)

                                # Skip invalid segments
                                if start >= end or end <= 0:
                                    continue

                                # Draw segment as horizontal bar
                                ax3.barh(
                                    row_pos,
                                    end - start,
                                    left=start,
                                    height=row_height,
                                    color=color,
                                    alpha=0.7,
                                    edgecolor="black",
                                    linewidth=0.5,
                                )

                                # Add speaker label in the middle of segment if long enough
                                if end - start > 2.0:
                                    ax3.text(
                                        start + (end - start) / 2,
                                        row_pos,
                                        f"화자 {speaker}",
                                        ha="center",
                                        va="center",
                                        fontsize=8,
                                        bbox=dict(
                                            boxstyle="round",
                                            facecolor="white",
                                            alpha=0.7,
                                        ),
                                    )

                        # Add legend for speakers
                        legend_elements = []
                        for speaker in sorted(speaker_rows.keys()):
                            color = diar_colors[speaker % len(diar_colors)]
                            legend_elements.append(
                                Patch(facecolor=color, label=f"화자 {speaker}")
                            )

                        ax3.legend(handles=legend_elements, loc="upper right")

                        # Set y-axis limits to show all rows with some padding
                        ax3.set_ylim(-0.1, len(speaker_rows) * row_spacing + 0.1)

                        # Add word markers on the third row diarization visualization
                        try:
                            # Display all word timestamps on the diarization plot
                            for word_data in word_timestamps:
                                word = word_data["word"]
                                start = word_data["start"]
                                end = word_data["end"]

                                # Get y position for the word markers
                                row_pos = 0
                                y_pos = row_pos + (row_height / 2) + 0.02

                                # Add vertical line at word START (green)
                                ax3.axvline(
                                    x=start,
                                    color="green",
                                    linestyle="-",
                                    alpha=0.4,
                                    linewidth=0.7,
                                )

                                # Add vertical line at word END (red)
                                ax3.axvline(
                                    x=end,
                                    color="red",
                                    linestyle=":",
                                    alpha=0.4,
                                    linewidth=0.7,
                                )

                                # Add word text at the word start position
                                ax3.text(
                                    start,
                                    y_pos,
                                    word,
                                    fontsize=7,
                                    color="black",
                                    rotation=90,
                                    ha="center",
                                    va="bottom",
                                    alpha=0.8,
                                    bbox=dict(
                                        boxstyle="round",
                                        facecolor="white",
                                        alpha=0.6,
                                        pad=0.1,
                                    ),
                                )

                                # Draw a horizontal span between start and end
                                y_min, y_max = ax3.get_ylim()
                                span_height = y_min + 0.05
                                ax3.plot(
                                    [start, end],
                                    [span_height, span_height],
                                    color="blue",
                                    alpha=0.3,
                                    linewidth=2,
                                )
                        except Exception as word_marker_error:
                            logger.warning(
                                f"Error adding word markers to diarization: {word_marker_error}"
                            )
                            # Continue even if word marking fails
                    else:
                        ax3.text(
                            view_end / 2,
                            0,
                            "No diarization data available",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                except Exception as diar_error:
                    logger.error(f"Error plotting diarization data: {diar_error}")
                    ax3.text(
                        view_end / 2,
                        0,
                        "Error loading diarization data",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

                # Plot cleared diarization segments in the fourth subplot
                ax4.set_title("Cleared Diarization Segments")
                ax4.set_xlabel("Time (seconds)")
                ax4.set_yticks([])  # Hide Y-axis

                # Use the same color scheme as the raw diarization
                try:
                    # Get the cleared diarization segments
                    if diarization_segments_cleared:
                        unique_speakers_in_cleared = list(
                            set(
                                segment.get("speaker", 0)
                                for segment in diarization_segments_cleared
                            )
                        )

                        # If we have cleared diarization segments, plot them
                        if diarization_segments_cleared:
                            # Create multiple rows for visualization
                            row_height = 0.15
                            row_spacing = 0.2
                            max_rows = len(
                                unique_speakers_in_cleared
                            )  # Use all unique speakers
                            speaker_rows = {}

                            # First pass to assign rows to speakers
                            for speaker_id in unique_speakers_in_cleared:
                                # Assign row position (0 at bottom, increasing upward)
                                speaker_rows[speaker_id] = len(speaker_rows)

                            # Plot segments with colors based on speaker in their assigned row
                            for segment in diarization_segments_cleared:
                                speaker = segment.get("speaker", 0)
                                # Only plot speakers that have assigned rows
                                if speaker in speaker_rows:
                                    row_idx = speaker_rows[speaker]
                                    row_pos = row_idx * row_spacing

                                    color = diar_colors[speaker % len(diar_colors)]
                                    start = segment.get("start", 0)
                                    end = segment.get("end", 0)

                                    # Skip invalid segments
                                    if start >= end or end <= 0:
                                        continue

                                    # Draw segment as horizontal bar
                                    ax4.barh(
                                        row_pos,
                                        end - start,
                                        left=start,
                                        height=row_height,
                                        color=color,
                                        alpha=0.7,
                                        edgecolor="black",
                                        linewidth=0.5,
                                    )

                                    # Add speaker label in the middle of segment if long enough
                                    if end - start > 2.0:
                                        ax4.text(
                                            start + (end - start) / 2,
                                            row_pos,
                                            f"화자 {speaker}",
                                            ha="center",
                                            va="center",
                                            fontsize=8,
                                            bbox=dict(
                                                boxstyle="round",
                                                facecolor="white",
                                                alpha=0.7,
                                            ),
                                        )

                            # Add legend for speakers
                            legend_elements = []
                            for speaker in sorted(speaker_rows.keys()):
                                color = diar_colors[speaker % len(diar_colors)]
                                legend_elements.append(
                                    Patch(facecolor=color, label=f"화자 {speaker}")
                                )

                            ax4.legend(handles=legend_elements, loc="upper right")

                            # Set y-axis limits to show all rows with some padding
                            ax4.set_ylim(-0.1, len(speaker_rows) * row_spacing + 0.1)
                        else:
                            ax4.text(
                                view_end / 2,
                                0,
                                "No cleared diarization data available",
                                ha="center",
                                va="center",
                                fontsize=12,
                            )
                    else:
                        ax4.text(
                            view_end / 2,
                            0,
                            "No cleared diarization data available",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                except Exception as diar_error:
                    logger.error(
                        f"Error plotting cleared diarization data: {diar_error}"
                    )
                    ax4.text(
                        view_end / 2,
                        0,
                        "Error loading cleared diarization data",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

                # Add word start/stop markers on the waveform
                try:
                    # Use different colors for start (green) and end (red) markers
                    for word_data in word_timestamps:
                        word = word_data["word"]
                        start = word_data["start"]
                        end = word_data["end"]

                        # Skip words outside the current view
                        if end < view_start or start > view_end:
                            continue

                        # Add vertical line at word start (green)
                        ax1.axvline(
                            x=start,
                            color="green",
                            linestyle="-",
                            alpha=0.4,
                            linewidth=0.5,
                        )

                        # Add vertical line at word end (red)
                        ax1.axvline(
                            x=end, color="red", linestyle=":", alpha=0.4, linewidth=0.5
                        )

                        # Draw word span on waveform
                        y_min, y_max = ax1.get_ylim()
                        span_height = (y_max - y_min) * 0.1

                        # Draw span line at bottom of waveform
                        ax1.plot(
                            [start, end],
                            [y_min + span_height * 0.5, y_min + span_height * 0.5],
                            color="blue",
                            alpha=0.3,
                            linewidth=2,
                        )

                        # Add word label centered on the span
                        if end - start > 0.3:  # Only add labels for longer words
                            ax1.text(
                                start + (end - start) / 2,
                                y_min + span_height,
                                word,
                                fontsize=8,
                                ha="center",
                                va="bottom",
                                bbox=dict(
                                    boxstyle="round", facecolor="white", alpha=0.7
                                ),
                            )
                except Exception as word_marker_error:
                    logger.warning(
                        f"Error adding word start/stop markers: {word_marker_error}"
                    )
                    # Continue without word markers

                # Adjust layout to prevent overlap
                plt.subplots_adjust(bottom=0.1, hspace=0.5)

                # Save the visualization
                output_path = f"temp/{audio_uuid}_diarization_visualization.png"
                plt.savefig(output_path)
                logger.info(f"Visualization saved to: {output_path}")

                # Display the plot
                plt.show()

            except Exception as viz_error:
                logger.error(
                    f"Error creating visualization: {viz_error}\n{traceback.format_exc()}"
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


async def process_analysis_job_gpt_diar_mlpyan2(analysis_job: AnalysisWork) -> None:
    """Process the analysis job using GPT for transcription and MLP-YAN for diarization."""
    try:
        audio_uuid = analysis_job.id
        if not audio_uuid:
            raise ValueError("Missing audio_uuid in analysis_job")

        # Set audio_uuid on the analysis_job object for consistency
        analysis_job.audio_uuid = audio_uuid

        # Now use audio_uuid in path construction
        transcript_dir = config.TEMP_DIR / "transcripts" / audio_uuid
        transcript_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리 자동 생성
        diar_debug_path = transcript_dir / f"id[{audio_uuid}]_diarization_debug.json"

        logger.info(
            f"Processing analysis job for {audio_uuid} with method: {analysis_job.options['diarization_method']}"
        )

        # Step 1: VAD splitting should already be done in process_audio_file
        # if not analysis_job.split_segments:
        #     raise ValueError("split_segments not found - VAD splitting must be done before this step")

        # Step 2 & 3: Run transcription and diarization in parallel
        tasks = []

        # Transcription task
        analysis_job.update_status(AudioStatus.TRANSCRIBING)
        transcription_task = asyncio.create_task(
            asyncio.to_thread(
                run_process_transcription, analysis_job, analysis_job.split_segments
            ),
            name="transcription",
        )
        tasks.append(transcription_task)

        # Diarization task
        analysis_job.update_status(AudioStatus.DIARIZING)
        diarization_task = asyncio.create_task(
            asyncio.to_thread(run_diarize_audio_whisperx, analysis_job),
            name="diarization",
        )
        tasks.append(diarization_task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions first
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_name = "transcription" if i == 0 else "diarization"
                logger.error(f"Error in {task_name} task: {result}")
                raise result

        result_transcription = results[0]
        diarization_segments = results[1]

        print("Async jobs done")
        print("len result_transcription", len(result_transcription["segments"]))
        print("len diarization_segments", len(diarization_segments))

        ## post process the whisperx result
        transcription_segments = result_transcription["segments"]
        trans_result = {"language": "ko", "segments": []}
        for transcription_segment in transcription_segments:
            whisperx_segment = {
                "id": 0,
                "seek": 0,
                "start": transcription_segment.get("start", 0),
                "end": transcription_segment.get("end", 0),
                "text": transcription_segment.get("text", "").strip(),
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
            trans_result["segments"].append(whisperx_segment)

        print("Assigning speakers to segments...")
        # diar_word_align_result = whisperx.assign_word_speakers(diarize_segments, aligned_result, fill_nearest=True)
        diar_word_align_result = whisperx.assign_word_speakers(
            diarization_segments, trans_result
        )
        diar_word_align_result["language"] = trans_result["language"]
        diar_segs = diar_word_align_result["segments"]

        print(f"len diar_segs {len(diar_segs)}")

        for seg in diar_segs:
            # print("seg: ", seg)
            speaker_id_num = -1
            if "speaker" in seg:
                speaker_short = seg["speaker"].replace("SPEAKER_", "")
                ## parse such as 00 01 02 ...
                try:
                    speaker_id_num = int(speaker_short)
                except ValueError as ve:
                    pass
            else:
                pass

            seg["speaker_str"] = seg["speaker"] if "speaker" in seg else "undecided"
            seg["speaker"] = speaker_id_num

            ## remove words because no longer needed
            if "words" in seg:
                del seg["words"]

        for segment in diar_segs:
            ## add for visualization
            segment["duration"] = segment["end"] - segment["start"]

        unique_speakers = list(set([seg["speaker"] for seg in diar_segs]))
        print(f"unique_speakers {unique_speakers}")
        num_speakers = len(unique_speakers)
        print(f"num_speakers {num_speakers}")

        analysis_job.complete_step(AudioStatus.TRANSCRIBING)
        analysis_job.complete_step(AudioStatus.DIARIZING)

        # Check for exceptions
        # for result in results:
        #     if isinstance(result, Exception):
        #         raise result

        # Save diarization results for frontend visualization
        audio_uuid = analysis_job.audio_uuid

        # Save frontend-friendly diarization data
        frontend_diarization = {
            "step_name": "diarizing",
            # "status": AudioStatus.DIARIZING,
            "diarization_segments": diar_segs,
            "total_duration": (
                max([seg["end"] for seg in diar_segs]) if diar_segs else 0
            ),
            "num_speakers": len(
                set(seg["speaker"] for seg in diar_segs if "speaker" in seg)
            ),
        }

        # Update job step with diarization visualization data for frontend
        analysis_job.update_step(frontend_diarization)

        # Save debug files for troubleshooting
        diar_debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diar_debug_path, "w", encoding="utf-8") as f:
            json.dump(diar_segs, f, ensure_ascii=False, indent=2)

        # Also save in uploads directory for frontend access
        uploads_dir = config.UPLOADS_DIR / audio_uuid
        uploads_dir.mkdir(parents=True, exist_ok=True)
        diar_frontend_path = uploads_dir / f"id[{audio_uuid}]_ml_diarized.json"
        with open(diar_frontend_path, "w", encoding="utf-8") as f:
            json.dump(frontend_diarization, f, ensure_ascii=False, indent=2)

        # Apply speaker info from diarization to transcription segments
        transcription_words = diar_segs
        # transcription_words_splits = result_transcription["text_splits"]

        # Save transcription segments before speaker assignment for debugging
        trans_debug_path = (
            transcript_dir / f"id[{audio_uuid}]_transcription_words_debug.json"
        )
        with open(trans_debug_path, "w", encoding="utf-8") as f:
            json.dump(transcription_words, f, ensure_ascii=False, indent=2)

        print(f"Finished transcription. Marking job as improving")
        # analysis_job.complete_step(AudioStatus.TRANSCRIBING)
        analysis_job.update_status(AudioStatus.IMPROVING)

        result_with_role = assign_speaker_roles(diar_word_align_result)
        save_path = (
            config.UPLOADS_DIR
            / audio_uuid
            / f"id[{audio_uuid}]_transcription_with_role.json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_with_role, f, ensure_ascii=False, indent=2)

        ## add id to segments
        for idx_seg, seg in enumerate(result_with_role["segments"]):
            result_with_role["segments"][idx_seg]["id"] = idx_seg

        separ_segments = result_with_role["segments"]
        consecutive_segments_result = create_consecutive_segments(separ_segments)
        continuous_segments = consecutive_segments_result["consecutive_segments"]

        save_path = (
            config.UPLOADS_DIR
            / audio_uuid
            / f"id[{audio_uuid}]_consecutive_segments.json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(consecutive_segments_result, f, ensure_ascii=False, indent=2)

        analysis_job.complete_step(AudioStatus.IMPROVING)

        # if config.is_save_temp_files:
        #     ## save 'trans_segs_with_ts_conti'
        #     save_path = config.TEMP_DIR / f"tmp060_trans_segs_with_ts_conti.json"
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         json.dump(trans_segs_with_ts_conti, f, ensure_ascii=False, indent=2)

        # print("transcription_segments", trans_segs_with_ts)
        # print("transcription_segments_with_diar", trans_segs_with_ts_with_diar)

        ## filter out segments with speaker -1
        len_before = len(continuous_segments)
        continuous_segments = [
            seg for seg in continuous_segments if seg["speaker_role"] != -1
        ]
        len_after = len(continuous_segments)
        logger.info(f"filtered out {len_before - len_after} segments with speaker -1")

        ## merge segments with same speaker continues speaking
        continuous_segments_after_filter = []
        prev_seg = None
        for idx_seg, seg in enumerate(continuous_segments):
            if prev_seg is None:
                prev_seg = seg
                continue
            if prev_seg["speaker"] == seg["speaker"]:  ## group with diar result
                # if prev_seg['speaker_role'] == seg['speaker_role']: ## group with same speaker_role
                prev_seg["end"] = seg["end"]
                prev_seg["text"] += " " + seg["text"].strip()
            else:
                continuous_segments_after_filter.append(prev_seg)
                prev_seg = seg
        # AFTER the loop, do this:
        if prev_seg is not None:
            continuous_segments_after_filter.append(prev_seg)

        # Create segments from transcription result with assigned speakers
        segments = []
        speakers = []
        current_speaker_id = 0
        speaker_id_map = {}

        # # Process segments to create Segment objects with proper speaker IDs
        # # for seg in trans_segs_with_ts_conti:
        # for seg in continuous_segments_after_filter:
        #     # speaker_id = seg.get("speaker")
        #     speaker_id = seg.get("speaker_role")
        #     # Skip "undecided" speaker ID in the mapping
        #     if speaker_id != -1 and speaker_id not in speaker_id_map:
        #         speaker_id_map[speaker_id] = current_speaker_id
        #         current_speaker_id += 1

        #     segment = Segment(
        #         id=len(segments),
        #         start=seg.get("start"),
        #         end=seg.get("end"),
        #         text=seg.get("text"),
        #         # If speaker_id is -1, keep it as is, otherwise map it
        #         speaker=-1 if speaker_id == -1 else speaker_id_map.get(speaker_id, 0)
        #     )
        #     segments.append(segment)

        for seg in continuous_segments_after_filter:
            original = seg["speaker_role"]  # or seg["speaker"], whichever you intended
            if original == -1:
                continue
            segments.append(
                Segment(
                    id=len(segments),
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    speaker=original,
                )
            )

        # Sort segments by start time
        segments.sort(key=lambda x: x.start)

        # Reassign IDs after sorting
        for i, segment in enumerate(segments):
            segment.id = i

        speaker_ids = sorted({seg.speaker for seg in segments if seg.speaker >= 0})
        num_speakers = len(speaker_ids)  # e.g. 3 if you saw [0,1,2]

        # 2) Build your Speaker list:
        speakers = [
            Speaker(id=i, role="counselor" if i == 0 else f"client{i}")
            for i in speaker_ids
        ]

        # Create speaker list, making sure the first speaker is the counselor
        # speakers = [
        #     Speaker(id=id, role="counselor" if id == 0 else f"client{id}")
        #     for id in range(len(speaker_id_map))
        # ]

        # Create the final transcription result with segments and speakers
        analysis_job.result = TranscriptionResult(segments=segments, speakers=speakers)

        if config.is_save_temp_files:
            save_path = config.TEMP_DIR / f"tmp080_analysis_job_result.json"
            save_analysis_work_to_json(analysis_job, save_path)

        ## save result file
        save_path = (
            config.UPLOADS_DIR
            / audio_uuid
            / f"id[{audio_uuid}]_transcription_result.json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(analysis_job.result.dict(), f, ensure_ascii=False, indent=2)

        # Save state after status update
        save_analysis_work_json(analysis_job)

        # Mark the improving step as completed
        # analysis_job.complete_step("improving")

        # Mark the job as completed
        print(f"Marking job as completed")
        analysis_job.update_status(AudioStatus.COMPLETING)

    except Exception as e:
        # Handle errors
        err_msg = f"ERROR in process_analysis_job_gpt_diar_mlpyan: {e}\n with traceback:\n{traceback.format_exc()}"
        logger.error(err_msg)
        analysis_job.error = str(e)
        analysis_job.update_status(AudioStatus.FAILED)


async def process_analysis_job_gpt_diar_mlpyan(
    analysis_job: AnalysisWork, is_display: bool = False
) -> None:
    """Process the analysis job using GPT for transcription and MLP-YAN for diarization."""
    try:
        # Get the audio UUID from the analysis job
        audio_uuid = analysis_job.id
        if not audio_uuid:
            raise ValueError("Missing audio_uuid in analysis_job")

        # Set audio_uuid on the analysis_job object for consistency
        analysis_job.audio_uuid = audio_uuid

        # Now use audio_uuid in path construction
        transcript_dir = config.TEMP_DIR / "transcripts" / audio_uuid
        transcript_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리 자동 생성
        diar_debug_path = transcript_dir / f"id[{audio_uuid}]_diarization_debug.json"

        logger.info(
            f"Processing analysis job for {audio_uuid} with method: {analysis_job.options['diarization_method']}"
        )

        # Step 1: VAD splitting
        await do_vad_split(analysis_job)

        # Step 2 & 3: Run transcription and diarization in parallel
        tasks = []

        # Transcription task
        analysis_job.update_status(AudioStatus.TRANSCRIBING)
        transcription_task = asyncio.create_task(
            asyncio.to_thread(
                run_process_transcription, analysis_job, analysis_job.split_segments
            ),
            name="transcription",
        )
        tasks.append(transcription_task)

        # Diarization task
        analysis_job.update_status(AudioStatus.DIARIZING)
        diarization_task = asyncio.create_task(
            # asyncio.to_thread(run_diarize_audio, analysis_job.file_path),
            asyncio.to_thread(
                run_diarize_audio_parallel, analysis_job, analysis_job.split_segments
            ),
            name="diarization",
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
            "step_name": "diarizing",
            # "status": AudioStatus.DIARIZING,
            "diarization_segments": diarization_segments,
            "total_duration": (
                max([seg["end"] for seg in diarization_segments])
                if diarization_segments
                else 0
            ),
            "num_speakers": len(
                set(seg["speaker"] for seg in diarization_segments if "speaker" in seg)
            ),
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
        transcription_words_splits = result_transcription["text_splits"]

        # Save transcription segments before speaker assignment for debugging
        trans_debug_path = (
            transcript_dir / f"id[{audio_uuid}]_transcription_words_debug.json"
        )
        with open(trans_debug_path, "w", encoding="utf-8") as f:
            json.dump(transcription_words, f, ensure_ascii=False, indent=2)

        is_do_text_improv = False
        if is_do_text_improv:
            ## update script with text improvement
            analysis_job.update_status(AudioStatus.IMPROVING)
            logger.info(f"improving transcription_lines with ai")
            logger.info("dbg11", result_transcription["text"])
            logger.info("dbg12", transcription_words_splits)
            # trans_lines = improve_transcription_lines(result_transcription["text"])
            trans_lines = improve_transcription_lines_parallel(
                transcription_words_splits
            )
            # logger.info("dbg22", trans_lines)

            ## (DBG) save 'improved_lines' to json file
            if config.is_save_temp_files:
                save_path = config.TEMP_DIR / f"tmp010_improved_lines.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(trans_lines, f, ensure_ascii=False, indent=2)
                save_path = config.TEMP_DIR / f"tmp020_transcription_words.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(transcription_words, f, ensure_ascii=False, indent=2)
                save_path = config.TEMP_DIR / f"tmp030_diarization_segments.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(diarization_segments, f, ensure_ascii=False, indent=2)

            # print("\ntrans_lines", trans_lines)
            # print("\ntranscription_words", transcription_words)
            # print("\ndiarization_segments", diarization_segments)

            logger.info(f"applying improved transcription_lines to transcription_words")
            trans_segs_with_ts = get_improved_lines_with_ts(
                trans_lines, transcription_words
            )
            # print("\ntrans_segs_with_ts", trans_segs_with_ts)

            if config.is_save_temp_files:
                ## save 'trans_segs_with_ts'
                save_path = config.TEMP_DIR / f"tmp040_trans_segs_with_ts.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(trans_segs_with_ts, f, ensure_ascii=False, indent=2)
        else:
            logger.info(f"skip text improvement")
            trans_segs_with_ts = (
                transcription_words  # this is already improved, segments.
            )

        logger.info(f"clear diarization result to have single speaker per segment")
        diarization_segments_cleared = get_cleared_diarization_segments(
            diarization_segments
        )
        if config.is_save_temp_files:
            save_path = config.TEMP_DIR / f"tmp050_cleared_diarization_segments.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(diarization_segments_cleared, f, ensure_ascii=False, indent=2)
        # print("\ncleared_diarization_segments", diarization_segments)

        logger.info(
            f"infer speaker from transcription_segments and diarization_segments"
        )
        trans_segs_with_ts_with_diar = get_seg_ts_with_diar_with_speaker_infer_wo_skip(
            trans_segs_with_ts, diarization_segments_cleared
        )

        ## merge segments with same speaker continues speaking
        trans_segs_with_ts_conti = []
        prev_seg = None
        for idx_seg, seg in enumerate(trans_segs_with_ts_with_diar):
            if prev_seg is None:
                prev_seg = seg
                continue
            if prev_seg["speaker"] == seg["speaker"]:
                prev_seg["end"] = seg["end"]
                prev_seg["text"] += " " + seg["text"].strip()
            else:
                trans_segs_with_ts_conti.append(prev_seg)
                prev_seg = seg
        # AFTER the loop, do this:
        if prev_seg is not None:
            trans_segs_with_ts_conti.append(prev_seg)

        if config.is_save_temp_files:
            ## save 'trans_segs_with_ts_conti'
            save_path = config.TEMP_DIR / f"tmp060_trans_segs_with_ts_conti.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(trans_segs_with_ts_conti, f, ensure_ascii=False, indent=2)

        # print("transcription_segments", trans_segs_with_ts)
        # print("transcription_segments_with_diar", trans_segs_with_ts_with_diar)

        # Create segments from transcription result with assigned speakers
        segments = []
        speakers = []
        current_speaker_id = 0
        speaker_id_map = {}

        # Process segments to create Segment objects with proper speaker IDs
        for seg in trans_segs_with_ts_conti:
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
                speaker=(
                    "undecided"
                    if speaker_id == "undecided"
                    else speaker_id_map.get(speaker_id, 0)
                ),
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
        analysis_job.result = TranscriptionResult(segments=segments, speakers=speakers)

        if config.is_save_temp_files:
            save_path = config.TEMP_DIR / f"tmp080_analysis_job_result.json"
            save_analysis_work_to_json(analysis_job, save_path)

        if is_display:
            # with matplotlib, display to timeline.
            # 첫 번째 행 :
            #     제목: "Audio Waveform with Transcription"
            #     내용: librosa를 사용한 오디오 파형 시각화
            #     표시 항목: 시간에 따른 오디오 진폭 그래프
            # 두 번째 행 :
            #     제목: "Pyannote Speaker Diarization Results"
            #     내용: pyannote 라이브러리 화자 분할 결과
            #     표시 항목: 시간별 화자 구분 색상 바, 화자 번호, 연결된 텍스트
            # 특징
            # 모든 행에서 시간 동기화된 뷰를 제공함
            # 각 행마다 다른 화자 분할 기술의 결과 비교 가능
            # 화자별 다른 색상 코드 사용 (각 방식마다 다른 색상표 적용)
            # 텍스트가 너무 긴 경우 max_char_in_label(15자) 제한으로 표시
            # 한글 폰트 지원을 위해 여러 폰트 후보 시도 (NanumGothic, Malgun Gothic 등)
            # 결과물은 PNG 이미지로 저장되고 plt.show()로 표시됨
            try:
                # Load audio for visualization
                audio, sr = librosa.load(analysis_job.file_path, sr=16000)
                duration = len(audio) / sr

                # Setup fonts for visualization
                font_candidates = [
                    "NanumGothic",
                    "Malgun Gothic",
                    "AppleGothic",
                    "Arial Unicode MS",
                    "NanumMyeongjo",
                ]
                font_found = False

                for font_name in font_candidates:
                    if any(
                        font_name.lower() in f.name.lower()
                        for f in mpl.font_manager.fontManager.ttflist
                    ):
                        mpl.rcParams["font.family"] = font_name
                        font_found = True
                        logger.info(f"Using font: {font_name}")
                        break

                if not font_found:
                    logger.warning(
                        "No suitable Korean font found. Some characters may not display correctly."
                    )

                # Alternative method for Korean fonts
                try:
                    # For macOS
                    if sys.platform == "darwin":
                        mpl.rc("font", family="AppleGothic")
                    # For Windows
                    elif sys.platform == "win32":
                        mpl.rc("font", family="Malgun Gothic")
                except:
                    pass

                # Create figure with 3 subplots and an additional subplot for the scrollbar
                fig = plt.figure(figsize=(14, 18))  # Increase height for 5 rows
                gs = fig.add_gridspec(
                    5, 1, height_ratios=[4, 4, 4, 4, 1]
                )  # 5 rows, last one for scrollbar

                # Create the main plotting axes
                ax1 = fig.add_subplot(gs[0])  # Audio waveform
                ax2 = fig.add_subplot(gs[1])  # Speaker diarization results
                ax3 = fig.add_subplot(gs[2])  # Raw diarization
                ax4 = fig.add_subplot(gs[3])  # Cleared diarization
                ax_scroll = fig.add_subplot(gs[4])  # Scrollbar axis

                # Set the default view range
                view_window = 30  # seconds
                view_start = 0
                view_end = min(view_window, duration)
                max_char_in_label = 15

                # Function to update all plots when scrolling
                def update_view(val):
                    view_start = val
                    view_end = min(val + view_window, duration)

                    # Update xlim for all plots
                    ax1.set_xlim(view_start, view_end)
                    ax2.set_xlim(view_start, view_end)
                    ax3.set_xlim(view_start, view_end)
                    ax4.set_xlim(view_start, view_end)

                    fig.canvas.draw_idle()

                # Set initial view limits for all plots
                ax1.set_xlim(view_start, view_end)
                ax2.set_xlim(view_start, view_end)
                ax3.set_xlim(view_start, view_end)
                ax4.set_xlim(view_start, view_end)

                # Create a slider
                slider = Slider(
                    ax_scroll,
                    "Time",
                    0,
                    max(0, duration - view_window),
                    valinit=0,
                    valstep=1,
                )
                slider.on_changed(update_view)

                # Plot audio waveform
                ax1.set_title("Audio Waveform with Transcription")
                librosa.display.waveshow(y=audio, sr=sr, ax=ax1)
                ax1.set_xlabel("Time (seconds)")
                ax1.set_ylabel("Amplitude")

                # Add word-level timestamps if available in the transcription
                # First, try to extract word-level timestamps from the transcription results
                word_timestamps = []

                # Check if we have any word-level timestamp information
                try:
                    logger.info("Looking for word-level timestamps for visualization")

                    # Try different sources for word-level transcription
                    for word_data in transcription_words:
                        if (
                            "text" in word_data
                            and "start" in word_data
                            and "end" in word_data
                        ):
                            word_timestamps.append(
                                {
                                    "word": word_data["text"],
                                    "start": word_data["start"],
                                    "end": word_data["end"],
                                }
                            )

                except Exception as word_error:
                    logger.warning(
                        f"Could not add word markers to waveform: {word_error}"
                    )
                    logger.warning(traceback.format_exc())
                    # Continue without word markers

                # Plot segments with different colors for speakers
                ax2.set_title("Speaker Diarization Results")
                ax2.set_xlabel("Time (seconds)")
                ax2.set_yticks([])  # Hide Y-axis

                # Define colors for speakers
                speaker_colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01"]

                unique_speakers_in_result = list(
                    set(segment.speaker for segment in segments)
                )
                print(f"len of unique_speakers: {len(unique_speakers_in_result)}")

                # Plot segments with colors based on speaker
                for segment in segments:
                    # Get speaker ID, handling "undecided" case
                    if segment.speaker == "undecided":
                        speaker = -1  # Special value for undecided
                    else:
                        speaker = segment.speaker

                    color = (
                        speaker_colors[speaker % len(speaker_colors)]
                        if speaker >= 0
                        else "#CCCCCC"
                    )
                    start = segment.start
                    end = segment.end
                    text = segment.text

                    # Draw segment
                    ax2.barh(
                        0, end - start, left=start, height=0.5, color=color, alpha=0.6
                    )

                    # Add speaker label and text
                    if speaker >= 0:
                        speaker_label = (
                            "상담사" if speaker == 0 else f"내담자 {speaker}"
                        )
                    else:
                        speaker_label = "미확인"

                    # Truncate text if too long
                    display_text = text
                    if len(text) > max_char_in_label:
                        display_text = text[: max_char_in_label - 3] + "..."

                    # Add text label
                    ax2.text(
                        start + (end - start) / 2,
                        0,
                        f"{speaker_label}: {display_text}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        rotation=90,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                    )

                # Add legend for speakers
                legend_elements = []

                # Find all unique speakers
                unique_speakers_in_result = set()
                for segment in segments:
                    if segment.speaker != "undecided":
                        unique_speakers_in_result.add(segment.speaker)

                # Add undecided speaker if needed
                has_undecided = any(
                    segment.speaker == "undecided" for segment in segments
                )

                for speaker in sorted(unique_speakers_in_result):
                    color = speaker_colors[speaker % len(speaker_colors)]
                    label = "상담사" if speaker == 0 else f"내담자 {speaker}"
                    legend_elements.append(Patch(facecolor=color, label=label))

                # Add undecided to legend if present
                if has_undecided:
                    legend_elements.append(Patch(facecolor="#CCCCCC", label="미확인"))

                ax2.legend(handles=legend_elements, loc="upper right")

                # Plot original diarization segments in the third subplot
                ax3.set_title("Pyannote Raw Diarization")
                ax3.set_xlabel("Time (seconds)")
                ax3.set_yticks([])  # Hide Y-axis

                # Define different colors for diarization visualization
                diar_colors = ["#8BC34A", "#E91E63", "#00BCD4", "#FF5722", "#9E9E9E"]

                # Get original diarization segments from the step results if available
                diarization_segments = []
                try:
                    # Try to get original diarization information
                    if "diarizing" in analysis_job.steps:
                        diar_step = analysis_job.steps["diarizing"]
                        if hasattr(diar_step, "get") and diar_step.get(
                            "diarization_segments"
                        ):
                            diarization_segments = diar_step["diarization_segments"]
                            logger.info(
                                f"Found {len(diarization_segments)} diarization segments to visualize"
                            )

                    # Fallback to frontend-friendly diarization data
                    if not diarization_segments and analysis_job.step_results.get(
                        "diarization"
                    ):
                        diarization_segments = analysis_job.step_results[
                            "diarization"
                        ].get("segments", [])
                        logger.info(
                            f"Using {len(diarization_segments)} segments from step_results"
                        )

                    # See if we can load from file as last resort
                    if not diarization_segments:
                        audio_uuid = analysis_job.audio_uuid
                        diar_path = (
                            config.UPLOADS_DIR
                            / str(audio_uuid)
                            / f"id[{audio_uuid}]_ml_diarized.json"
                        )
                        if os.path.exists(diar_path):
                            with open(diar_path, "r") as f:
                                diar_data = json.load(f)
                                if "diarization_segments" in diar_data:
                                    diarization_segments = diar_data[
                                        "diarization_segments"
                                    ]
                                    logger.info(
                                        f"Loaded {len(diarization_segments)} segments from file"
                                    )

                    unique_speakers_in_diar = list(
                        set(
                            segment.get("speaker", 0)
                            for segment in diarization_segments
                        )
                    )

                    # If we have diarization segments, plot them
                    if diarization_segments:
                        # Create multiple rows for visualization (up to 5)
                        row_height = 0.15
                        row_spacing = 0.2
                        max_rows = len(
                            unique_speakers_in_diar
                        )  # Use all unique speakers instead of fixed max
                        speaker_rows = {}

                        # First pass to assign rows to speakers
                        for speaker_id in unique_speakers_in_diar:
                            # Assign row position (0 at bottom, increasing upward)
                            speaker_rows[speaker_id] = len(speaker_rows)

                        # Plot segments with colors based on speaker in their assigned row
                        for segment in diarization_segments:
                            speaker = segment.get("speaker", 0)
                            # Only plot speakers that have assigned rows
                            if speaker in speaker_rows:
                                row_idx = speaker_rows[speaker]
                                row_pos = row_idx * row_spacing

                                color = diar_colors[speaker % len(diar_colors)]
                                start = segment.get("start", 0)
                                end = segment.get("end", 0)

                                # Skip invalid segments
                                if start >= end or end <= 0:
                                    continue

                                # Draw segment as horizontal bar
                                ax3.barh(
                                    row_pos,
                                    end - start,
                                    left=start,
                                    height=row_height,
                                    color=color,
                                    alpha=0.7,
                                    edgecolor="black",
                                    linewidth=0.5,
                                )

                                # Add speaker label in the middle of segment if long enough
                                if end - start > 2.0:
                                    ax3.text(
                                        start + (end - start) / 2,
                                        row_pos,
                                        f"화자 {speaker}",
                                        ha="center",
                                        va="center",
                                        fontsize=8,
                                        bbox=dict(
                                            boxstyle="round",
                                            facecolor="white",
                                            alpha=0.7,
                                        ),
                                    )

                        # Add legend for speakers
                        legend_elements = []
                        for speaker in sorted(speaker_rows.keys()):
                            color = diar_colors[speaker % len(diar_colors)]
                            legend_elements.append(
                                Patch(facecolor=color, label=f"화자 {speaker}")
                            )

                        ax3.legend(handles=legend_elements, loc="upper right")

                        # Set y-axis limits to show all rows with some padding
                        ax3.set_ylim(-0.1, len(speaker_rows) * row_spacing + 0.1)

                        # Add word markers on the third row diarization visualization
                        try:
                            # Display all word timestamps on the diarization plot
                            for word_data in word_timestamps:
                                word = word_data["word"]
                                start = word_data["start"]
                                end = word_data["end"]

                                # Get y position for the word markers
                                row_pos = 0
                                y_pos = row_pos + (row_height / 2) + 0.02

                                # Add vertical line at word START (green)
                                ax3.axvline(
                                    x=start,
                                    color="green",
                                    linestyle="-",
                                    alpha=0.4,
                                    linewidth=0.7,
                                )

                                # Add vertical line at word END (red)
                                ax3.axvline(
                                    x=end,
                                    color="red",
                                    linestyle=":",
                                    alpha=0.4,
                                    linewidth=0.7,
                                )

                                # Add word text at the word start position
                                ax3.text(
                                    start,
                                    y_pos,
                                    word,
                                    fontsize=7,
                                    color="black",
                                    rotation=90,
                                    ha="center",
                                    va="bottom",
                                    alpha=0.8,
                                    bbox=dict(
                                        boxstyle="round",
                                        facecolor="white",
                                        alpha=0.6,
                                        pad=0.1,
                                    ),
                                )

                                # Draw a horizontal span between start and end
                                y_min, y_max = ax3.get_ylim()
                                span_height = y_min + 0.05
                                ax3.plot(
                                    [start, end],
                                    [span_height, span_height],
                                    color="blue",
                                    alpha=0.3,
                                    linewidth=2,
                                )
                        except Exception as word_marker_error:
                            logger.warning(
                                f"Error adding word markers to diarization: {word_marker_error}"
                            )
                            # Continue even if word marking fails
                    else:
                        ax3.text(
                            view_end / 2,
                            0,
                            "No diarization data available",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                except Exception as diar_error:
                    logger.error(f"Error plotting diarization data: {diar_error}")
                    ax3.text(
                        view_end / 2,
                        0,
                        "Error loading diarization data",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

                # Plot cleared diarization segments in the fourth subplot
                ax4.set_title("Cleared Diarization Segments")
                ax4.set_xlabel("Time (seconds)")
                ax4.set_yticks([])  # Hide Y-axis

                # Use the same color scheme as the raw diarization
                try:
                    # Get the cleared diarization segments
                    if diarization_segments_cleared:
                        unique_speakers_in_cleared = list(
                            set(
                                segment.get("speaker", 0)
                                for segment in diarization_segments_cleared
                            )
                        )

                        # If we have cleared diarization segments, plot them
                        if diarization_segments_cleared:
                            # Create multiple rows for visualization
                            row_height = 0.15
                            row_spacing = 0.2
                            max_rows = len(
                                unique_speakers_in_cleared
                            )  # Use all unique speakers
                            speaker_rows = {}

                            # First pass to assign rows to speakers
                            for speaker_id in unique_speakers_in_cleared:
                                # Assign row position (0 at bottom, increasing upward)
                                speaker_rows[speaker_id] = len(speaker_rows)

                            # Plot segments with colors based on speaker in their assigned row
                            for segment in diarization_segments_cleared:
                                speaker = segment.get("speaker", 0)
                                # Only plot speakers that have assigned rows
                                if speaker in speaker_rows:
                                    row_idx = speaker_rows[speaker]
                                    row_pos = row_idx * row_spacing

                                    color = diar_colors[speaker % len(diar_colors)]
                                    start = segment.get("start", 0)
                                    end = segment.get("end", 0)

                                    # Skip invalid segments
                                    if start >= end or end <= 0:
                                        continue

                                    # Draw segment as horizontal bar
                                    ax4.barh(
                                        row_pos,
                                        end - start,
                                        left=start,
                                        height=row_height,
                                        color=color,
                                        alpha=0.7,
                                        edgecolor="black",
                                        linewidth=0.5,
                                    )

                                    # Add speaker label in the middle of segment if long enough
                                    if end - start > 2.0:
                                        ax4.text(
                                            start + (end - start) / 2,
                                            row_pos,
                                            f"화자 {speaker}",
                                            ha="center",
                                            va="center",
                                            fontsize=8,
                                            bbox=dict(
                                                boxstyle="round",
                                                facecolor="white",
                                                alpha=0.7,
                                            ),
                                        )

                            # Add legend for speakers
                            legend_elements = []
                            for speaker in sorted(speaker_rows.keys()):
                                color = diar_colors[speaker % len(diar_colors)]
                                legend_elements.append(
                                    Patch(facecolor=color, label=f"화자 {speaker}")
                                )

                            ax4.legend(handles=legend_elements, loc="upper right")

                            # Set y-axis limits to show all rows with some padding
                            ax4.set_ylim(-0.1, len(speaker_rows) * row_spacing + 0.1)
                        else:
                            ax4.text(
                                view_end / 2,
                                0,
                                "No cleared diarization data available",
                                ha="center",
                                va="center",
                                fontsize=12,
                            )
                    else:
                        ax4.text(
                            view_end / 2,
                            0,
                            "No cleared diarization data available",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )
                except Exception as diar_error:
                    logger.error(
                        f"Error plotting cleared diarization data: {diar_error}"
                    )
                    ax4.text(
                        view_end / 2,
                        0,
                        "Error loading cleared diarization data",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

                # Add word start/stop markers on the waveform
                try:
                    # Use different colors for start (green) and end (red) markers
                    for word_data in word_timestamps:
                        word = word_data["word"]
                        start = word_data["start"]
                        end = word_data["end"]

                        # Skip words outside the current view
                        if end < view_start or start > view_end:
                            continue

                        # Add vertical line at word start (green)
                        ax1.axvline(
                            x=start,
                            color="green",
                            linestyle="-",
                            alpha=0.4,
                            linewidth=0.5,
                        )

                        # Add vertical line at word end (red)
                        ax1.axvline(
                            x=end, color="red", linestyle=":", alpha=0.4, linewidth=0.5
                        )

                        # Draw word span on waveform
                        y_min, y_max = ax1.get_ylim()
                        span_height = (y_max - y_min) * 0.1

                        # Draw span line at bottom of waveform
                        ax1.plot(
                            [start, end],
                            [y_min + span_height * 0.5, y_min + span_height * 0.5],
                            color="blue",
                            alpha=0.3,
                            linewidth=2,
                        )

                        # Add word label centered on the span
                        if end - start > 0.3:  # Only add labels for longer words
                            ax1.text(
                                start + (end - start) / 2,
                                y_min + span_height,
                                word,
                                fontsize=8,
                                ha="center",
                                va="bottom",
                                bbox=dict(
                                    boxstyle="round", facecolor="white", alpha=0.7
                                ),
                            )
                except Exception as word_marker_error:
                    logger.warning(
                        f"Error adding word start/stop markers: {word_marker_error}"
                    )
                    # Continue without word markers

                # Adjust layout to prevent overlap
                plt.subplots_adjust(bottom=0.1, hspace=0.5)

                # Save the visualization
                output_path = f"temp/{audio_uuid}_diarization_visualization.png"
                plt.savefig(output_path)
                logger.info(f"Visualization saved to: {output_path}")

                # Display the plot
                plt.show()

            except Exception as viz_error:
                logger.error(
                    f"Error creating visualization: {viz_error}\n{traceback.format_exc()}"
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
    logger.info(
        f"Processing analysis job for {analysis_job.audio_uuid} with method: {analysis_job.options['diarization_method']}"
    )
    await do_vad_split(analysis_job)

    ## add tasks
    tasks = []
    analysis_job.update_status(AudioStatus.TRANSCRIBING)
    transcription_task = asyncio.create_task(
        asyncio.to_thread(
            run_process_transcription, analysis_job, analysis_job.split_segments
        ),
        name="transcription",
    )
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
        speaker_segments.append(
            {"start": seg["start"], "end": seg["end"], "speaker": speaker_id}
        )

    # Create speakers list
    speakers = [
        {"id": speaker_id, "role": "counselor" if speaker_id == 0 else "client"}
        for speaker_id in sorted(speaker_ids)
    ]

    # Store diarization result
    analysis_job.store_step_result(
        "diarization", {"speakers": speakers, "segments": speaker_segments}
    )

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
            speaker_diarized="",
        )
        for i, seg in enumerate(transcription_segments)
    ]

    # Sort segments by start time
    segments.sort(key=lambda x: x.start)

    ## save the result
    analysis_job.result = TranscriptionResult(
        text=result_transcription["text"], segments=segments, speakers=[]
    )

    # Store improving result (complete result with speaker identification)
    improving_segments = []
    for i, seg in enumerate(segments):
        improving_segments.append(
            {
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
            }
        )

    # Store the improving result
    analysis_job.store_step_result("improving", {"segments": improving_segments})

    save_transcription_result_json(
        analysis_job.audio_uuid, analysis_job.result, suffix="transcript"
    )
    save_analysis_work_json(analysis_job)

    pass


async def process_audio_file(
    analysis_job: AnalysisWork, background_tasks: BackgroundTasks
) -> None:
    """Process the audio file for the analysis job."""
    try:

        print(
            f"Processing audio file: {analysis_job.id} {analysis_job.status} {analysis_job.steps}"
        )
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
        elif diarization_method == "stt_apigpt_diar_apigpt3":
            await process_analysis_job_gpt_diar_gpt3(analysis_job)
        elif diarization_method == "stt_apiasm_diar_apiasm":
            await process_analysis_job_asm_diar_asm(analysis_job)
        elif diarization_method == "stt_apigpt_diar_mlpyan":
            await process_analysis_job_gpt_diar_mlpyan(analysis_job)
        elif diarization_method == "stt_apigpt_diar_mlpyan2":
            await process_analysis_job_gpt_diar_mlpyan2(analysis_job)
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


async def process_whisperx_transcription_and_diarization(
    analysis_job: AnalysisWork, split_segments: List[Dict]
) -> Dict:
    """Process transcription for all split segments."""
    total_splits = len(split_segments)
    segments_list = [None] * total_splits
    text_list = [None] * total_splits
    # full_text = [None] * total_splits
    processed_splits = 0
    diarization_method = analysis_job.options["diarization_method"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_language = "ko"  # 기본값 설정

    MAX_SPEAKERS_IN_SPLIT = 5

    def _transcribe(segment_path: str) -> Dict[str, Any]:
        client = get_openai_client()
        with open(segment_path, "rb") as audio_file:
            logger.debug("Sending audio to OpenAI API...")
            openai_result = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="ko",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        # OpenAI 결과를 딕셔너리로 변환 (이미 딕셔너리인 경우 그대로 사용)
        if not isinstance(openai_result, dict):
            openai_result = openai_result.model_dump()
            # print(f"openai_result: {openai_result}")

        # OpenAI 결과를 WhisperX 형식으로 변환
        logger.debug("Converting OpenAI result to compatible format...")

        words = openai_result.get("words", [])

        # 결과 구조 생성
        result = {"language": "ko", "segments": []}

        for word in words:
            whisperx_segment = {
                "id": 0,
                "seek": 0,
                "start": word.get("start", 0),
                "end": word.get("end", 0),
                "text": word.get("word", "").strip(),
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
            result["segments"].append(whisperx_segment)

        # print(f"whisperx_segments: {result}")

        print("Aligning segments... heuristic alignment")
        # 단어 단위 타임스탬프를 정렬 결과에 직접 추가
        aligned_result = {
            "segments": result["segments"].copy(),  # 세그먼트 복사
            "language": original_language,
        }
        words = openai_result.get("words", [])
        # 각 세그먼트에 단어 단위 타임스탬프 추가
        for segment in aligned_result["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            # segment_words = []

            # # 이 세그먼트에 속하는 단어들 찾기
            # for word in words:
            #     word_start = word.get("start", 0)
            #     if segment_start <= word_start <= segment_end:
            #         segment_words.append({
            #             "word": word.get("word", ""),
            #             "start": word.get("start", 0),
            #             "end": word.get("end", 0),
            #             "probability": 1.0
            #         })

            # 세그먼트에 단어 정보 추가
            # segment["words"] = segment_words

        return aligned_result

    def _diarize(segment_path: str, device: str, max_speakers: int = 5):
        ### pipe2 - Diarization Job
        custom_model = "custom_diarize/config.yaml"

        HF_TOKEN = os.getenv("HF_TOKEN")
        print("Loading diarization model... token: ", HF_TOKEN)
        diarize_model = whisperx.DiarizationPipeline(
            model_name=custom_model, use_auth_token=HF_TOKEN, device=device
        )

        print("Diarizing segments...")

        audio = whisperx.load_audio(segment_path)
        # audio = AudioSegment.from_file(segment_path)
        # diarize_segments = diarize_model(audio) #this fixes diarization error innate in pyannote model.
        # diarize_segments = diarize_model(audio, num_speakers=None) #this fixes diarization error innate in pyannote model.
        diarize_segments = diarize_model(
            audio, num_speakers=None, max_speakers=MAX_SPEAKERS_IN_SPLIT
        )  # this fixes diarization error innate in pyannote model.ㅇ

        ## memory cleanup
        del diarize_model
        torch.cuda.empty_cache()

        return diarize_segments

    # for idx_seg, segment in enumerate(split_segments):
    # for idx_split, split_segment in enumerate(split_segments):
    def fn_whisperx_transcription_and_diarization(idx_split, split_segment):
        try:
            nonlocal processed_splits

            segment_path = split_segment["path"]
            logger.info(
                f"Processing split {idx_split+1}/{total_splits}: {segment_path} in process_whisperx_transcription_and_diarization"
            )

            ### pipe1 - Transcription Job
            # client = get_openai_client()
            # with open(segment_path, "rb") as audio_file:
            #     logger.debug("Sending audio to OpenAI API...")
            #     openai_result = client.audio.transcriptions.create(
            #         file=audio_file,
            #         model="whisper-1",
            #         language="ko",
            #         response_format="verbose_json",
            #         timestamp_granularities=["word"]
            #     )

            # # OpenAI 결과를 딕셔너리로 변환 (이미 딕셔너리인 경우 그대로 사용)
            # if not isinstance(openai_result, dict):
            #     openai_result = openai_result.model_dump()
            #     print(f"openai_result: {openai_result}")

            # # OpenAI 결과를 WhisperX 형식으로 변환
            # logger.debug("Converting OpenAI result to compatible format...")

            # words = openai_result.get("words", [])

            # # 결과 구조 생성
            # result = {
            #     "language": "ko",
            #     "segments": []
            # }

            # for word in words:
            #     whisperx_segment = {
            #         "id": 0,
            #         "seek": 0,
            #         "start": word.get("start", 0),
            #         "end": word.get("end", 0),
            #         "text": word.get("word", "").strip(),
            #         "tokens": [],
            #         "temperature": 0.0,
            #         "avg_logprob": 0.0,
            #         "compression_ratio": 1.0,
            #         "no_speech_prob": 0.0
            #     }
            #     result["segments"].append(whisperx_segment)

            # # print(f"whisperx_segments: {result}")

            # print("Aligning segments... heuristic alignment")
            # # 단어 단위 타임스탬프를 정렬 결과에 직접 추가
            # aligned_result = {
            #     "segments": result["segments"].copy(),  # 세그먼트 복사
            #     "language": original_language
            # }
            # words = openai_result.get("words", [])
            # # 각 세그먼트에 단어 단위 타임스탬프 추가
            # for segment in aligned_result["segments"]:
            #     segment_start = segment["start"]
            #     segment_end = segment["end"]
            #     # segment_words = []

            #     # # 이 세그먼트에 속하는 단어들 찾기
            #     # for word in words:
            #     #     word_start = word.get("start", 0)
            #     #     if segment_start <= word_start <= segment_end:
            #     #         segment_words.append({
            #     #             "word": word.get("word", ""),
            #     #             "start": word.get("start", 0),
            #     #             "end": word.get("end", 0),
            #     #             "probability": 1.0
            #     #         })

            #     # 세그먼트에 단어 정보 추가
            #     # segment["words"] = segment_words

            # ### pipe2 - Diarization Job
            # custom_model = "custom_diarize/config.yaml"

            # HF_TOKEN = os.getenv("HF_TOKEN")
            # print("Loading diarization model... token: ", HF_TOKEN)
            # diarize_model = whisperx.DiarizationPipeline(
            #     model_name=custom_model,
            #     use_auth_token=HF_TOKEN,
            #     device=device
            # )

            # print("Diarizing segments...")

            # MAX_SPEAKERS_IN_SPLIT = 5
            # audio = whisperx.load_audio(segment_path)
            # # audio = AudioSegment.from_file(segment_path)
            # # diarize_segments = diarize_model(audio) #this fixes diarization error innate in pyannote model.ㅇ
            # # diarize_segments = diarize_model(audio, num_speakers=None) #this fixes diarization error innate in pyannote model.ㅇ
            # diarize_segments = diarize_model(audio, num_speakers=None, max_speakers=MAX_SPEAKERS_IN_SPLIT) #this fixes diarization error innate in pyannote model.ㅇ

            ### merged pipe3 - Assign Speakers to Segments

            is_use_thread = True
            if is_use_thread:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_trans = executor.submit(_transcribe, segment_path)
                    fut_diar = executor.submit(_diarize, segment_path, device)

                    aligned_result = fut_trans.result()  # 전사 완료 대기
                    diarize_segments = fut_diar.result()  # diarization 완료 대기
            else:
                aligned_result = _transcribe(segment_path)
                diarize_segments = _diarize(segment_path, device)

            # print("diarize_segments RAW: ", diarize_segments)
            # print("aligned_result: ", aligned_result)

            print("Assigning speakers to segments...")
            # diar_word_align_result = whisperx.assign_word_speakers(diarize_segments, aligned_result, fill_nearest=True)
            diar_word_align_result = whisperx.assign_word_speakers(
                diarize_segments, aligned_result
            )
            diar_word_align_result["language"] = original_language

            diar_segs = diar_word_align_result["segments"]

            # print("diar_word_align_result: ", diar_word_align_result)

            for seg in diar_segs:
                # print("seg: ", seg)
                speaker_id_num = -1
                if "speaker" in seg:
                    speaker_short = seg["speaker"].replace("SPEAKER_", "")
                    ## parse such as 00 01 02 ...
                    try:
                        speaker_id_num = int(speaker_short)
                    except ValueError as ve:
                        pass
                else:
                    pass

                seg["speaker_str"] = seg["speaker"] if "speaker" in seg else "undecided"
                seg["speaker"] = speaker_id_num

                ## remove words because no longer needed
                if "words" in seg:
                    del seg["words"]

            # print(f"Diarization result with assigning speakers: {diar_word_align_result}")

            ## apply idx seg
            offset_time = split_segment["start"]

            for segment in diar_segs:
                ## add for visualization
                segment["duration"] = segment["end"] - segment["start"]

                ## offset time for splits
                segment["start"] += offset_time
                segment["end"] += offset_time
                if segment["speaker"] != -1:
                    segment["speaker"] = (
                        segment["speaker"] + idx_split * MAX_SPEAKERS_IN_SPLIT
                    )

            unique_speakers = list(set([seg["speaker"] for seg in diar_segs]))
            print(f"unique_speakers in split {idx_split}: {unique_speakers}")
            num_speakers = len(unique_speakers)
            print(f"num_speakers in split {idx_split}: {num_speakers}")

            # Save transcription result as JSON

            whisper_transcript_json = {
                "index": idx_split,
                "segments": diar_segs,
                "speakers": unique_speakers,
                "original_language": original_language,
            }
            save_transcription_result(segment_path, whisper_transcript_json)

            processed_splits += 1
            analysis_job.update_step(
                {
                    "step_name": "transcribing",  # Always include explicit step_name
                    "status": "in_progress",
                    "total_splits": total_splits,
                    "processed_splits": processed_splits,
                    "percent_complete": (processed_splits / total_splits) * 100,
                }
            )

            segments_list[idx_split] = diar_segs

            # 메모리 확보를 위해 모델 언로드
            gc.collect()

        except Exception as e:
            logger.exception(
                f"Split {idx_split} failed, error: {e} with traceback: {traceback.format_exc()}"
            )
            raise

    is_use_thread = True
    futures = []
    if is_use_thread:
        with ThreadPoolExecutor(max_workers=3) as executor:
            for idx_split, split_segment in enumerate(split_segments):
                futures.append(
                    executor.submit(
                        fn_whisperx_transcription_and_diarization,
                        idx_split,
                        split_segment,
                    )
                )

            for future in as_completed(futures):
                future.result()
    else:
        for idx_split, split_segment in enumerate(split_segments):
            fn_whisperx_transcription_and_diarization(idx_split, split_segment)

    ## merging results
    segments_merged = []
    for idx_split, split_segment in enumerate(split_segments):
        print(f"split_segment: {split_segment}")
        segments_merged.extend(segments_list[idx_split])

    # print(f"segments_merged: {segments_merged}")
    unique_speakers = list(set([seg["speaker"] for seg in segments_merged]))
    print(f"total unique_speakers: {unique_speakers}")
    num_speakers = len(unique_speakers)
    print(f"total num_speakers: {num_speakers}")

    whisper_transcript_json_merged = {
        "index": -1,
        "segments": segments_merged,
        "speakers": unique_speakers,
        "original_language": original_language,
    }

    total_duration = split_segments[-1]["end"] - split_segments[0]["start"]
    analysis_job.update_step(
        {
            "step_name": "diarizing",
            "status": "completed",
            "diarization_segments": segments_merged,
            "total_duration": total_duration,
            "num_speakers": len(set(seg["speaker"] for seg in segments_merged)),
        }
    )

    analysis_job.complete_step(AudioStatus.TRANSCRIBING)
    analysis_job.complete_step(AudioStatus.DIARIZING)

    return whisper_transcript_json_merged


async def process_transcription(
    analysis_job: AnalysisWork, split_segments: List[Dict]
) -> Dict:
    """Process transcription for all split segments."""
    total_splits = len(split_segments)
    segments_list = [None] * total_splits
    text_list = [None] * total_splits
    # full_text = [None] * total_splits
    processed_splits = 0
    diarization_method = analysis_job.options["diarization_method"]

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {
            executor.submit(
                transcribe_segment,
                i,
                seg["path"] if isinstance(seg, dict) else seg,  # Handle both formats
                total_splits,
                diarization_method,
                analysis_job.options,  # Pass options to transcribe_segment
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
                            logger.error(
                                f"Error awaiting API result: {e}\n{traceback.format_exc()}"
                            )
                            continue

                    # Calculate time offset based on segment format
                    time_offset = sum(
                        len(
                            AudioSegment.from_file(
                                s["path"] if isinstance(s, dict) else s
                            )
                        )
                        / 1000
                        for s in split_segments[: result["index"]]
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
                    analysis_job.update_step(
                        {
                            "step_name": "transcribing",  # Always include explicit step_name
                            "status": "in_progress",
                            "total_splits": total_splits,
                            "processed_splits": processed_splits,
                            "percent_complete": (processed_splits / total_splits) * 100,
                        }
                    )

            except Exception as e:
                logger.error(
                    f"Error processing future result: {e}\n{traceback.format_exc()}"
                )
                continue

    all_segments = []
    full_text = []
    for i in range(total_splits):
        if segments_list[i] is not None:
            all_segments.extend(segments_list[i])
        if text_list[i] is not None:
            full_text.append(text_list[i])

    # Create transcription result
    # transcription_result = {
    #     "segments": all_segments,
    #     "text": " ".join(full_text)
    # }

    # Store transcription result as intermediate result
    transcription_segments = []
    for i, seg in enumerate(all_segments):
        # Add "undecided" as speaker for transcription step
        transcription_segments.append(
            {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": "undecided",
            }
        )

    # Store the transcription result in the step_results
    analysis_job.store_step_result(
        "transcription", {"segments": transcription_segments}
    )

    # After all splits are processed, mark transcribing as completed
    analysis_job.complete_step("transcribing")

    # Create the full text from the segments
    full_text_str = " ".join(full_text)

    return {
        "segments": transcription_segments,
        "text": full_text_str,  # Add the text key to the returned dictionary
        "text_splits": full_text,
    }


def transcribe_segment(i, split_path, total_splits, diarization_method, options=None):
    """Transcribes a single audio segment and returns the result."""
    logger.info(f"Processing split {i+1}/{total_splits}: {split_path}")

    try:

        # Use the WAV file directly since it's already in the correct format
        # if diarization_method == "stt_apigpt_diar_apigpt":

        if diarization_method == "stt_apiasm_diar_apiasm":
            logger.info(
                f"Using Assembly AI API for transcription of split {i+1} / {total_splits}"
            )
            api_result = asyncio.run(
                process_audio_with_assembly_ai(split_path, i, total_splits)
            )
        else:
            logger.info(
                f"Using OpenAI API for transcription of split {i+1} / {total_splits}"
            )
            api_result = asyncio.run(
                process_audio_with_openai_api(split_path, i, total_splits, options)
            )

        result = {
            "index": i,
            "segments": api_result["segments"],
            "text": api_result["text"],
            "is_coroutine": False,
        }

        # Save transcription result as JSON
        save_transcription_result(split_path, result)

        return result
    except Exception as split_error:
        logger.error(
            f"ERROR processing split {i+1}: {split_error}\n{traceback.format_exc()}"
        )
        return None  # Continue processing other segments

        # Continue even if saving fails


if __name__ == "__main__":

    pass
