import os
import time
import whisperx
import torch
import dotenv
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import openai
from openai import OpenAI
from moviepy.editor import *
import cv2
from PIL import Image, ImageDraw, ImageFont
import textwrap
import subprocess

# 환경 변수 로드 (HuggingFace 토큰을 위해)
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def process_audio(audio_path, output_dir="./output", model_name="large-v3", cut_linut_time=None):
    """
    한국어 오디오 파일을 처리하여 다양한 형식으로 출력
    
    Parameters:
        audio_path (str): 처리할 오디오 파일 경로
        output_dir (str): 출력 파일을 저장할 디렉토리
        model_name (str): 사용할 Whisper 모델 이름
    """
    # 타임스탬프를 추가해 각 실행 결과를 구분
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 로그 파일 설정
    log_path = os.path.join(run_dir, "process_log.txt")
    
    def log_message(message):
        """로그 메시지를 콘솔과 파일에 기록"""
        print(message)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{message}\n")
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    log_message(f"Processing started: {audio_path}")
    log_message(f"Output directory: {run_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_message(f"Using device: {device}")
    
    # 여기에 오디오 길이 제한 코드 추가
    if cut_linut_time is not None and cut_linut_time > 0:
        # 오디오 로드
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 샘플링 레이트가 16000Hz이므로 샘플 수 계산
            max_samples = int(cut_linut_time * sr)
            if len(y) > max_samples:
                log_message(f"Truncating audio to first {cut_linut_time} seconds (from {len(y)/sr:.2f} seconds)")
                y_truncated = y[:max_samples]
                
                # 임시 파일로 저장
                truncated_path = os.path.join(run_dir, f"{base_name}_truncated.wav")
                import soundfile as sf
                sf.write(truncated_path, y_truncated, sr)
                
                # 오디오 파일 경로를 트런케이트된 파일로 업데이트
                audio_path = truncated_path
                log_message(f"Working with truncated audio: {audio_path}")
            else:
                log_message(f"Audio is shorter than {cut_linut_time} seconds")
        except (ImportError, Exception) as e:
            log_message(f"Warning: Could not truncate audio: {str(e)}")
    
    # 오디오 로드
    log_message("Loading audio...")
    
    result = None
    original_language = "ko"  # 기본값 설정

    if not OPENAI_API_KEY:
        log_message("Error: OPENAI_API_KEY not found in environment variables")
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    if model_name == "whisper-1":
        # OpenAI API를 사용하여 전사
        log_message(f"Using OpenAI Whisper API for transcription...")
        
        try:
            with open(audio_path, "rb") as audio_file:
                log_message("Sending audio to OpenAI API...")
                openai_result = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko",
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )
            
            # OpenAI 결과를 WhisperX 형식으로 변환
            log_message("Converting OpenAI result to compatible format...")
            
            # OpenAI 결과를 딕셔너리로 변환 (이미 딕셔너리인 경우 그대로 사용)
            if not isinstance(openai_result, dict):
                openai_result = openai_result.model_dump()
            
            # 결과 구조 생성
            result = {
                "language": "ko",
                "segments": []
            }
            
            if "words" in openai_result:
                log_message("Processing word-level timestamps from OpenAI API")
                words = openai_result.get("words", [])
                
                # 먼저 세그먼트가 있는지 확인
                segments = openai_result.get("segments", [])
                
                if segments:
                    # 세그먼트가 있는 경우, 각 세그먼트에 단어 추가
                    for i, segment in enumerate(segments):
                        segment_start = segment.get("start", 0)
                        segment_end = segment.get("end", 0)
                        segment_text = segment.get("text", "")
                        
                        # 이 세그먼트에 속하는 단어들 찾기
                        segment_words = []
                        for word in words:
                            word_start = word.get("start", 0)
                            if segment_start <= word_start <= segment_end:
                                segment_words.append(word)
                        
                        whisperx_segment = {
                            "id": i,
                            "seek": 0,
                            "start": segment_start,
                            "end": segment_end,
                            "text": segment_text,
                            "tokens": [],
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 1.0,
                            "no_speech_prob": 0.0
                        }
                        
                        result["segments"].append(whisperx_segment)
                else:
                    # 세그먼트가 없는 경우, 전체 텍스트를 하나의 세그먼트로 생성
                    # full_text = " ".join([w.get("word", "").strip() for w in words])
                    # start_time = words[0].get("start", 0) if words else 0
                    # end_time = words[-1].get("end", 0) if words else 0
                    
                    # # 하나의 세그먼트 생성
                    # whisperx_segment = {
                    #     "id": 0,
                    #     "seek": 0,
                    #     "start": start_time,
                    #     "end": end_time,
                    #     "text": full_text,
                    #     "tokens": [],
                    #     "temperature": 0.0,
                    #     "avg_logprob": 0.0,
                    #     "compression_ratio": 1.0,
                    #     "no_speech_prob": 0.0
                    # }

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
                            "no_speech_prob": 0.0
                        }
                        result["segments"].append(whisperx_segment)
                
                log_message(f"Created {len(result['segments'])} segments from OpenAI word-level response")
            else:
                # 세그먼트 변환
                for i, segment in enumerate(openai_result.get("segments", [])):
                    result["segments"].append({
                        "id": i,
                        "seek": 0,
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "text": segment.get("text", ""),
                        "tokens": [],  # 필요시 토큰 정보 추가
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.0
                    })
            
            log_message(f"OpenAI transcription complete with {len(result['segments'])} segments")
            
            # 오디오 데이터 로드 (alignment를 위해 필요)
            audio = whisperx.load_audio(audio_path)
            
        except Exception as e:
            log_message(f"Error with OpenAI API: {str(e)}")
            raise
    else:
        # 1. WhisperX 모델 로드
        compute_type = "float16"  # 메모리 효율성을 위한 계산 타입
        batch_size = 8  # GTX 3070에 적절한 배치 사이즈
        
        log_message(f"Loading model {model_name}...")
        model = whisperx.load_model(
            model_name, 
            device, 
            compute_type=compute_type, 
            language="ko",
            vad_options={
                "chunk_size": 3,      # 기본값 30에서 10으로 변경
                "vad_onset": 0.1,      # 기본값 0.500에서 낮춤
                "vad_offset": 0.1      # 기본값 0.363에서 낮춤
            }
        )
        
        # 2. 오디오 로드 및 전사 처리
        audio = whisperx.load_audio(audio_path)

        if cut_linut_time is not None and cut_linut_time > 0:
            # 샘플링 레이트가 16000Hz이므로 샘플 수 계산
            max_samples = int(cut_linut_time * 16000)
            if len(audio) > max_samples:
                log_message(f"Truncating audio to first {cut_linut_time} seconds (from {len(audio)/16000:.2f} seconds)")
                audio = audio[:max_samples]
            else:
                log_message(f"Audio is shorter than {cut_linut_time} seconds")
        
        log_message("Transcribing with VAD...")
        result = model.transcribe(
            audio, 
            batch_size=batch_size,
            language="ko",
            chunk_size=3,
            print_progress=True
        )
        
        # 메모리 확보를 위해 모델 언로드
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # 단계 1: 초기 전사 결과 저장
    initial_transcription_path = os.path.join(run_dir, f"{base_name}_01_initial_transcription.json")
    with open(initial_transcription_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 단계 1 결과를 텍스트 파일로도 저장
    initial_transcription_txt = os.path.join(run_dir, f"{base_name}_01_initial_transcription.txt")
    with open(initial_transcription_txt, "w", encoding="utf-8") as f:
        f.write(f"Language detected: {result.get('language', 'ko')}\n\n")
        for i, segment in enumerate(result["segments"], 1):
            f.write(f"[{i}] {format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    
    log_message(f"Initial transcription saved to: {initial_transcription_path}")
    
    # 원본 언어 저장
    original_language = result.get("language", "ko")
    
    # 3. 강제 정렬 (단어 타임스탬프 생성)
    log_message("Loading alignment model...")
    model_align, metadata = whisperx.load_align_model(
        language_code=original_language,
        device=device
    )
    
    # OpenAI Whisper API를 사용한 경우 단어 단위 정보 직접 처리 가능
    if model_name == "whisper-1" and "words" in openai_result:
        log_message("Using word-level timestamps directly from OpenAI API")
        
        # 단어 단위 타임스탬프를 정렬 결과에 직접 추가
        aligned_result = {
            "segments": result["segments"].copy(),  # 세그먼트 복사
            "language": original_language
        }
        
        words = openai_result.get("words", [])
        
        # 각 세그먼트에 단어 단위 타임스탬프 추가
        for segment in aligned_result["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            segment_words = []
            
            # 이 세그먼트에 속하는 단어들 찾기
            for word in words:
                word_start = word.get("start", 0)
                if segment_start <= word_start <= segment_end:
                    segment_words.append({
                        "word": word.get("word", ""),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "probability": 1.0
                    })
            
            # 세그먼트에 단어 정보 추가
            segment["words"] = segment_words
    else:
        log_message("Performing alignment for word-level timestamps...")
        aligned_result = whisperx.align(
            result["segments"],
            model_align,
            metadata,
            audio,
            device,
            return_char_alignments=False,
            ## added
            interpolate_method="linear" #interpo. default was "nearest"
        )
    
    # 단계 2: 정렬 후 결과 저장
    alignment_path = os.path.join(run_dir, f"{base_name}_02_aligned.json")
    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, ensure_ascii=False, indent=2)
    
    # 단계 2 결과를 텍스트 파일로도 저장 (단어 타임스탬프 표시)
    alignment_txt = os.path.join(run_dir, f"{base_name}_02_aligned.txt")
    with open(alignment_txt, "w", encoding="utf-8") as f:
        f.write(f"Aligned transcription with word-level timestamps:\n\n")
        for i, segment in enumerate(aligned_result["segments"], 1):
            f.write(f"[{i}] {format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n")
            if "words" in segment:
                f.write("Words:\n")
                for word in segment["words"]:
                    f.write(f"  {word['word']} ({format_timestamp(word['start'])} --> {format_timestamp(word['end'])})\n")
            f.write("\n")
    
    log_message(f"Aligned transcription saved to: {alignment_path}")
    
    # 메모리 확보를 위해 정렬 모델 언로드
    del model_align
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. 화자 분리 (Diarization)
    final_result = aligned_result  # 기본값 설정
    
    if HF_TOKEN:
        log_message("Performing speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=device
        )
        
        diarize_segments = diarize_model(audio)
        
        # 단계 3: 화자 분리 결과 저장 (정렬 전) - 수정된 부분
        diarize_path = os.path.join(run_dir, f"{base_name}_03_diarize_segments.json")
        # Segment 객체를 직렬화 가능한 형식으로 변환
        serializable_diarize_data = []
        for i, row in diarize_segments.iterrows():
            serializable_diarize_data.append({
                'start': float(row['start']),
                'end': float(row['end']),
                'speaker': row['speaker']
            })
            
        with open(diarize_path, "w", encoding="utf-8") as f:
            json.dump(serializable_diarize_data, f, ensure_ascii=False, indent=2)
        
        # 화자 분리 결과를 텍스트로도 저장
        diarize_txt = os.path.join(run_dir, f"{base_name}_03_diarize_segments.txt")
        with open(diarize_txt, "w", encoding="utf-8") as f:
            f.write("Speaker diarization results:\n\n")
            for i, row in diarize_segments.iterrows():
                f.write(f"Segment {i+1}: {format_timestamp(row['start'])} --> {format_timestamp(row['end'])}\n")
                f.write(f"Speaker: {row['speaker']}\n\n")
        
        log_message(f"Diarization segments saved to: {diarize_path}")
        
        # 화자 분리 결과를 전사 결과와 합치기
        # final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result, fill_nearest=True)
        
        # 단계 4: 화자 분리 후 최종 결과 저장
        diarized_result_path = os.path.join(run_dir, f"{base_name}_04_diarized_result.json")
        with open(diarized_result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        # 화자 분리 후 최종 결과를 텍스트로도 저장
        diarized_txt = os.path.join(run_dir, f"{base_name}_04_diarized_result.txt")
        with open(diarized_txt, "w", encoding="utf-8") as f:
            f.write("Final transcription with speaker diarization:\n\n")
            for i, segment in enumerate(final_result["segments"], 1):
                speaker = segment.get("speaker", "UNKNOWN")
                f.write(f"[{i}] {format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])} [Speaker: {speaker}]\n")
                f.write(f"{segment['text'].strip()}\n")
                if "words" in segment:
                    f.write("Words:\n")
                    for word in segment["words"]:
                        word_speaker = word.get("speaker", speaker)
                        f.write(f"  {word['word']} ({format_timestamp(word['start'])} --> {format_timestamp(word['end'])}) [Speaker: {word_speaker}]\n")
                f.write("\n")
        
        log_message(f"Final diarized result saved to: {diarized_result_path}")
        
        # 메모리 확보를 위해 화자 분리 모델 언로드
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        log_message("Warning: HF_TOKEN not found, skipping diarization")
    
    # 언어 정보가 없는 경우 추가 (필수)
    if "language" not in final_result:
        final_result["language"] = original_language

    ## TODO: 화자 역할 할당
    def assign_speaker_roles(result, client):
        """
        화자별 역할 할당하는 함수
        
        Parameters:
            result (dict): 전사 결과 데이터
            client: OpenAI 클라이언트
        
        Returns:
            dict: 역할이 할당된 전사 결과
        """
        # 1. 화자별로 발화 내용 모으기 (대화 순서대로)
        conversation_text = []
        
        # 세그먼트 시간 기준으로 정렬
        sorted_segments = sorted(result["segments"], key=lambda x: x.get("start", 0))
        
        current_speaker = None
        current_text = []
        
        for segment in sorted_segments:
            if "speaker" in segment:
                speaker = segment["speaker"]
                text = segment["text"].strip()
                
                # 화자가 바뀌면 이전 텍스트 저장하고 새로 시작
                if current_speaker is not None and current_speaker != speaker:
                    if current_text:
                        conversation_text.append(f"[{current_speaker}] {' '.join(current_text)}")
                    current_text = [text]
                    current_speaker = speaker
                else:
                    # 같은 화자가 계속 말하는 경우
                    current_speaker = speaker
                    current_text.append(text)
        
        # 마지막 화자의 텍스트 추가
        if current_speaker is not None and current_text:
            conversation_text.append(f"[{current_speaker}] {' '.join(current_text)}")
        
        # 화자가 없는 경우 처리
        if not conversation_text:
            print("No speaker information found in the result.")
            return result
        
        # 대화 텍스트를 하나의 문자열로 결합
        conversation_string = "\n".join(conversation_text)
        
        # 2. OpenAI API를 통해 역할 할당
        json_schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "properties": {
                        "is_counseling": {
                            "type": "boolean",
                            "description": "대화가 심리상담 상황인지 여부"
                        },
                        "client_count": {
                            "type": "integer",
                            "description": "상담자를 제외한 내담자의 수"
                        },
                        "speakers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker_id": {
                                        "type": "string",
                                        "description": "화자 ID"
                                    },
                                    "role": {
                                        "type": "string",
                                        "description": "화자의 역할 이름",
                                        "enum": [
                                            "상담사",
                                            "내담자1",
                                            "내담자2",
                                            "내담자3",
                                            "내담자4",
                                            "내담자5",
                                            "내담자6",
                                            "내담자7",
                                            "내담자8",
                                            "내담자9",
                                            "내담자10"
                                        ]
                                    },
                                    "role_detail": {
                                        "type": "string",
                                        "description": "화자의 역할 상세 설명"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "description": "역할 할당의 확신도 (0-1)"
                                    }
                                },
                                "required": ["speaker_id", "role", "role_detail", "confidence"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["is_counseling", "client_count", "speakers"],
                    "additionalProperties": False
                }
            },
            "required": ["analysis"],
            "additionalProperties": False
        }
        
        # API 요청 메시지 구성
        messages = [
            {"role": "system", "content": "You are an expert in analyzing conversational data, especially in counseling sessions. Your task is to identify the roles of different speakers based on their speech patterns and content."},
            {"role": "user", "content": f"""
Please analyze the following conversation and identify the roles of each speaker.
Assume this is a counseling session. Determine how many clients there are and assign roles to each speaker ID.
Note that speech diarization might be inaccurate, so multiple speaker IDs might actually belong to the same person.

Conversation:
{conversation_string}

Provide your analysis in a structured format.
            """}
        ]

        print(f"role infer message: {messages}")
        
        try:
            # OpenAI API 호출
            completion = client.chat.completions.create(
                model="gpt-4.1",
                temperature=0.4,
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
            
            # 응답 처리
            response_content = completion.choices[0].message.content
            analysis = json.loads(response_content)

            print(f"role infer response: {response_content}")
            
            # 3. 결과를 전사 데이터에 추가
            result["speaker_analysis"] = analysis["analysis"]
            
            # 화자 ID에 역할 매핑
            speaker_roles = {speaker["speaker_id"]: speaker["role"] for speaker in analysis["analysis"]["speakers"]}
            
            # 세그먼트와 단어에 역할 정보 추가
            for segment in result["segments"]:
                if "speaker" in segment:
                    segment["speaker_role"] = speaker_roles.get(segment["speaker"], "Unknown")
                    
                    # 단어 수준에서도 역할 정보 추가
                    if "words" in segment:
                        for word in segment["words"]:
                            if "speaker" in word:
                                word["speaker_role"] = speaker_roles.get(word["speaker"], "Unknown")
            
            print(f"Speaker role assignment completed. Found {analysis['analysis']['client_count']} clients.")
            print(f"Speaker roles: {speaker_roles}")
            
        except Exception as e:
            print(f"Error in speaker role assignment: {str(e)}")
        
        return result
    
    # 5. 화자 역할 할당 수행 (OpenAI 클라이언트가 있는 경우만)
    if "speaker" in final_result.get("segments", [{}])[0] and OPENAI_API_KEY:
        log_message("Performing speaker role assignment...")
        final_result = assign_speaker_roles(final_result, client)
        
        # 역할 할당 결과 저장
        roles_result_path = os.path.join(run_dir, f"{base_name}_05_speaker_roles.json")
        with open(roles_result_path, "w", encoding="utf-8") as f:
            # 명시적으로 speaker_role 정보가 포함되어 있는지 확인
            roles_data = {
                "language": final_result.get("language", "ko"),
                "speaker_analysis": final_result.get("speaker_analysis", {}),
                "segments": []
            }
            
            # 각 세그먼트에 대해 speaker와 speaker_role 정보 명시적으로 추가
            for segment in final_result["segments"]:
                segment_copy = segment.copy()
                if "speaker" in segment:
                    segment_copy["speaker"] = segment["speaker"]
                    segment_copy["speaker_role"] = segment.get("speaker_role", "Unknown")
                roles_data["segments"].append(segment_copy)
            
            json.dump(roles_data, f, ensure_ascii=False, indent=2)
        log_message(f"Speaker role assignment result saved to: {roles_result_path}")
    else:
        print("Warning: No speaker information found in the result or OpenAI API key is not set.")
    
    # 언어 정보가 없는 경우 추가 (필수)
    if "language" not in final_result:
        final_result["language"] = original_language

    ## TODO: 연속 처리
    def create_consecutive_segments(segments):
        """
        같은 역할을 가진 연속된 세그먼트를 병합
        같은 역할의 세그먼트 n초 이내로 다시 나타날 경우에도 연속으로 처리
        동시발화 상황도 처리함
        
        Parameters:
            segments (list): 전사 세그먼트 목록
            
        Returns:
            dict: 병합된 연속 세그먼트 목록과 병합 로그
        """
        if not segments:
            return {"consecutive_segments": [], "merge_logs": []}
            
        # 세그먼트 시간 기준으로 정렬
        sorted_segments = sorted(segments, key=lambda x: x.get("start", 0))
        
        # 역할별로 세그먼트 그룹화
        role_groups = {}
        for segment in sorted_segments:
            role = segment.get("speaker_role", "Unknown")
            if role not in role_groups:
                role_groups[role] = []
            role_groups[role].append(segment.copy())
        
        # 각 역할 그룹 내에서 근접 세그먼트 병합
        merged_groups = {}
        merge_logs = []
        allow_overlap_time = 2.0
        
        for role, role_segments in role_groups.items():
            # 각 역할에 대해 클러스터 형성
            clusters = []
            
            for segment in role_segments:
                added_to_cluster = False
                
                # 기존 클러스터에 추가 가능한지 확인
                for cluster in clusters:
                    last_segment = cluster[-1]
                    time_gap = segment.get("start", 0) - last_segment.get("end", 0)
                    
                    # 직접 연속이거나 허용 시간 내 간격이면 클러스터에 추가
                    if time_gap <= allow_overlap_time:
                        # 시간 간격이 있는 경우 로그 생성
                        if time_gap > 0:
                            merge_log = {
                                "role": role,
                                "first_segment": {
                                    "speaker": last_segment.get("speaker", "UNKNOWN"),
                                    "start": last_segment.get("start", 0),
                                    "end": last_segment.get("end", 0),
                                    "text": last_segment.get("text", "")
                                },
                                "second_segment": {
                                    "speaker": segment.get("speaker", "UNKNOWN"),
                                    "start": segment.get("start", 0),
                                    "end": segment.get("end", 0),
                                    "text": segment.get("text", "")
                                },
                                "time_gap": time_gap
                            }
                            merge_logs.append(merge_log)
                            print(f"[근접 병합] 역할: {role}, 시간 간격: {time_gap:.2f}초")
                            print(f"  첫 세그먼트: {last_segment.get('speaker', 'UNKNOWN')} - {last_segment.get('text', '')[:30]}...")
                            print(f"  다음 세그먼트: {segment.get('speaker', 'UNKNOWN')} - {segment.get('text', '')[:30]}...")
                        
                        cluster.append(segment)
                        added_to_cluster = True
                        break
                
                # 어떤 클러스터에도 추가되지 않았으면 새 클러스터 생성
                if not added_to_cluster:
                    clusters.append([segment])
            
            # 각 클러스터를 하나의 세그먼트로 병합
            merged_segments = []
            
            for cluster in clusters:
                if len(cluster) == 1:
                    # 단일 세그먼트는 그대로 유지
                    merged_segments.append(cluster[0])
                else:
                    # 여러 세그먼트 병합
                    merged = cluster[0].copy()
                    merged["end"] = cluster[-1]["end"]
                    
                    # 텍스트 병합
                    texts = []
                    for i, seg in enumerate(cluster):
                        if i > 0:
                            time_gap = seg.get("start", 0) - cluster[i-1].get("end", 0)
                            if time_gap > 0:
                                texts.append("\n" + seg.get("text", ""))
                            else:
                                texts.append(" " + seg.get("text", ""))
                        else:
                            texts.append(seg.get("text", ""))
                    
                    merged["text"] = "".join(texts)
                    
                    # 단어 목록 병합 (있는 경우)
                    if any("words" in seg for seg in cluster):
                        merged["words"] = []
                        for seg in cluster:
                            if "words" in seg:
                                merged["words"].extend(seg["words"])
                    
                    merged_segments.append(merged)
            
            merged_groups[role] = merged_segments
        
        # 모든 역할의 세그먼트를 하나의 리스트로 결합하고 시간순 정렬
        consecutive_segments = []
        for role_segments in merged_groups.values():
            consecutive_segments.extend(role_segments)
        
        consecutive_segments = sorted(consecutive_segments, key=lambda x: x.get("start", 0))
        
        # 연속 세그먼트에 ID 부여
        for i, segment in enumerate(consecutive_segments):
            segment["contiseg_id"] = i
        
        return {
            "consecutive_segments": consecutive_segments,
            "merge_logs": merge_logs
        }
    
    # 연속 세그먼트 생성
    consecutive_segments_result = create_consecutive_segments(final_result["segments"])
    continuous_segments = consecutive_segments_result["consecutive_segments"]
    merge_logs = consecutive_segments_result["merge_logs"]
    
    final_result["continuous_segments"] = continuous_segments
    final_result["segment_merge_logs"] = merge_logs
    
    # 연속 세그먼트 결과 저장
    contiseg_result_path = os.path.join(run_dir, f"{base_name}_06_continuous_segments.json")
    with open(contiseg_result_path, "w", encoding="utf-8") as f:
        contiseg_data = {
            "language": final_result.get("language", "ko"),
            "continuous_segments": continuous_segments,
            "merge_logs": merge_logs
        }
        json.dump(contiseg_data, f, ensure_ascii=False, indent=2)
    log_message(f"Continuous segments saved to: {contiseg_result_path}")
    
    # 병합 로그가 있는 경우 별도 파일로도 저장
    if merge_logs:
        merge_logs_path = os.path.join(run_dir, f"{base_name}_06_segment_merge_logs.json")
        with open(merge_logs_path, "w", encoding="utf-8") as f:
            json.dump(merge_logs, f, ensure_ascii=False, indent=4)
        log_message(f"Found {len(merge_logs)} near-time segment merges. Logs saved to: {merge_logs_path}")
        
        # 텍스트 형식으로도 저장
        merge_logs_txt_path = os.path.join(run_dir, f"{base_name}_06_segment_merge_logs.txt")
        with open(merge_logs_txt_path, "w", encoding="utf-8") as f:
            f.write(f"역할 기반 세그먼트 근접 병합 로그 (총 {len(merge_logs)}개)\n")
            f.write("=============================================\n\n")
            
            for i, log in enumerate(merge_logs, 1):
                role = log["role"]
                time_gap = log["time_gap"]
                first_segment = log["first_segment"]
                second_segment = log["second_segment"]
                
                f.write(f"병합 #{i} - 역할: {role}\n")
                f.write(f"시간 간격: {time_gap:.2f}초\n")
                f.write(f"첫 세그먼트: [{first_segment['speaker']}] {first_segment['start']:.2f}s - {first_segment['end']:.2f}s\n")
                f.write(f"텍스트: {first_segment['text']}\n")
                f.write(f"다음 세그먼트: [{second_segment['speaker']}] {second_segment['start']:.2f}s - {second_segment['end']:.2f}s\n")
                f.write(f"텍스트: {second_segment['text']}\n\n")
        
        log_message(f"Segment merge logs also saved as text to: {merge_logs_txt_path}")

    ## TODO: detect overlap and save it as file.
    def detect_speech_overlaps(segments):
        """
        발화 중첩(overlap)을 감지하는 함수
        
        Parameters:
            segments (list): 전사 세그먼트 목록
            
        Returns:
            list: 중첩 발화 정보 목록
        """
        # 세그먼트 시간 기준으로 정렬
        sorted_segments = sorted(segments, key=lambda x: x.get("start", 0))
        overlaps = []
        
        # 각 세그먼트와 다른 세그먼트 비교하여 중첩 확인
        for i, segment1 in enumerate(sorted_segments):
            speaker1 = segment1.get("speaker", "UNKNOWN")
            role1 = segment1.get("speaker_role", "Unknown")
            start1 = segment1.get("start", 0)
            end1 = segment1.get("end", 0)
            
            for j in range(i+1, len(sorted_segments)):
                segment2 = sorted_segments[j]
                speaker2 = segment2.get("speaker", "UNKNOWN")
                role2 = segment2.get("speaker_role", "Unknown")
                start2 = segment2.get("start", 0)
                end2 = segment2.get("end", 0)
                
                # 다른 화자이고 시간이 중첩되는 경우
                if speaker1 != speaker2 and start2 < end1 and start1 < end2:
                    # 중첩 구간 계산
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    overlap_duration = overlap_end - overlap_start
                    
                    # 중첩이 0.01초보다 큰 경우만 저장 (노이즈 방지)
                    if overlap_duration > 0.01:
                        overlap_info = {
                            "start": overlap_start,
                            "end": overlap_end,
                            "duration": overlap_duration,
                            "speakers": [
                                {"speaker": speaker1, "role": role1, "text": segment1["text"]},
                                {"speaker": speaker2, "role": role2, "text": segment2["text"]}
                            ]
                        }
                        overlaps.append(overlap_info)
        
        return overlaps

    # 중첩 발화 감지 및 저장
    overlaps = detect_speech_overlaps(continuous_segments)
    final_result["speech_overlaps"] = overlaps

    # 중첩 발화 결과 저장
    overlaps_result_path = os.path.join(run_dir, f"{base_name}_07_speech_overlaps.json")
    with open(overlaps_result_path, "w", encoding="utf-8") as f:
        overlaps_data = {
            "language": final_result.get("language", "ko"),
            "speech_overlaps": overlaps
        }
        json.dump(overlaps_data, f, ensure_ascii=False, indent=2)

    # 중첩 발화 결과를 텍스트로도 저장
    overlaps_txt_path = os.path.join(run_dir, f"{base_name}_07_speech_overlaps.txt")
    with open(overlaps_txt_path, "w", encoding="utf-8") as f:
        f.write(f"발화 중첩 감지 결과 (총 {len(overlaps)}개)\n")
        f.write("======================================\n\n")
        
        for i, overlap in enumerate(overlaps, 1):
            start_time = format_timestamp(overlap["start"])
            end_time = format_timestamp(overlap["end"])
            duration = overlap["duration"]
            
            f.write(f"중첩 #{i}\n")
            f.write(f"시간: {start_time} --> {end_time} (길이: {duration:.2f}초)\n")
            f.write("화자:\n")
            
            for speaker_info in overlap["speakers"]:
                speaker = speaker_info["speaker"]
                role = speaker_info["role"]
                text = speaker_info["text"]
                f.write(f"  - [{speaker} - {role}] {text}\n")
            
            f.write("\n")

    log_message(f"Speech overlap detection completed. Found {len(overlaps)} overlaps.")
    log_message(f"Overlaps saved to: {overlaps_result_path} and {overlaps_txt_path}")

    # 5. 최종 결과 출력 파일 (표준 형식)
    json_path = os.path.join(run_dir, f"{base_name}.json")
    srt_path = os.path.join(run_dir, f"{base_name}.srt")
    txt_path = os.path.join(run_dir, f"{base_name}.txt")
    
    # JSON 파일 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    # 단순 텍스트 파일 저장 (발화자 포함)
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in final_result["segments"]:
            speaker = segment.get("speaker", "SPEAKER")
            speaker_role = segment.get("speaker_role", "")
            speaker_info = f"[{speaker}]" if not speaker_role else f"[{speaker} - {speaker_role}]"
            f.write(f"{speaker_info} {segment['text'].strip()}\n")
    
    # SRT 파일 직접 생성
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(final_result["segments"], 1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            speaker = segment.get("speaker", "SPEAKER")
            speaker_role = segment.get("speaker_role", "")
            speaker_info = f"[{speaker}]" if not speaker_role else f"[{speaker} - {speaker_role}]"
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{speaker_info} {text}\n\n")

    # TODO: make video with audio and such
    log_message("Creating visualization video...")
    video_path = create_visualization_video(audio_path, final_result, run_dir, base_name)
    log_message(f"Visualization video created: {video_path}")
    
    # 처리 요약 정보
    summary_path = os.path.join(run_dir, "processing_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"WhisperX Processing Summary\n")
        f.write(f"=========================\n\n")
        f.write(f"Input file: {audio_path}\n")
        f.write(f"Model used: {model_name}\n")
        f.write(f"Language: {final_result['language']}\n")
        f.write(f"Device: {device}\n")
        
        if model_name != "whisper-1":
            f.write(f"Compute type: {compute_type}\n")
            f.write(f"Batch size: {batch_size}\n")
        else:
            f.write(f"Used OpenAI Whisper API\n")
        f.write("\n")
        
        segment_count = len(final_result["segments"])
        total_duration = final_result["segments"][-1]["end"] if segment_count > 0 else 0
        f.write(f"Total segments: {segment_count}\n")
        f.write(f"Audio duration: {format_timestamp(total_duration)}\n\n")
        
        f.write("Generated files:\n")
        for root, dirs, files in os.walk(run_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                f.write(f"- {file} ({file_size:.2f} KB)\n")
    
    log_message(f"Processing complete! All output files saved to: {run_dir}")
    log_message(f"Standard output files: {json_path}, {srt_path}, {txt_path}")
    return run_dir

def format_timestamp(seconds):
    """SRT 형식에 맞는 타임스탬프 포맷 반환"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    msecs = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{msecs:03d}"

def create_visualization_video(audio_path, transcription_data, output_dir, base_name):
    """
    오디오 파일과 전사 결과를 사용하여 시각화 비디오를 생성합니다.
    
    Parameters:
        audio_path (str): 오디오 파일 경로
        transcription_data (dict): 전사 결과 데이터
        output_dir (str): 출력 디렉토리
        base_name (str): 파일 기본 이름
        
    Returns:
        str: 생성된 비디오 파일 경로
    """
    # 출력 비디오 파일 경로
    video_path = os.path.join(output_dir, f"{base_name}_visualization.mp4")
    
    # 세그먼트 정보 가져오기
    segments = transcription_data.get("segments", [])
    continuous_segments = transcription_data.get("continuous_segments", [])
    speech_overlaps = transcription_data.get("speech_overlaps", [])
    
    if not segments:
        print("No segments found in transcription data.")
        return None
    
    # 비디오 설정
    width, height = 1280, 720
    fps = 12
    
    # 오디오 로드
    try:
        audio_clip = AudioFileClip(audio_path)
        
        # 비디오 전체 길이 설정
        duration = audio_clip.duration
        if not duration and segments:
            # 오디오 길이를 가져올 수 없으면 마지막 세그먼트의 종료 시간 사용
            duration = segments[-1]["end"] + 5  # 여유 시간 추가
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        # 오디오 로드 실패 시 전사 데이터의 마지막 세그먼트 기준으로 설정
        if segments:
            duration = segments[-1]["end"] + 5
        else:
            duration = 60  # 기본값
    
    # 색상 매핑 (화자별 고유 색상)
    speakers = set()
    for segment in segments:
        if "speaker" in segment:
            speakers.add(segment["speaker"])
    
    # 화자 색상 매핑
    speaker_colors = {}
    speaker_colors_palette = [
        (255, 50, 50),    # 빨강
        (50, 150, 255),   # 파랑
        (50, 180, 50),    # 녹색
        (255, 180, 30),   # 노랑
        (180, 50, 180),   # 보라
        (255, 140, 0),    # 주황
        (0, 200, 200),    # 청록
        (130, 80, 50),    # 갈색
        (240, 100, 170)   # 분홍
    ]
    
    for i, speaker in enumerate(speakers):
        speaker_colors[speaker] = speaker_colors_palette[i % len(speaker_colors_palette)]
    
    # 역할별 고유 색상 설정
    roles = set()
    for segment in segments:
        if "speaker_role" in segment:
            roles.add(segment.get("speaker_role", "Unknown"))
    
    role_colors = {}
    role_colors_palette = [
        (100, 100, 255),  # 파스텔 파랑
        (255, 100, 100),  # 파스텔 빨강
        (100, 255, 100),  # 파스텔 녹색
        (255, 255, 100),  # 파스텔 노랑
        (255, 100, 255),  # 파스텔 보라
        (255, 180, 140),  # 파스텔 주황
        (100, 255, 255),  # 파스텔 청록
        (200, 150, 120),  # 파스텔 갈색
        (255, 180, 220)   # 파스텔 분홍
    ]
    
    for i, role in enumerate(roles):
        role_colors[role] = role_colors_palette[i % len(role_colors_palette)]
    
    # 한글 폰트 찾기 함수
    def find_korean_font():
        # 시스템 별로 자주 사용되는 한글 폰트 목록
        font_candidates = [
            # Windows
            "malgun.ttf", "Malgun Gothic", "맑은 고딕", "gulim.ttc", "Gulim", "굴림",
            # macOS
            "AppleGothic.ttf", "AppleGothic", "Apple Gothic",
            # Linux
            "NanumGothic.ttf", "Nanum Gothic", "NanumGothicBold.ttf",
            "NotoSansCJK-Regular.ttc", "Noto Sans CJK KR", "Noto Sans CJK",
            # Universal fallbacks
            "Arial Unicode MS", "Segoe UI"
        ]
        
        for font_name in font_candidates:
            try:
                # 폰트 로드 시도
                font = ImageFont.truetype(font_name, 24)
                print(f"Using Korean font: {font_name}")
                return font_name
            except (OSError, IOError):
                continue
        
        # 시스템 폰트 경로 확인
        font_paths = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            "/Library/Fonts",
            "/System/Library/Fonts",
            "C:\\Windows\\Fonts"
        ]
        
        # 폰트 확장자
        font_extensions = ['.ttf', '.ttc', '.otf']
        
        # 경로 내에서 폰트 파일 검색
        for path in font_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in font_extensions):
                            for keyword in ['korean', 'korea', 'hangul', 'cjk', 'gothic', 'nanum', 'malgun', 'gulim', 'batang']:
                                if keyword in file.lower():
                                    try:
                                        font_path = os.path.join(root, file)
                                        font = ImageFont.truetype(font_path, 24)
                                        print(f"Found Korean font: {font_path}")
                                        return font_path
                                    except (OSError, IOError):
                                        continue
        
        # 모든 시도 실패시 기본 폰트 반환
        print("Warning: No Korean font found, using default font")
        return None
    
    # 한글 폰트 찾기
    korean_font_name = find_korean_font()
    
    # 기본 이미지 생성 함수
    def create_frame(t):
        # 빈 이미지 생성
        img = np.ones((height, width, 3), dtype=np.uint8) * 240  # 연한 회색 배경
        
        # PIL 이미지로 변환
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            # 글꼴 설정
            if korean_font_name:
                # 한글 지원 폰트 사용
                title_font = ImageFont.truetype(korean_font_name, 30)
                text_font = ImageFont.truetype(korean_font_name, 24)
                time_font = ImageFont.truetype(korean_font_name, 20)
                label_font = ImageFont.truetype(korean_font_name, 18)
            else:
                # 대체 방법: PIL 기본 폰트
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
                time_font = ImageFont.load_default()
                label_font = ImageFont.load_default()
                print("Using default font - Korean may not display correctly")
        except:
            # 기본 글꼴 사용
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            time_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            print("Font loading error - Korean may not display correctly")
        
        # 제목 표시
        title_text = "오디오 전사본 (Audio Transcription) with Roles"
        title_width = draw.textlength(title_text, font=title_font) if hasattr(draw, 'textlength') else 350
        draw.text((width//2 - title_width//2, 30), title_text, fill=(0, 0, 0), font=title_font)
        
        # 현재 시간 표시 (좌측 하단)
        minutes = int(t // 60)
        seconds = int(t % 60)
        milliseconds = int((t % 1) * 1000)
        time_text = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        draw.text((20, height - 30), time_text, fill=(0, 0, 0), font=time_font)
        
        # 타임라인 1: 기본 세그먼트 (화자 기준)
        timeline1_y = height - 130
        timeline_height = 25
        
        # 타임라인 레이블
        draw.text((10, timeline1_y + 5), "화자", fill=(0, 0, 0), font=label_font)
        
        # 타임라인 배경 그리기
        draw.rectangle([(50, timeline1_y), (width - 50, timeline1_y + timeline_height)], 
                       fill=(200, 200, 200), outline=(0, 0, 0))
        
        # 중첩 구간 먼저 표시 (타임라인 1)
        for overlap in speech_overlaps:
            start = overlap.get("start", 0)
            end = overlap.get("end", 0)
            
            # 세그먼트가 타임라인에 표시되는 위치 계산
            segment_start_x = 50 + (width - 100) * (start / duration)
            segment_end_x = 50 + (width - 100) * (end / duration)
            
            # 중첩 구간 표시 (빨간색 배경으로 강조)
            draw.rectangle([(segment_start_x, timeline1_y), (segment_end_x, timeline1_y + timeline_height)], 
                          fill=(255, 150, 150), outline=(200, 0, 0), width=1)
        
        # 세그먼트 타임라인에 표시
        for segment in segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            # 세그먼트가 타임라인에 표시되는 위치 계산
            segment_start_x = 50 + (width - 100) * (start / duration)
            segment_end_x = 50 + (width - 100) * (end / duration)
            
            # 화자 색상 결정
            speaker = segment.get("speaker", "UNKNOWN")
            color = speaker_colors.get(speaker, (150, 150, 150))
            
            # 세그먼트 표시
            draw.rectangle([(segment_start_x, timeline1_y), (segment_end_x, timeline1_y + timeline_height)], 
                          fill=color, outline=(0, 0, 0))
        
        # 타임라인 2: 연속 세그먼트 (역할 기준)
        timeline2_y = height - 80
        
        # 타임라인 레이블
        draw.text((10, timeline2_y + 5), "역할", fill=(0, 0, 0), font=label_font)
        
        # 타임라인 배경 그리기
        draw.rectangle([(50, timeline2_y), (width - 50, timeline2_y + timeline_height)], 
                       fill=(200, 200, 200), outline=(0, 0, 0))
        
        # 중첩 구간 먼저 표시 (타임라인 2)
        for overlap in speech_overlaps:
            start = overlap.get("start", 0)
            end = overlap.get("end", 0)
            
            # 세그먼트가 타임라인에 표시되는 위치 계산
            segment_start_x = 50 + (width - 100) * (start / duration)
            segment_end_x = 50 + (width - 100) * (end / duration)
            
            # 중첩 구간 표시 (빨간색 배경으로 강조)
            draw.rectangle([(segment_start_x, timeline2_y), (segment_end_x, timeline2_y + timeline_height)], 
                          fill=(255, 150, 150), outline=(200, 0, 0), width=1)
        
        # 연속 세그먼트 타임라인에 표시
        for segment in continuous_segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            # 세그먼트가 타임라인에 표시되는 위치 계산
            segment_start_x = 50 + (width - 100) * (start / duration)
            segment_end_x = 50 + (width - 100) * (end / duration)
            
            # 역할 색상 결정
            role = segment.get("speaker_role", "Unknown")
            color = role_colors.get(role, (150, 150, 150))
            
            # 세그먼트 표시
            draw.rectangle([(segment_start_x, timeline2_y), (segment_end_x, timeline2_y + timeline_height)], 
                          fill=color, outline=(0, 0, 0))
        
        # 현재 활성 세그먼트 찾기 (기본 세그먼트)
        active_segments = [segment for segment in segments 
                           if segment.get("start", 0) <= t <= segment.get("end", 0)]
        
        # 현재 활성 연속 세그먼트 찾기
        active_contisegs = [segment for segment in continuous_segments 
                            if segment.get("start", 0) <= t <= segment.get("end", 0)]
        
        # 활성 세그먼트 텍스트 표시
        y_offset = 150
        
        # 먼저 연속 세그먼트 표시 (존재하는 경우)
        for segment in active_contisegs:
            role = segment.get("speaker_role", "Unknown")
            text = segment.get("text", "")
            color = role_colors.get(role, (150, 150, 150))
            
            # 역할 박스 그리기
            role_box = [(100, y_offset - 40), (width - 100, y_offset + 100)]
            draw.rectangle(role_box, fill=(240, 240, 240), outline=color, width=3)
            
            # 역할 표시
            draw.text((120, y_offset - 30), f"역할: {role}", fill=(0, 0, 0), font=text_font)
            
            # 시간 정보 표시
            start_time = format_timestamp(segment.get("start", 0))
            end_time = format_timestamp(segment.get("end", 0))
            time_info = f"{start_time} --> {end_time}"
            draw.text((width - 400, y_offset - 30), time_info, fill=(0, 0, 0), font=time_font)
            
            # 긴 텍스트 줄바꿈
            wrapped_text = textwrap.fill(text, width=50)
            draw.text((120, y_offset + 10), wrapped_text, fill=(0, 0, 0), font=text_font)
            
            y_offset += 150
        
        # 그 다음 기본 세그먼트 표시
        for segment in active_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            speaker_role = segment.get("speaker_role", "")
            text = segment.get("text", "")
            color = speaker_colors.get(speaker, (150, 150, 150))
            
            # 화자 박스 그리기
            speaker_box = [(100, y_offset - 40), (width - 100, y_offset + 100)]
            draw.rectangle(speaker_box, fill=(240, 240, 240), outline=color, width=3)
            
            # 화자 이름과 역할 명확하게 표시
            display_speaker = f"화자: {speaker}"
            if speaker_role:
                display_speaker = f"화자: {speaker} (역할: {speaker_role})"
            draw.text((120, y_offset - 30), display_speaker, fill=(0, 0, 0), font=text_font)
            
            # 시간 정보 표시
            start_time = format_timestamp(segment.get("start", 0))
            end_time = format_timestamp(segment.get("end", 0))
            time_info = f"{start_time} --> {end_time}"
            draw.text((width - 400, y_offset - 30), time_info, fill=(0, 0, 0), font=time_font)
            
            # 긴 텍스트 줄바꿈
            wrapped_text = textwrap.fill(text, width=50)
            draw.text((120, y_offset + 10), wrapped_text, fill=(0, 0, 0), font=text_font)
            
            y_offset += 150
        
        # 현재 위치 표시 (항상 맨 위에 그리기)
        if duration > 0:
            position_x = 50 + (width - 100) * (t / duration)
            cursor_wid = 4
            cursor_wid_half = cursor_wid // 2
            # 타임라인 1 위치 표시
            draw.rectangle([(position_x - cursor_wid_half, timeline1_y - cursor_wid_half), (position_x + cursor_wid_half, timeline1_y + timeline_height + cursor_wid_half)], 
                          fill=(255, 0, 0))
            # 타임라인 2 위치 표시
            draw.rectangle([(position_x - cursor_wid_half, timeline2_y - cursor_wid_half), (position_x + cursor_wid_half, timeline2_y + timeline_height + cursor_wid_half)], 
                          fill=(255, 0, 0))
        
        # 역할 범례 그리기
        legend_y = height - 40
        legend_text_y = legend_y + 5
        legend_square_size = 15
        legend_x_start = 300
        legend_x_gap = 100
        
        # 역할 범례 타이틀
        draw.text((250, legend_text_y), "역할:", fill=(0, 0, 0), font=label_font)
        
        # 각 역할별 범례 표시
        for i, role in enumerate(sorted(role_colors.keys())):
            legend_x = legend_x_start + (i * legend_x_gap)
            if legend_x + 120 > width - 250:  # 화면 너비를 벗어나면 표시 중단
                break
                
            color = role_colors[role]
            # 색상 사각형 그리기
            draw.rectangle(
                [(legend_x, legend_y), (legend_x + legend_square_size, legend_y + legend_square_size)],
                fill=color,
                outline=(0, 0, 0)
            )
            # 역할 텍스트 그리기
            draw.text(
                (legend_x + legend_square_size + 5, legend_text_y),
                role,
                fill=(0, 0, 0),
                font=label_font
            )
            
        # 화자 범례 그리기 (왼쪽에 배치)
        speaker_legend_y = height - 65
        speaker_legend_text_y = speaker_legend_y + 5
        
        # 화자 범례 타이틀
        draw.text((250, speaker_legend_text_y), "화자:", fill=(0, 0, 0), font=label_font)
        
        # 각 화자별 범례 표시
        for i, speaker in enumerate(sorted(speaker_colors.keys())):
            legend_x = legend_x_start + (i * legend_x_gap)
            if legend_x + 120 > width - 250:  # 화면 너비를 벗어나면 표시 중단
                break
                
            color = speaker_colors[speaker]
            # 색상 사각형 그리기
            draw.rectangle(
                [(legend_x, speaker_legend_y), (legend_x + legend_square_size, speaker_legend_y + legend_square_size)],
                fill=color,
                outline=(0, 0, 0)
            )
            # 화자 텍스트 그리기
            draw.text(
                (legend_x + legend_square_size + 5, speaker_legend_text_y),
                speaker,
                fill=(0, 0, 0),
                font=label_font
            )
        
        # 중첩 발화 범례 표시
        overlap_legend_x = width - 200
        overlap_legend_y = height - 65  # 화자 범례와 같은 높이
        draw.rectangle(
            [(overlap_legend_x, overlap_legend_y), (overlap_legend_x + legend_square_size, overlap_legend_y + legend_square_size)],
            fill=(255, 150, 150),
            outline=(200, 0, 0)
        )
        draw.text(
            (overlap_legend_x + legend_square_size + 5, overlap_legend_y + 5),
            "발화 중첩시 여기 팝업",
            fill=(200, 0, 0),
            font=label_font
        )

        # 중첩 타임라인 범례 (화자 타임라인)
        overlap_legend_x2 = width - 200
        overlap_legend_y2 = height - 40  # 역할 범례와 같은 높이
        draw.rectangle(
            [(overlap_legend_x2, overlap_legend_y2), (overlap_legend_x2 + legend_square_size, overlap_legend_y2 + legend_square_size)],
            fill=(255, 170, 170),
            outline=(200, 0, 0)
        )
        draw.text(
            (overlap_legend_x2 + legend_square_size + 5, overlap_legend_y2 + 5),
            "타임라인 중첩시 여기 팝업",
            fill=(200, 0, 0),
            font=label_font
        )
        
        # 중첩 발화 표시 (우측 하단)
        # 현재 시간(t)에 해당하는 중첩 발화 찾기
        active_overlaps = [overlap for overlap in speech_overlaps 
                          if overlap.get("start", 0) <= t <= overlap.get("end", 0)]
        
        if active_overlaps:
            # 중첩 발화가 있는 경우 알림 표시
            overlap_box_y = height - 90
            overlap_box_height = 45
            overlap_box = [(width - 400, overlap_box_y), (width - 20, overlap_box_y + overlap_box_height)]
            
            # 중첩 표시 배경 - 주의를 끄는 색상으로
            draw.rectangle(overlap_box, fill=(255, 230, 230), outline=(255, 0, 0), width=2)
            
            # 중첩 발화 정보 표시
            overlap = active_overlaps[0]  # 첫 번째 중첩 정보만 표시
            speakers_text = " & ".join([s["speaker"] for s in overlap["speakers"]])
            roles_text = " & ".join([s["role"] for s in overlap["speakers"]])
            
            draw.text((width - 390, overlap_box_y + 5), 
                     f"발화 중첩 감지! ({overlap['duration']:.1f}초)", 
                     fill=(200, 0, 0), 
                     font=text_font)
            
            draw.text((width - 390, overlap_box_y + 25), 
                     f"화자: {speakers_text}", 
                     fill=(0, 0, 0), 
                     font=label_font)
            
            draw.text((width - 390, overlap_box_y + 45), 
                     f"역할: {roles_text}", 
                     fill=(0, 0, 0), 
                     font=label_font)
        
        # NumPy 배열로 변환
        return np.array(pil_img)
    
    # 비디오 클립 생성
    video_clip = VideoClip(create_frame, duration=duration)
    
    # 오디오 추가
    try:
        video_clip = video_clip.set_audio(audio_clip)
    except:
        print("Warning: Could not add audio to video")
    
    # 비디오 저장
    try:
        video_clip.write_videofile(video_path, fps=fps, codec='libx264', audio_codec='aac')
        print(f"Video saved to: {video_path}")
    except Exception as e:
        print(f"Error writing video file: {str(e)}")
        try:
            # 대체 방법: ffmpeg 직접 호출
            temp_video_path = os.path.join(output_dir, f"{base_name}_temp.mp4")
            video_clip.write_videofile(temp_video_path, fps=fps, codec='libx264', audio=False)
            
            # ffmpeg로 오디오 추가
            cmd = [
                'ffmpeg', '-i', temp_video_path, '-i', audio_path, 
                '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', video_path
            ]
            subprocess.run(cmd, check=True)
            
            # 임시 파일 삭제
            os.remove(temp_video_path)
            print(f"Video (with audio added via ffmpeg) saved to: {video_path}")
        except Exception as e:
            print(f"Error with alternative video creation method: {str(e)}")
            
            # 최종 대안: 오디오 없이 저장
            try:
                video_clip.write_videofile(video_path, fps=fps, codec='libx264', audio=False)
                print(f"Video (without audio) saved to: {video_path}")
            except:
                print("Failed to create video visualization")
                return None
    
    # 리소스 정리
    try:
        video_clip.close()
        audio_clip.close()
    except:
        pass
    
    return video_path

if __name__ == "__main__":
    time_start = time.time()
    # file_path = "./my_files/audio-test_short_34sec.mp3"
    # file_path = "./my_files/김수현 1_250203_35min.m4a"
    file_path = "./my_files/커플 7_250203_1h19min 3자.m4a"
    # file_path = "./my_files/lsh_9_250328.MP3"
    # file_path = "./my_files/Recording 408 8주차.wav"

    if not os.path.exists(file_path):
        print(f"오류: 파일 '{file_path}'을 찾을 수 없습니다.")
    else:
        output_dir = process_audio(file_path, model_name='whisper-1', cut_linut_time=60*5)
        # output_dir = process_audio(file_path, model_name='whisper-1', cut_linut_time=60*10)
        # output_dir = process_audio(file_path, model_name='whisper-1')
        print(f"모든 출력 파일이 다음 디렉토리에 저장되었습니다: {output_dir}")
    time_end = time.time()
    print(f"총 처리 시간: {time_end - time_start:.2f}초")