import asyncio
import json
import os
import tempfile
import time
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
import re
import ollama

# Get logger
logger = get_logger(__name__)

# OpenAI client (lazy loading - will be loaded when needed)
_openai_client = None


def ask_ai_with_format(message, jsonformat, model="gemma3:4b"):
    if model.startswith("gpt"):
        return ask_openai_with_format(message, jsonformat, model)
    else:
        return ask_ollama_with_format(message, jsonformat, model)

def ask_ollama_with_format(messages, jsonformat, model="gemma3:4b"):
    response: ollama.ChatResponse = ollama.chat(
        model=model,
        messages=messages,
        format=jsonformat,
        stream=False
    )
    # print("response", response)
    # Extract the response content
    if response and 'message' in response:
        return json.loads(response['message']['content'], strict=False)
    else:
        return None

def ask_openai_with_format(messages, jsonformat, model="gpt-4.1-mini", temperature=0.3):
    completion = get_openai_client().chat.completions.create(
        model=model,  # or whichever model you prefer
        temperature=temperature,        # Adjust as needed
        # model="o3-mini",  # or whichever model you prefer
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": jsonformat
            }
        }
    )
    raw_response = completion.choices[0].message.content
    structured_response = json.loads(raw_response)
    return structured_response

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
                + json.dumps(segments, ensure_ascii=False)
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

def improve_transcription_lines_parallel(text_splits):
    """
    - Improve transcription text with gpt-4o
    - Use parallel processing to speed up the process
    - Maintains the order of the original text_splits
    """
    futures = []
    print(f"Improving {len(text_splits)} text splits in parallel")
    with ThreadPoolExecutor(max_workers=6) as executor:
        for idx_text_split, text_split in enumerate(text_splits):
            print(f"Improving text split {idx_text_split} of {len(text_splits)}")
            futures.append(executor.submit(improve_transcription_lines, text_split))
    
    # Instead of using as_completed, iterate over futures to preserve order
    improved_lines = []
    for idx, future in enumerate(futures):
        lines = future.result()
        for idx_line, line in enumerate(lines):
            print(f"Improved lines length: {len(line)} for idx {idx_line}, text: {line}")
            improved_lines.append(line)
    
    return improved_lines
    
    
    pass

def improve_transcription_lines(text):
    """
    - improve transcription text with gpt-4o
    - correct transcription words using improved text with cosine similarity
    """

#     prompt_text_improvement = """
# The following text is the result of speech-to-text (STT) transcription from a psychological counseling session. The STT contains errors. 

# First, Correct the text according to the context.
# Preserve the natural flow of spoken language. Use proper spacing and punctuation. Do not include explanations. Do not paraphrase.
# You may change the text to make it more natural and correct.
# When guessing the best text improvment, be aware that text often include expressions of emotions, personal feelings, and discussions of sensitive topics such as self-harm, suicidal thoughts, or intent to harm others.

# Second, break lines at each sentence aggressively. Break lines as much as possible.
# Break lines at natural pauses, sentence endings, question marks or periods, and possible speaker changes. Preserve the original meaning and flow of speech.
# """
# You may change the text to make it more natural and correct.
    prompt_text_improvement = """
The following text is the result of speech-to-text (STT) transcription from a psychological counseling session.
The STT contains errors. Correct the content according to the context.
When guessing the best text improvment, be aware that text often include expressions of emotions, personal feelings, and discussions of sensitive topics such as self-harm, suicidal thoughts, or intent to harm others.
Preserve the natural flow of spoken language. Use proper spacing and punctuation. Do not include explanations. Do not paraphrase.
Break lines at each sentence very aggressively, as aggressively as possible.
Break lines at natural pauses, sentence endings, question marks, periods, commas, conjunctions, connectives, and possible speaker changes.
Preserve the original meaning and flow of speech.
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
            temperature=0.4,        # Adjust as needed
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

    return improved_lines

def get_seg_ts_with_diar_with_speaker_infer_short(trans_segs_with_ts, diarization_segments):
    """
    - get transcription segments with diarization segments
    - for each diar_label, pick 5 segments.
    - with 5 segments, infer and assign speaker to each segment.
    """

    trans_segs_with_ts_with_diar = get_seg_ts_with_diar_wo_ai(trans_segs_with_ts, diarization_segments)
    
    # Group segments by diar_label using a dictionary
    diar_seg_dicts = {}
    for seg in trans_segs_with_ts_with_diar:
        diar_label = seg['diar_label']
        if diar_label not in diar_seg_dicts:
            diar_seg_dicts[diar_label] = []
        diar_seg_dicts[diar_label].append(seg)

    # Take some segments per diar_label
    for diar_label in diar_seg_dicts:
        if len(diar_seg_dicts[diar_label]) > 10:
            diar_seg_dicts[diar_label] = diar_seg_dicts[diar_label][:10]

    text_diars = ""
    for diar_label in diar_seg_dicts:
        diar_text = f"DIAR[{diar_label}] : \n"
        for seg in diar_seg_dicts[diar_label]:
            text_in_seg = seg['text']
            if len(text_in_seg) > 25:
                text_in_seg = text_in_seg[:25]
            diar_text += f" - {text_in_seg}\n"
        text_diars += diar_text + "\n"

    text = text_diars

    # Counselor tends to start the conversation more, ask more questions, cut-in more, use professional, empathetic, and clear language, often asking reflective and open-ended questions, providing guidance in a calm and supportive manner.
    prompt_text_improvement = """You are a helpful assistant to assign speaker information to diarization result.
There might be one counsler and at one or more clients.
Counselor tends to initiate the conversation, and guides the discussion, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance, Tends to interrupt politely.
Client tends to express personal emotions and experiences. The client's language may be less structured and more emotionally charged.
Read the samples of each diarization label, and guess which speaker is which.
Same diarization label means high probability of same speaker.

Input:
DIAR[diarization label] 
- text1
- text2

Output:
DIAR: (same as input)
SPEAKER: speaker id. 0 for Counselor, 1, 2, 3 ... for clients.
"""

    messages = [
        {"role": "system", "content": prompt_text_improvement},
        {"role": "user", "content": text}
    ]
    print(f"improving transcription with diarization result, text: {text}")

    json_schema = { ## string list
        "type": "object",
        "properties": {
            "diar_speakers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "diar": {"type": "number"},
                        "speaker": {"type": "number"}
                    },
                    "required": ["diar", "speaker"],
                    "additionalProperties": False
                }
            },
            "num_speakers": {"type": "number"}
        },
        "required": ["diar_speakers", "num_speakers"],
        "additionalProperties": False
    }

    diar_speakers = None
    num_speakers = None
    try:
        completion = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",  # or whichever model you prefer
            temperature=0.4,        # Adjust as needed
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
        response_content = completion.choices[0].message.content
        print(f"respone of seg_ts_with_diar: {response_content}")
        response_data = json.loads(response_content)
        diar_speakers = response_data.get("diar_speakers", [])
        num_speakers = response_data.get("num_speakers", None)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    print("diar_speakers", diar_speakers)

    for trans_seg in trans_segs_with_ts_with_diar:
        for diar_speaker in diar_speakers:
            if diar_speaker['diar'] == trans_seg['diar_label']:
                trans_seg['speaker'] = diar_speaker['speaker']
    
    # print("improved trans_segs_with_ts", trans_segs_with_ts_with_diar)
    print("guessed num_speakers", num_speakers)
    
    return trans_segs_with_ts_with_diar

def get_seg_ts_with_speaker_infer_wo_skip(trans_segs_with_ts_raw):
    """
    - get transcription segments with diarization segments
    """
    input_text = ""

    for idx_seg, seg in enumerate(trans_segs_with_ts_raw):
        seg['idx'] = idx_seg
    for idx_seg, seg in enumerate(trans_segs_with_ts_raw):
        # input_text += f"text: {seg['text']}, start: {seg['start']}, end: {seg['end']}, diar_label: {seg['diar_label']}\n"
        text_in_seg = seg['text']
        if len(text_in_seg) > 40:
            text_in_seg = text_in_seg[:15] + "..." + text_in_seg[-15:]
        elif len(text_in_seg) > 20:
            last_text = text_in_seg[-20:]
            is_special_char = re.match(r'^[^\w\s]+$', last_text)
            if is_special_char:
                text_in_seg = text_in_seg[:-20] + last_text
            else:
                text_in_seg = text_in_seg[:20] + "."
        # input_text += f"{seg['idx']}({seg['diar_label']}):{text_in_seg}\n"
        input_text += f"{seg['idx']}({-1}):{text_in_seg}\n"

    text = input_text
    
# Counselor tends to start the conversation more, ask more questions, cut-in more, use professional, empathetic, and clear language, often asking reflective and open-ended questions, providing guidance in a calm and supportive manner.

# Counselor tends to initiate the conversation, and guides the discussion, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance, Tends to interrupt politely.
# Client tends to express personal emotions, relationships, and experiences. The client's language may be less structured and more emotionally charged.

# IDX: index of the segment
# DIAR: diarization label. Same number means high probability of same speaker, but may have errors. -1 means no speaker is assigned.
# TEXT: text of the segment. fragment of the text.

    prompt_text_improvement = """You are a helpful assistant to assign speaker information to diarization result.
There might be one Counsler and at one or more clients.
Read the text, and guess how many speakers are there.

Counselor tends to initiate the conversation, and guides the discussion, gives explanation, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance. Tends to interrupt.

Given the data, assign speaker information to each segment.
Read the text, and assign the role of the speaker to each segment.

Data meaning is like this:
IDX: index of the segment
DIAR: diarization label. -1 means undecided.
TEXT: text of the segment. fragment of the text.

Data format is like this:
IDX(DIAR):TEXT

Output should be like this:
IDX: (same as input)
DIAR: (same as input, or guessed diar number)
SPEAKER: speaker number. 0 for Counselor, 1, 2, 3 ... for clients.
"""


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
                        "idx": {"type": "number"},
                        "diar": {"type": "number"},
                        "speaker": {"type": "number"}
                    },
                    "required": ["idx", "diar", "speaker"],
                    "additionalProperties": False
                }
            },
            "num_speakers": {"type": "number"}
        },
        "required": ["trans_segs_with_ts", "num_speakers"],
        "additionalProperties": False
    }

    trans_segs_with_ts = None
    num_speakers = None
    try:
        completion = get_openai_client().chat.completions.create(
            # model="gpt-4.1-mini",  # or whichever model you prefer
            # model="gpt-4o",  # or whichever model you prefer
            model="gpt-4.1-mini",  # or whichever model you prefer
            temperature=0.4,        # Adjust as needed
            messages=messages,
            max_tokens=16384, #default (max 16384 for gpt-4o, x2 for gpt-4.1)
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
        num_speakers = response_data.get("num_speakers", None)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    if trans_segs_with_ts is None:
        logger.error("Failed to improve transcription text somehow")
        return None
    
    ## for all 'trans_segs_with_ts', find 'text' with 'idx' in 'trans_segs_with_ts_with_diar'
    for seg in trans_segs_with_ts:
        for seg_in in trans_segs_with_ts_raw:
            if seg['idx'] == seg_in['idx']:
                seg['text'] = seg_in['text']
                seg['start'] = seg_in['start']
                seg['end'] = seg_in['end']
                

    print("improved trans_segs_with_ts", trans_segs_with_ts)
    print("guessed num_speakers", num_speakers)
    
    return trans_segs_with_ts

def get_seg_ts_with_diar_with_speaker_infer_wo_skip(trans_segs_with_ts, diarization_segments):
    """
    - get transcription segments with diarization segments
    """
    

    trans_segs_with_ts_with_diar = get_seg_ts_with_diar_wo_ai(trans_segs_with_ts, diarization_segments)

    if config.is_save_temp_files:
        save_path = config.TEMP_DIR / f"tmp055_trans_segs_with_ts_with_diar.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(trans_segs_with_ts_with_diar, f, ensure_ascii=False, indent=2)
    
    # Count each diar_label to understand distribution
    diar_count = {}
    for seg in trans_segs_with_ts_with_diar:
        diar_label = seg['diar_label']
        if diar_label not in diar_count:
            diar_count[diar_label] = 0
        diar_count[diar_label] += 1
    
    ## num max speaker id of diarization_segments
    num_speakers_max_diar_seg = max([seg['speaker'] for seg in diarization_segments]) + 1
    print(f"Diarization label counts: {diar_count}, num_speakers_max_diar_seg: {num_speakers_max_diar_seg}")
    
    input_text = ""
    for idx_seg, seg in enumerate(trans_segs_with_ts_with_diar):
        # input_text += f"text: {seg['text']}, start: {seg['start']}, end: {seg['end']}, diar_label: {seg['diar_label']}\n"
        text_in_seg = seg['text']
        if len(text_in_seg) > 40:
            text_in_seg = text_in_seg[:15] + "..." + text_in_seg[-15:]
        elif len(text_in_seg) > 20:
            last_text = text_in_seg[-20:]
            is_special_char = re.match(r'^[^\w\s]+$', last_text)
            if is_special_char:
                text_in_seg = text_in_seg[:-20] + last_text
            else:
                text_in_seg = text_in_seg[:20] + "."
        input_text += f"{seg['idx']}({seg['diar_label']}):{text_in_seg}\n"
        # input_text += f"{seg['idx']}({-1}):{text_in_seg}\n"

    text = input_text
    
# Counselor tends to start the conversation more, ask more questions, cut-in more, use professional, empathetic, and clear language, often asking reflective and open-ended questions, providing guidance in a calm and supportive manner.

# Counselor tends to initiate the conversation, and guides the discussion, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance, Tends to interrupt politely.
# Client tends to express personal emotions, relationships, and experiences. The client's language may be less structured and more emotionally charged.

# IDX: index of the segment
# DIAR: diarization label. Same number means high probability of same speaker, but may have errors. -1 means no speaker is assigned.
# TEXT: text of the segment. fragment of the text.

    prompt_text_improvement = """You are a helpful assistant to assign speaker information to diarization result.
There might be one Counsler and at one or more clients.
Read the text, and guess how many speakers are there.

Counselor tends to initiate the conversation, and guides the discussion, gives explanation, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance. Tends to interrupt.
Client tends to tell their experiences, express emotions. Client tends to speak continuously.

Given the data, assign speaker information to each segment.
Read the text, and assign the role of the speaker to each segment.

Data meaning is like this:
IDX: index of the segment
DIAR: diarization label. -1 means undecided.
TEXT: text of the segment. fragment of the text.

Data format is like this:
IDX(DIAR):TEXT

Output should be like this:
IDX: (same as input)
DIAR: (same as input, or guessed diar number)
SPEAKER: speaker number. 0 for Counselor, 1, 2, 3 ... for clients.
"""


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
                        "idx": {"type": "number"},
                        "diar": {"type": "number"},
                        "speaker": {"type": "number"}
                    },
                    "required": ["idx", "diar", "speaker"],
                    "additionalProperties": False
                }
            },
            "num_speakers": {"type": "number"}
        },
        "required": ["trans_segs_with_ts", "num_speakers"],
        "additionalProperties": False
    }

    trans_segs_with_ts = None
    num_speakers = None
    try:
        completion = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",  # or whichever model you prefer
            temperature=0.4,        # Adjust as needed
            messages=messages,
            max_tokens=16384, #default (max 16384 for gpt-4o, x2 for gpt-4.1)
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
        num_speakers = response_data.get("num_speakers", None)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    if trans_segs_with_ts is None:
        logger.error("Failed to improve transcription text somehow")
        return None
    
    ## for all 'trans_segs_with_ts', find 'text' with 'idx' in 'trans_segs_with_ts_with_diar'
    for seg in trans_segs_with_ts:
        for seg_in in trans_segs_with_ts_with_diar:
            if seg['idx'] == seg_in['idx']:
                seg['text'] = seg_in['text']
                seg['start'] = seg_in['start']
                seg['end'] = seg_in['end']
                

    print("improved trans_segs_with_ts", trans_segs_with_ts)
    print("guessed num_speakers", num_speakers)
    
    return trans_segs_with_ts

    
    pass


def get_seg_ts_with_diar_with_speaker_infer(trans_segs_with_ts, diarization_segments):
    """
    - get transcription segments with diarization segments

    used prompt with 'skip'
    prompt:
    for 'get_seg_ts_with_diar_with_speaker_infer', if more than 6 consecutive sequences that have same 'diar_label', leave just 2 left, and 2 right, removing others.
    get_seg_ts_with_diar_with_speaker_infer function, after AI infer, interpolate 'speaker' if 'idx' is not continuously exist in 'trans_segs_with_ts'. 
    """

    trans_segs_with_ts_with_diar = get_seg_ts_with_diar_wo_ai(trans_segs_with_ts, diarization_segments)

    if config.is_save_temp_files:
        save_path = config.TEMP_DIR / f"tmp056_trans_segs_with_ts_with_diar_with_skips.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(trans_segs_with_ts_with_diar, f, ensure_ascii=False, indent=2)
    
    # Reduce long consecutive sequences with same diar_label
    filtered_segs = []
    current_diar = None
    current_seq = []
    
    # Group consecutive segments with same diar_label
    for seg in trans_segs_with_ts_with_diar:
        if seg['diar_label'] != current_diar:
            # Process previous sequence if it exists
            if current_seq:
                if len(current_seq) >= 7:
                    print(f"long consecutive seq found. {len(current_seq)} segments -> leaving only few.")
                    filtered_segs.extend(current_seq[:3])
                    filtered_segs.extend(current_seq[-3:])
                else:
                    # Keep all segments in shorter sequences
                    filtered_segs.extend(current_seq)
            # Start new sequence
            current_diar = seg['diar_label']
            current_seq = [seg]
        else:
            # Continue current sequence
            current_seq.append(seg)
    
    # Process the last sequence
    if current_seq:
        if len(current_seq) >= 7:
            print(f"long consecutive seq found. {len(current_seq)} segments -> leaving only few.")
            filtered_segs.extend(current_seq[:3])
            filtered_segs.extend(current_seq[-3:])
        else:
            filtered_segs.extend(current_seq)

    
    save_path = config.TEMP_DIR / f"tmp057_filtered_trans_segs_with_ts_with_diar_with_skips.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(filtered_segs, f, ensure_ascii=False, indent=2)
    
    # Generate input text from filtered segments
    input_text = ""
    for idx_seg, seg in enumerate(filtered_segs):
        text_in_seg = seg['text']
        if len(text_in_seg) > 20:
            text_in_seg = text_in_seg[:20] # + "..."
        input_text += f"{seg['idx']}({seg['diar_label']}):{text_in_seg}\n"

    text = input_text
    
    prompt_text_improvement = """You are a helpful assistant to assign speaker information to diarization result.
There might be one counsler and at one or more clients.
Read the text, and guess how many speakers are there.

Counselor tends to initiate the conversation, and guides the discussion, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance, Tends to interrupt politely.
Client tends to express personal emotions, relationships, and experiences. The client's language may be less structured and more emotionally charged.

Given the data, assign speaker information to each segment.
Read the text, and assign the role of the speaker to each segment.

Data meaning is like this:
IDX: index of the segment
DIAR: diarization label. Same number means high probability of same speaker, but may have errors. -1 means no speaker is assigned.
TEXT: text of the segment. fragment of the text.

Data format is like this:
IDX(DIAR):TEXT

Output json element should be like this:
idx: (same as input)
diar: (same as input, or guessed diar number)
speaker: speaker number. 0 for Counselor, 1, 2, 3 ... for clients.
"""

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
                        "idx": {"type": "number"},
                        "diar": {"type": "number"},
                        "speaker": {"type": "number"}
                    },
                    "required": ["idx", "diar", "speaker"],
                    "additionalProperties": False
                }
            },
            "num_speakers": {"type": "number"}
        },
        "required": ["trans_segs_with_ts", "num_speakers"],
        "additionalProperties": False
    }

    trans_segs_with_ts_inferred = None
    num_speakers = None
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
        response_content = completion.choices[0].message.content
        print(f"respone of seg_ts_with_diar: {response_content}")
        response_data = json.loads(response_content)
        trans_segs_with_ts_inferred = response_data.get("trans_segs_with_ts", [])
        num_speakers = response_data.get("num_speakers", None)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    if trans_segs_with_ts_inferred is None:
        logger.error("Failed to improve transcription text somehow")
        return None
    
    if config.is_save_temp_files:
        save_path = config.TEMP_DIR / f"tmp058_trans_segs_with_ts_inferred.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(trans_segs_with_ts_inferred, f, ensure_ascii=False, indent=2)
    
    
    ## interpolate 'speaker' if 'idx' is not continuous. some idx might be missing.
    # Create a mapping of idx -> speaker from AI-inferred segments
    idx_to_speaker = {}
    for seg in trans_segs_with_ts_inferred:
        idx_to_speaker[seg['idx']] = seg['speaker']
    
    # Create a list of all segment indices that have been processed
    processed_indices = sorted(idx_to_speaker.keys())
    
    # Create final result array with all segments from the original input
    result_segments = []
    
    for seg in trans_segs_with_ts_with_diar:
        idx = seg['idx']
        result_seg = {
            'idx': idx,
            'text': seg['text'],
            'start': seg['start'],
            'end': seg['end']
        }
        
        # If the segment was processed by AI, use that speaker
        if idx in idx_to_speaker:
            result_seg['speaker'] = idx_to_speaker[idx]
        else:
            # Otherwise, interpolate from nearest segments
            # Find the nearest segment before this one
            prev_idx = None
            for processed_idx in processed_indices:
                if processed_idx < idx:
                    prev_idx = processed_idx
                else:
                    break
            
            # Find the nearest segment after this one
            next_idx = None
            for processed_idx in reversed(processed_indices):
                if processed_idx > idx:
                    next_idx = processed_idx
                else:
                    break
            
            # Interpolate speaker based on nearest segments
            if prev_idx is not None and next_idx is not None:
                # If both exist, use the nearest one
                prev_speaker = idx_to_speaker[prev_idx]
                next_speaker = idx_to_speaker[next_idx]
                
                if prev_speaker == next_speaker:
                    # If both have the same speaker, use that
                    result_seg['speaker'] = prev_speaker
                else:
                    # Otherwise, use the closest one
                    if (idx - prev_idx) <= (next_idx - idx):
                        result_seg['speaker'] = prev_speaker
                    else:
                        result_seg['speaker'] = next_speaker
            elif prev_idx is not None:
                # Only previous segment exists
                result_seg['speaker'] = idx_to_speaker[prev_idx]
            elif next_idx is not None:
                # Only next segment exists
                result_seg['speaker'] = idx_to_speaker[next_idx]
            else:
                # Neither exists (shouldn't happen if we have any processed segments)
                result_seg['speaker'] = 0  # Default to counselor
        
        result_segments.append(result_seg)

    if config.is_save_temp_files:
        save_path = config.TEMP_DIR / f"tmp059_result_segments_after_interpolation.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_segments, f, ensure_ascii=False, indent=2)
    
    
    
    print("improved and interpolated trans_segs_with_ts", result_segments)
    print("guessed num_speakers", num_speakers)
    
    return result_segments

    
    pass

def get_seg_ts_with_diar_wo_ai_with_finding_closest(trans_segs_with_ts, diarization_segments):
    """
    trans_segs_with_ts: [{'text': str, 'start': float, 'end': float}, ...]
    diarization_segments: [{'start': float, 'end': float, 'speaker': str/int}, ...]

    각 전사 세그먼트에 대해, 다이어라이제이션 세그먼트와의 겹치는 시간을 계산하여
    가장 겹치는 시간이 큰 다이어라이제이션 세그먼트의 화자 정보를 할당합니다.
    If no overlap, find the closest segment by distance.

    Returns:
        list of dict: [{'text': str, 'start': float, 'end': float, 'speaker': str/int}, ...]
    """
    updated_segments = []
    for idx_seg, t_seg in enumerate(trans_segs_with_ts):
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        
        max_overlap = 0
        min_distance = float('inf')
        assigned_speaker = -1
        
        for d_seg in diarization_segments:
            d_start = d_seg["start"]
            d_end = d_seg["end"]
            
            # 겹치는 시간 계산
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = overlap_end - overlap_start
            
            # 0보다 큰 경우에만 겹친다고 판단
            if overlap > 0 and overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = d_seg["speaker"]
                min_distance = 0  # Reset minimum distance as we found an overlap
            
            # Calculate distance for non-overlapping segments
            elif overlap <= 0:
                # Calculate closest distance
                if t_end < d_start:  # Transcript ends before diarization starts
                    distance = d_start - t_end
                elif d_end < t_start:  # Diarization ends before transcript starts
                    distance = t_start - d_end
                else:
                    distance = 0  # Should not happen as we checked for overlap
                
                if distance < min_distance:
                    # Only assign if we haven't found an overlap yet
                    if max_overlap == 0:
                        min_distance = distance
                        assigned_speaker = d_seg["speaker"]
                
        # 가장 많이 겹치는 화자 정보를 세그먼트에 담는다.
        updated_segments.append({
            "idx": idx_seg,
            "text": t_seg["text"],
            "start": t_start,
            "end": t_end,
            "diar_label": assigned_speaker
        })
    
    return updated_segments

def get_seg_ts_with_diar_wo_ai(trans_segs_with_ts, diarization_segments):
    """
    trans_segs_with_ts: [{'text': str, 'start': float, 'end': float}, ...]
    diarization_segments: [{'start': float, 'end': float, 'speaker': str/int}, ...]

    각 전사 세그먼트에 대해, 다이어라이제이션 세그먼트와의 겹치는 시간을 계산하여
    가장 겹치는 시간이 큰 다이어라이제이션 세그먼트의 화자 정보를 할당합니다.
    If no overlap, the diar_label remains -1.

    Returns:
        list of dict: [{'text': str, 'start': float, 'end': float, 'diar_label': str/int}, ...]
    """
    updated_segments = []
    for idx_seg, t_seg in enumerate(trans_segs_with_ts):
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        
        max_overlap = 0
        assigned_speaker = -1  # Default to -1 (no speaker assigned)
        
        for d_seg in diarization_segments:
            d_start = d_seg["start"]
            d_end = d_seg["end"]
            
            # 겹치는 시간 계산
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = overlap_end - overlap_start
            
            # 0보다 큰 경우에만 겹친다고 판단하고, 최대 겹침을 업데이트
            if overlap > 0 and overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = d_seg["speaker"]
                
        # 가장 많이 겹치는 화자 정보를 세그먼트에 담는다.
        # If max_overlap remains 0, assigned_speaker will still be -1.
        updated_segments.append({
            "idx": idx_seg,
            "text": t_seg["text"],
            "start": t_start,
            "end": t_end,
            "diar_label": assigned_speaker
        })
    
    return updated_segments

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


if __name__ == "__main__":


    prompt_text_improvement = """You are a helpful assistant to assign speaker information to diarization result.
There might be one Counsler and at one or more clients.
Read the text, and guess how many speakers are there.

Counselor tends to initiate the conversation, and guides the discussion, gives explanation, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance. Tends to interrupt.
Client tends to tell their experiences, express emotions. Client tends to speak continuously.

Given the data, assign speaker information to each segment.
Read the text, and assign the role of the speaker to each segment.

Data meaning is like this:
IDX: index of the segment
DIAR: diarization label. -1 means undecided.
TEXT: text of the segment. fragment of the text.

Data format is like this:
IDX(DIAR):TEXT

Output should be like this:
IDX: (same as input)
DIAR: (same as input, or guessed diar number)
SPEAKER: speaker number. 0 for Counselor, 1, 2, 3 ... for clients.
"""

    text = """
1(0): 아, 네, 카카오라고.
2(1): 오늘 막 주가가 올랐다고 막 그러던데.
3(1): 15년 2월 3일이고요.
4(1): 상담 받아보신 경험이 있으시네요?
5(0): 아, 네, 한 재작년 때.
6(0): 그때는 회사에서 잘 적응 못 해가지고 한 달에 몇 개 받았었어요.
7(0): 한 달에 한 번?
8(1): 어디서 받으셨어요?
9(0): 그 이름이 기억이 안 나는데 당산에 있는...
10(0): 어떤 당산?
11(1): 여기는 뭐... 아, 좀 빨리 시작해볼까도 생각했는데.
12(1): 누가 계세요?
13(1): 저희 어머니, 남동생이고요.
14(0): 같이 살고 있진 않아요.
15(1): 네, 알겠습니다.
16(1): 결혼하시려고 하는데 좀 그런 이슈 가지고 하시고.
17(1): 이 뒤에는 저희는 상담 비밀보장이 원칙이고요.
18(1): 그렇지만 자신이나 타인을 상의할 여지가 있다고 할 때는 제가 리포트해야 할 의무가 있어서.
19(1): 자신이나 타인을 상의한다는 의미는 자살의 의사가 있거나 타인을 해하거나 이럴 때는 그걸 리포트해야 된다는 얘기고.
20(1): 그리고 상담은 녹음을 전체로 해요.
21(2): 녹음을 하는 이유는 제가 놓치는 게 있을 수도 있고, 또 어떤 경우는 우리 수연 씨 이야기 이제 우리 다시 들어보면서 또 점검할 수도 있고 그래서.
22(2): 그런 내용으로 동의하시고 이제 하시면 될 것 같아요.
23(2): 여기에 사인해 주세요.
24(2): 그래서 오늘 이렇게 어렵게 시간 내셨는데, 그래도 뭐가 조금 달라지면 내가 여기 오길 잘했다 생각하실까요?
25(2): 결혼을 지금 약속을 하고 계신 거예요?
26(2): 네.
27(2): 아, 네.
28(2): 그 뭔가 제가 되게 4년 넘게 사귄 남자친구랑 결혼을 약속을 하고, 결혼 준비를 진행을 하는데, 뭔가 제가 확신이 계속 안 서가지고 계속 헤어지자고도 하고.
"""


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
                        "idx": {"type": "number"},
                        "diar": {"type": "number"},
                        "speaker": {"type": "number"}
                    },
                    "required": ["idx", "diar", "speaker"],
                    "additionalProperties": False
                }
            },
            "num_speakers": {"type": "number"}
        },
        "required": ["trans_segs_with_ts", "num_speakers"],
        "additionalProperties": False
    }

    elap_times = []
    print("asking1")
    time_start = time.time()
    res1 = ask_ai_with_format(messages, json_schema, model="gemma3:4b")
    time_end = time.time()
    elap_times.append(time_end - time_start)
    print(f"res1 {time_end - time_start}s", res1)

    print("asking2")
    time_start = time.time()
    res2 = ask_ai_with_format(messages, json_schema, model="gpt-4.1-mini")
    time_end = time.time()
    elap_times.append(time_end - time_start)
    print(f"res2 {time_end - time_start}s", res2)

    print("asking3")
    time_start = time.time()
    res3 = ask_ai_with_format(messages, json_schema, model="gpt-4o")
    time_end = time.time()
    elap_times.append(time_end - time_start)
    print(f"res3 {time_end - time_start}s", res3)

    print("asking4")
    time_start = time.time()
    res4 = ask_ai_with_format(messages, json_schema, model="gpt-4o-mini")
    time_end = time.time()
    elap_times.append(time_end - time_start)
    print(f"res4 {time_end - time_start}s", res4)


    print("elap_times")
    for idx, elap_time in enumerate(elap_times):
        print(f"elap_time {idx}: {elap_time}s")

    pass

# 응답을 사전처리해서 \uXXXX escape를 무효화 (예: \u26 → \\u26)
def safe_json_loads(s):
    try:
        return json.loads(s)
    except json.decoder.JSONDecodeError as e:
        print("🔴 JSONDecodeError 발생! 백업 로딩 시도.")
        print("에러 메시지:", str(e))
        # 이스케이프 문제 있을 경우 대응: \uXXXX를 임시로 무효화
        safe_s = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', s)
        return json.loads(safe_s)

def assign_speaker_roles(result):
    """
    화자별 역할 할당하는 함수
    
    Parameters:
        result (dict): 전사 결과 데이터
    
    Returns:
        dict: 역할이 할당된 전사 결과
    """

    client = get_openai_client()
    
    # 1. 화자별로 발화 내용 모으기 (대화 순서대로)
    conversation_text = []
    
    # 세그먼트 시간 기준으로 정렬
    sorted_segments = sorted(result["segments"], key=lambda x: x.get("start", 0))
    
    current_speaker = None
    current_text = []
    
    is_limit_text_when_infer = True
    max_speaker_occurencies = 7 ## set this 1000 if not want to limit.
    dict_speaker_occurencies = {}
    
    for segment in sorted_segments:
        if "speaker" in segment:
            speaker = segment["speaker"]
            text = segment["text"].strip()
            
            # 화자가 바뀌면 이전 텍스트 저장하고 새로 시작
            if current_speaker is not None and current_speaker != speaker and current_speaker != -1:
                if current_text:

                    count = dict_speaker_occurencies.get(current_speaker, 0)
                    if count < max_speaker_occurencies:
                        if is_limit_text_when_infer:
                            joined_text = ' '.join(current_text)
                            tok_text = joined_text.split()
                            # if len(joined_text) > 100: ## with len
                            #     joined_text = joined_text[:100] + '...'
                            if len(tok_text) > 15: ## with tok
                                joined_text = ' '.join(tok_text[:10]) + " ... " + ' '.join(tok_text[-3:])
                            conversation_text.append(f"[sid:{current_speaker}] {joined_text}")
                        else:
                            conversation_text.append(f"[sid:{current_speaker}] {' '.join(current_text)}")
                        dict_speaker_occurencies[current_speaker] = count + 1
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
                    "counsling_group_type": {
                        "type": "string",
                        "description": "Individual/Couple/Family/SupportGroup"
                    },
                    "counsling_about": {
                        "type": "string",
                        "description": "Topic of the counseling session"
                    },
                    "client_count": {
                        "type": "integer",
                        "description": "Number of clients excluding the counselor"
                    },
                    "speakers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sid": {
                                    "type": "integer",
                                    "description": "Speaker ID"
                                },
                                "role": {
                                    "type": "integer",
                                    "description": "Speaker role. 0 for counselor, 1, 2, 3 ... for different clients."
                                },
                                "role_detail": {
                                    "type": "string",
                                    "description": "Detailed description of speaker's role"
                                },
                                "role_nickname": {
                                    "type": "string",
                                    "description": "Nickname of speaker. Use real name if possible like '~~씨' or relation name like '남자친구','엄마'."
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence level of role assignment (0-1)"
                                }
                            },
                            "required": ["sid", "role", "role_detail", "role_nickname", "confidence"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["counsling_group_type", "counsling_about", "client_count", "speakers"],
                "additionalProperties": False
            }
        },
        "required": ["analysis"],
        "additionalProperties": False
    }

#     Counselor tends to initiate the conversation, and guides the discussion, gives explanation, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance. Tends to interrupt.

# Counselor tends to initiate the conversation, and guides the discussion, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance, Tends to interrupt politely.
# Client tends to express personal emotions and experiences. The client's language may be less structured and more emotionally charged.
# Counselor tends to initiate the conversation, ask open-ended questions, and guides the discussion, gives explanation, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance. Tends to interrupt.

# Counselor tends to ask open-ended questions, and guides the discussion, asks reflective, use empathetic and supportive language, open-ended questions; provides calm, supportive guidance. Tends to interrupt.
# Client tends to express personal emotions and experiences.
    
    # API 요청 메시지 구성
    messages = [
        {"role": "system", "content": "You are an expert in analyzing conversational data, especially in counseling sessions. Your task is to identify the roles of different speakers roles based on their speech patterns and content."},
        {"role": "user", "content": f"""
Please analyze the following conversation and identify the roles of each speaker.
Assume this is a counseling session. Determine how many clients there are and assign roles to each speaker ID.
Note that speech diarization might be inaccurate, so multiple speaker IDs might actually belong to the same person.
Number of client is undecided, but it should be 1 or more, and less than 5.
         
**Guidelines for Role Identification**

**Counselor**  
- Uses open‑ended questions to encourage elaboration (e.g., “How did that make you feel?”, “What was going through your mind at that moment?”)  
- Reflects and paraphrases the client’s statements to demonstrate understanding (e.g., “So, you’re saying that…”, “It sounds like…”).  
- Validates emotions with empathetic language (e.g., “I hear how difficult that was for you.”)  
- Guides gently without prescribing solutions, inviting the client to find their own answers.  
- Maintains a calm, supportive tone—steady pace and soft inflection.  
- Occasionally interrupts to clarify or summarize (e.g., “Let me pause you there—what I’m hearing is…”).  
- Periodically summarizes key points to keep the session on track (e.g., “To recap what you’ve shared so far…”).

**Client**  
- Expresses personal feelings, emotional states, and lived experiences (e.g., “I feel really anxious.” / “Last week, this happened…”).  
- Speaks in first‑person (“I feel…”, “I experienced…”).  
- Shares concrete examples or memories from their life.  
- May ask for advice, reassurance, or confirmation (e.g., “Is this normal?”, “What should I do?”).  
- Shows emotional shifts or distress signals (pauses, voice tremors, sighs).  
- At times hesitates or self‑edits, reflecting difficulty in expressing feelings.

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
        print(f"role infer response: {response_content}")
        # analysis = json.loads(response_content)
        analysis = safe_json_loads(response_content)

        
        # 3. 결과를 전사 데이터에 추가
        result["speaker_analysis"] = analysis["analysis"]
        
        # 화자 ID에 역할 매핑
        speaker_roles = {speaker["sid"]: speaker["role"] for speaker in analysis["analysis"]["speakers"]}
        ## 추가 코드
        speaker_roles.update({-1: -1})
        
        # 세그먼트와 단어에 역할 정보 추가
        for segment in result["segments"]:
            if "speaker" in segment:
                segment["speaker_role"] = speaker_roles.get(segment["speaker"], -1)
                
                # 단어 수준에서도 역할 정보 추가
                if "words" in segment:
                    for word in segment["words"]:
                        if "speaker" in word:
                            word["speaker_role"] = speaker_roles.get(word.get("speaker", -1), -1)
        
        print(f"Speaker role assignment completed. Found {analysis['analysis']['client_count']} clients.")
        print(f"Speaker roles: {speaker_roles}")
        
    except Exception as e:
        print(f"Error in speaker role assignment: {str(e)}")
    
    return result