## copied from maum_anal_from_transcript5.py
import pickle
import platform
import time
import matplotlib.pyplot as plt
import json
from typing import List, Dict
from collections import defaultdict

# --- Psycho timeline video visualization ---
from moviepy.editor import ImageSequenceClip
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from backend.logic.voice_analysis.ai_utils import ask_openai_with_format
from backend.logic.transcript_analysis.keyword_analysis import analyze_text_frequencies, format_word_frequencies, get_text_frequencies

def analyze_word_segments(data: Dict) -> List[Dict]:
    word_segments = data["word_segments"]

    # Step 1: cadence 계산 (WPM으로 변환: CPS * 60 / 2.5)
    cadences = []
    for seg in word_segments:
        duration = seg["duration"]
        text_len = len(seg["text"].strip())
        cps = text_len / duration if duration > 0 else 0
        wpm = cps * 60 / 2.5  # 평균 단어당 2.5자 기준
        seg["cadence"] = wpm  # cadence는 WPM으로 사용
        cadences.append((seg["speaker_role"], wpm))

    # Step 2: speaker_role 별 cadence 평균 계산
    cadence_by_role = defaultdict(list)
    for role, cadence in cadences:
        if role != -1:
            cadence_by_role[role].append(cadence)

    avg_cadence_by_role = {
        role: sum(vals) / len(vals) if vals else 1.0
        for role, vals in cadence_by_role.items()
    }

    # Step 3: cadence_prop 계산
    for seg in word_segments:
        role = seg["speaker_role"]
        avg_cad = avg_cadence_by_role.get(role, 1.0)
        seg["cadence_prop"] = seg["cadence"] / avg_cad if avg_cad else 1.0

    # Step 4: Pause 계산 (같은 speaker_role 이어지는 경우)
    pause_by_role = defaultdict(list)
    for i in range(1, len(word_segments)):
        prev = word_segments[i - 1]
        curr = word_segments[i]
        if prev["speaker_role"] == curr["speaker_role"] and curr["start"] > prev["end"]:
            pause = curr["start"] - prev["end"]
            pause_by_role[curr["speaker_role"]].append(pause)
            curr["pause"] = pause
        else:
            curr["pause"] = 0.0

    avg_pause_by_role = {
        role: sum(pauses) / len(pauses) if pauses else 1.0
        for role, pauses in pause_by_role.items()
    }

    # Step 5: pause_prop 계산
    for seg in word_segments:
        role = seg["speaker_role"]
        if "pause" not in seg:
            seg["pause"] = 0.0
        avg_pause = avg_pause_by_role.get(role, 1.0)
        seg["pause_prop"] = seg["pause"] / avg_pause if avg_pause and seg["pause"] > 0 else 0.0

    ## summary
    data['summary'] = {
        "avg_cadence": avg_cadence_by_role,
        "avg_pause": avg_pause_by_role
    }

    data["word_segments"] = word_segments

    

    # Step 6: duration change rate in consecutive_segments (per speaker_role)
    consecutive_segments = data.get("consecutive_segments", [])
    segments_by_role = defaultdict(list)
    for seg in consecutive_segments:
        segments_by_role[seg["speaker_role"]].append(seg)

    duration_change_rates_by_role = {}

    for role, segs in segments_by_role.items():
        change_rates = []
        for i in range(1, len(segs)):
            prev_duration = segs[i - 1]["duration"]
            curr_duration = segs[i]["duration"]
            if prev_duration > 0:
                change_rate = (curr_duration - prev_duration) / prev_duration
            else:
                change_rate = 0.0
            segs[i]["duration_change_rate"] = change_rate
            change_rates.append(change_rate)
        duration_change_rates_by_role[role] = sum(change_rates) / len(change_rates) if change_rates else 0.0

    # •	0.0 → 변화 없음 (curr = prev)
	# •	> 0.0 → 발화 길이가 길어졌음
	# •	< 0.0 → 발화 길이가 짧아졌음
	# •	1.0 → 두 배 길어짐
	# •	-0.5 → 절반으로 줄어듦

    data["consecutive_segments"] = [seg for role in segments_by_role for seg in segments_by_role[role]]
    data["summary"]["avg_duration_change_rate"] = duration_change_rates_by_role

    return data



def analyze_word_segments_post(data: Dict) -> List[Dict]:
    word_segments = data["word_segments"]
    consecutive_segments = data["consecutive_segments"]


    # Step 6: duration change rate in consecutive_segments (per speaker_role)
    # consecutive_segments = data.get("consecutive_segments", [])
    # segments_by_role = defaultdict(list)
    
    # Step 7: Calculate talking_times per role (including role -1) using word_segments
    talking_times_by_role = defaultdict(float)
    for seg in word_segments:
        role = seg["speaker_role"]
        duration = seg["duration"]
        talking_times_by_role[role] += duration
    
    data["summary"]["talking_times"] = dict(talking_times_by_role)

    # Step 8: Calculate avg_sentiment for all segments that have sentiment data
    pos_senti_values = []
    neg_senti_values = []
    
    for seg in consecutive_segments:
        if "sentiment" in seg:
            if "pos_senti" in seg["sentiment"]:
                pos_senti_values.append(seg["sentiment"]["pos_senti"])
            if "neg_senti" in seg["sentiment"]:
                neg_senti_values.append(seg["sentiment"]["neg_senti"])
    
    avg_sentiment = {
        "pos_senti": sum(pos_senti_values) / len(pos_senti_values) if pos_senti_values else 0.0,
        "neg_senti": sum(neg_senti_values) / len(neg_senti_values) if neg_senti_values else 0.0
    }
    
    data["summary"]["avg_sentiment"] = avg_sentiment

    # Step 9: Calculate avg_tense for all segments that have tense data
    fut_tense_values = []
    pas_tense_values = []
    
    for seg in consecutive_segments:
        if "tense" in seg:
            if "fut_tense" in seg["tense"]:
                fut_tense_values.append(seg["tense"]["fut_tense"])
            if "pas_tense" in seg["tense"]:
                pas_tense_values.append(seg["tense"]["pas_tense"])
    
    avg_tense = {
        "fut_tense": sum(fut_tense_values) / len(fut_tense_values) if fut_tense_values else 0.0,
        "pas_tense": sum(pas_tense_values) / len(pas_tense_values) if pas_tense_values else 0.0
    }
    
    data["summary"]["avg_tense"] = avg_tense

    texts = []
    for seg in consecutive_segments:
        texts.append(seg['text'])

    list_pos_tags = ['NNG', 'NNP', 'VV']
    frequencies = get_text_frequencies(texts, list_pos_tags)
    formatted_results = format_word_frequencies(frequencies, list_pos_tags, top_n=10)
    data['summary']['frequencies'] = formatted_results

    return data

def analyze_sentiment_and_tense(consecutive_segments):
    """
    Analyze sentiment and tense for each consecutive segment using GPT-4.1-mini.
    Process all segments at once in a single API call.
    Focus on intensity values rather than boolean flags.
    
    Args:
        consecutive_segments: List of consecutive speech segments with text content
        
    Returns:
        List of segments with added sentiment and tense analysis
    """
    # Prepare JSON schema for GPT response
    json_schema = {
        "type": "object",
        "properties": {
            "analyzed_segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "number", 
                        },
                        "pos_senti": {
                            "type": "number"
                        },
                        "neg_senti": {
                            "type": "number"
                        },
                        "fut_tense": {
                            "type": "number"
                        },
                        "pas_tense": {
                            "type": "number"
                        },
                        ## when debugging
                        # "senti_reason": {
                        #     "type": "string"
                        # },
                        # "tense_reason": {
                        #     "type": "string"
                        # }
                    },
                    # "required": ["id", "pos_senti", "neg_senti", "fut_tense", "pas_tense", "senti_reason", "tense_reason"],
                    "required": ["id", "pos_senti", "neg_senti", "fut_tense", "pas_tense"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["analyzed_segments"],
        "additionalProperties": False
    }
    
    # Format all segments with IDX prefix
    text = ""
    valid_segments = []
    
    for id, seg in enumerate(consecutive_segments):
        segment_text = seg.get("text", "").strip()
        if segment_text:
            text += f"ID {id}: {segment_text}\n"
            valid_segments.append(id)

        # if idx > 200:
        #     print("Test break ...")
        #     break
    
    if not text:
        # No valid segments to analyze
        print("No text to analyze in segments")
        return consecutive_segments
    
    # Create messages for GPT
    messages = [
        {
            "role": "system",
            "content": """Analyze each text segment. For each item below, rate on a scale from 0 to 1:

- pos_senti (sentiment): How positive is the sentiment expressed?
- neg_senti (sentiment): How negative is the sentiment expressed?
- fut_tense (tense): How much does the text focus on future events?
- pas_tense (tense): How much does the text focus on past events?

For sentiment, use the following hints for scoring, but score freely in between.
- 0.0 means no presence at all.
- 0.3 means slight presence (subtle or implied).
- 0.5 means little presence (noticeable but not strong).
- 0.7 means strong presence (clearly expressed).
- 1.0 means clear and dominant presence.

For tense, use the following hints for scoring, but score freely in between.
- 0.0 means no presence at all.
- 0.3 means little presence (noticeable but not strong).
- 0.7 means talking event about future or past.
- 1.0 means direct mention of future or past, or time related words.

Also, consider the tone used in the text when assessing sentiment and tense.
Multiple sentiments and tenses can appear simultaneously, and scores are independent.
Return the rows with any of the row values are not 0.
"""

# Add additional information. If no value is larger than 0.5, then put empty('') string.
# - senti_reason: reason why the sentiment is scored as such. In Korean.
# - tense_reason: reason why the tense is scored as such. In Korean.
        },
        {
            "role": "user",
            "content": text
        }
    ]
    
    try:
        # Call GPT with the structured response format
        start_time = time.time()
        print("Analyzing sentiment and tense... with data: ", text)
        analysis_result = ask_openai_with_format(messages, json_schema, model="gpt-4.1", temperature=0.2)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        
        # Create a copy of the original segments to preserve all data
        analyzed_segments = [seg.copy() for seg in consecutive_segments]

        print("analysis_result", analysis_result)
        
        # Add analysis results to the respective segments
        for result in analysis_result.get("analyzed_segments", []):
            id = result.get("id")
            if id is not None and 0 <= id < len(analyzed_segments):
                # Extract all intensity values directly from result
                pos_senti = result.get("pos_senti", 0)
                neg_senti = result.get("neg_senti", 0)
                fut_tense = result.get("fut_tense", 0)
                pas_tense = result.get("pas_tense", 0)
                
                # Create sentiment object with only intensity values
                analyzed_segments[id]["sentiment"] = {
                    "pos_senti": pos_senti,
                    "neg_senti": neg_senti
                }
                
                # Create tense object with only intensity values
                analyzed_segments[id]["tense"] = {
                    "fut_tense": fut_tense,
                    "pas_tense": pas_tense
                }
        
        print(f"Successfully analyzed {len(analysis_result.get('analyzed_segments', []))} segments")
        return analyzed_segments
        
    except Exception as e:
        print(f"Error analyzing segments: {e}")
        # Return original segments if analysis fails
        return consecutive_segments

def make_psycho_timeline_chart_data(consecutive_segments, word_segments):
    """
    Prepare data for the timeline chart.
    
    Args:
        consecutive_segments: List of consecutive speech segments
        word_segments: List of word-level segments with metrics
        
    Returns:
        Dictionary containing all data needed for rendering the chart
    """
    # Analyze sentiment and tense for all consecutive segments
    ## if do sentiment/tense analysis
    ## with reason, 70 sec. without reason, 30 sec.
    # print("Analyzing sentiment and tense...")
    # analyzed_segments = analyze_sentiment_and_tense(consecutive_segments)
    # ## save
    # with open("outputs/analyzed_segments.json", "w") as f:
    #     json.dump(analyzed_segments, f, indent=2, ensure_ascii=False)
    # consecutive_segments = analyzed_segments
    # print("Finished analyzing sentiment and tense.")
    
    # Calculate timeline range
    all_segments_ends = [s["end"] for s in consecutive_segments] + [s["end"] for s in word_segments]
    max_time = max(all_segments_ends) if all_segments_ends else 10
    
    # Identify actual speech segments from consecutive_segments
    speech_segments = []
    for seg in consecutive_segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        if end > start:
            speech_segments.append((start, end))
    
    # Merge overlapping segments
    if speech_segments:
        speech_segments.sort()
        merged_segments = [speech_segments[0]]
        for current in speech_segments[1:]:
            prev = merged_segments[-1]
            # If current segment overlaps with previous, merge them
            if current[0] <= prev[1]:
                merged_segments[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged_segments.append(current)
        speech_segments = merged_segments

    # Collect metric data points (including speaker roles >= 0)
    graph_data = {
        "cadence": [],
        "cadence_prop": [],
        "pause": [],
        "pause_prop": [],
        "positive_sentiment": [],
        "negative_sentiment": [],
        "future_tense": [],
        "past_tense": []
    }
    
    # Include all speakers with roles >= 0
    filtered_word_segments = [seg for seg in word_segments if seg.get("speaker_role", -1) >= 0]
    
    # Group data by speaker for separate metrics
    speaker_metrics = {}
    
    for seg in filtered_word_segments:
        start_time = seg.get("start", 0)
        cadence = seg.get("cadence", 0)
        cadence_prop = seg.get("cadence_prop", 0)
        pause = seg.get("pause", 0)
        pause_prop = seg.get("pause_prop", 0)
        speaker_role = seg.get("speaker_role", -1)
        
        # Add to combined metrics
        graph_data["cadence"].append((start_time, cadence))
        graph_data["cadence_prop"].append((start_time, cadence_prop))
        graph_data["pause"].append((start_time, pause))
        graph_data["pause_prop"].append((start_time, pause_prop))
    
        # Add to speaker-specific metrics
        if speaker_role not in speaker_metrics:
            speaker_metrics[speaker_role] = {
                "cadence": [],
                "cadence_prop": [],
                "pause": [],
                "pause_prop": [],
                "positive_sentiment": [],
                "negative_sentiment": [],
                "future_tense": [],
                "past_tense": []
            }
        
        speaker_metrics[speaker_role]["cadence"].append((start_time, cadence))
        speaker_metrics[speaker_role]["cadence_prop"].append((start_time, cadence_prop))
        speaker_metrics[speaker_role]["pause"].append((start_time, pause))
        speaker_metrics[speaker_role]["pause_prop"].append((start_time, pause_prop))
    
    # Add sentiment and tense data from consecutive segments
    for seg in consecutive_segments:
        if "sentiment" in seg and "tense" in seg:
            start_time = seg.get("start", 0)
            mid_time = (seg.get("start", 0) + seg.get("end", 0)) / 2  # Use middle of segment for plotting
            speaker_role = seg.get("speaker_role", -1)
            
            # Extract sentiment and tense values
            pos_senti = seg["sentiment"].get("pos_senti", 0)
            neg_senti = seg["sentiment"].get("neg_senti", 0)
            fut_tense = seg["tense"].get("fut_tense", 0)
            pas_tense = seg["tense"].get("pas_tense", 0)
            
            # Add all data points without threshold filtering
            graph_data["positive_sentiment"].append((mid_time, pos_senti))
            # Negate negative sentiment value for display
            graph_data["negative_sentiment"].append((mid_time, -neg_senti))
            graph_data["future_tense"].append((mid_time, fut_tense))
            # Negate past tense value for display
            graph_data["past_tense"].append((mid_time, -pas_tense))
            
            # Add to speaker-specific metrics if applicable
            if speaker_role >= 0 and speaker_role in speaker_metrics:
                speaker_metrics[speaker_role]["positive_sentiment"].append((mid_time, pos_senti))
                # Negate negative sentiment value for display
                speaker_metrics[speaker_role]["negative_sentiment"].append((mid_time, -neg_senti))
                speaker_metrics[speaker_role]["future_tense"].append((mid_time, fut_tense))
                # Negate past tense value for display
                speaker_metrics[speaker_role]["past_tense"].append((mid_time, -pas_tense))
    
    # Sort data by time
    for key in graph_data:
        graph_data[key].sort(key=lambda x: x[0])
    
    for speaker in speaker_metrics:
        for key in speaker_metrics[speaker]:
            speaker_metrics[speaker][key].sort(key=lambda x: x[0])
    
    # Calculate moving averages with proper handling of non-speech segments
    def calculate_speech_aware_ma(data_points, window_seconds=3.0, speech_segments=None):
        """
        Calculate moving average that respects speech segment boundaries.
        Points outside speech segments get zero values.
        
        Args:
            data_points: List of (time, value) tuples
            window_seconds: Size of the window in seconds
            speech_segments: List of (start_time, end_time) tuples representing speech segments
            
        Returns:
            List of (time, smoothed_value) tuples, with zeros in non-speech areas
        """
        if not data_points or len(data_points) < 2:
            return data_points
        
        # Sort points by time
        sorted_points = sorted(data_points, key=lambda x: x[0])
        result = []
        
        # For each point, calculate moving average
        for time_point, value in sorted_points:
            # First check if this point is within a speech segment
            in_speech = False
            if speech_segments:
                for start, end in speech_segments:
                    if start <= time_point <= end:
                        in_speech = True
                        break
            
            # If not in speech segment, set value to 0
            if not in_speech:
                result.append((time_point, 0))
                continue
            
            # Define time window
            window_start = time_point - window_seconds/2
            window_end = time_point + window_seconds/2
            
            # Find points in window that are also in speech segments
            window_values = []
            for t, v in sorted_points:
                if window_start <= t <= window_end:
                    # Check if this point is in a speech segment
                    point_in_speech = False
                    for start, end in speech_segments:
                        if start <= t <= end:
                            # Only include points from same speech segment
                            for start2, end2 in speech_segments:
                                if (start2 <= time_point <= end2) and (start2 <= t <= end2):
                                    point_in_speech = True
                                    break
                            break
                    
                    if point_in_speech:
                        window_values.append(v)
            
            # Calculate average if there are points in the window
            if window_values:
                avg_value = sum(window_values) / len(window_values)
                result.append((time_point, avg_value))
            else:
                # No points in window, use original value
                result.append((time_point, value))
        
        # Add additional points at speech segment boundaries with zero values
        # This ensures lines don't connect across gaps
        boundary_points = []
        for start, end in speech_segments:
            # Add points just before start and just after end with value 0
            if start > 0:
                boundary_points.append((start - 0.001, 0))
            boundary_points.append((start, 0))  # Start point
            boundary_points.append((end, 0))  # End point
            if end < max_time:
                boundary_points.append((end + 0.001, 0))
        
        # Add these boundary points to the result
        result.extend(boundary_points)
        
        # Sort final result by time
        result.sort(key=lambda x: x[0])
        
        return result
    
    # Apply speech-aware moving average to all metrics
    moving_avg_data = {}
    for key in graph_data:
        moving_avg_data[f"{key}_ma"] = calculate_speech_aware_ma(
            graph_data[key], 
            window_seconds=3.0, 
            speech_segments=speech_segments
        )
    
    # Apply to speaker-specific metrics
    speaker_ma_metrics = {}
    for speaker in speaker_metrics:
        # Get speech segments for this speaker only
        speaker_speech_segments = []
        for seg in consecutive_segments:
            if seg.get("speaker_role", -1) == speaker:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                if end > start:
                    speaker_speech_segments.append((start, end))
        
        # Merge overlapping segments for this speaker
        if speaker_speech_segments:
            speaker_speech_segments.sort()
            merged = [speaker_speech_segments[0]]
            for current in speaker_speech_segments[1:]:
                prev = merged[-1]
                if current[0] <= prev[1]:
                    merged[-1] = (prev[0], max(prev[1], current[1]))
                else:
                    merged.append(current)
            speaker_speech_segments = merged
        
        # Calculate moving averages for this speaker
        speaker_ma_metrics[speaker] = {}
        for key in speaker_metrics[speaker]:
            speaker_ma_metrics[speaker][f"{key}_ma"] = calculate_speech_aware_ma(
                speaker_metrics[speaker][key],
                window_seconds=3.0,
                speech_segments=speaker_speech_segments
            )
    
    # Find min and max values for scaling with tighter buffer
    graph_ranges = {}
    for key in graph_data:
        if graph_data[key]:
            values = [dp[1] for dp in graph_data[key]]
            min_val = min(values)
            max_val = max(values)
            # Reduce buffer from 10% to 3% for tighter ranges
            buffer = (max_val - min_val) * 0.03 if max_val > min_val else 0.03
            graph_ranges[key] = (min_val - buffer, max_val + buffer)
        else:
            graph_ranges[key] = (0, 1)
    
    # Group segments by speaker (including roles >= 0)
    speakers = set(seg.get("speaker_role", -1) for seg in consecutive_segments)
    speakers = sorted([s for s in speakers if s >= 0])
    
    # Prepare speaker timeline data
    speaker_segments = {}
    for speaker in speakers:
        speaker_segments[speaker] = []
        for seg in consecutive_segments:
            if seg.get("speaker_role", -1) == speaker:
                speaker_segments[speaker].append(seg)
    
    # Return the prepared data
    return {
        "max_time": max_time,
        "graph_data": graph_data,
        "moving_avg_data": moving_avg_data,
        "graph_ranges": graph_ranges,
        "speakers": speakers,
        "speaker_segments": speaker_segments,
        "speaker_metrics": speaker_metrics,
        "speaker_ma_metrics": speaker_ma_metrics,
        "speech_segments": speech_segments,
        "seconds": int(max_time) + 1,
        "consecutive_segments": consecutive_segments  # Include the segments with sentiment/tense data
    }

def render_psycho_timeline_chart(chart_data, output_path="outputs/psycho_timeline.png"):
    """
    Render the timeline chart showing speaker activity and metrics.
    
    Args:
        chart_data: Dictionary containing data prepared by make_psycho_timeline_chart_data
        output_path: Path to save the output chart image
    """
    # Extract data from chart_data
    max_time = chart_data["max_time"]
    graph_data = chart_data["graph_data"]  # Raw data
    moving_avg_data = chart_data["moving_avg_data"]  # Moving average data (used for cadence only)
    graph_ranges = chart_data["graph_ranges"]
    speakers = chart_data["speakers"]
    speaker_segments = chart_data["speaker_segments"]
    speaker_metrics = chart_data["speaker_metrics"]  # Raw speaker-specific metrics
    speaker_ma_metrics = chart_data["speaker_ma_metrics"]  # Moving average speaker metrics (used for cadence only)
    speech_segments = chart_data["speech_segments"]
    seconds = chart_data["seconds"]
    
    # Set up the figure with subplots in the order: cadence -> pause -> sentiment -> tense -> speaker timelines
    num_speaker_rows = len(speakers)
    total_rows = 4 + num_speaker_rows  # Cadence + Pause + Sentiment + Tense + speaker timelines
    
    fig_height = 10 + (num_speaker_rows * 0.8)  # Adjust height for the rows
    fig = plt.figure(figsize=(12, fig_height))
    
    # Define height ratios - main metrics get more space
    height_ratios = [1.5, 1.5, 1.5, 1.5]  # Cadence, Pause, Sentiment, Tense
    height_ratios += [0.8] * num_speaker_rows  # Speaker timeline rows
    
    gs = GridSpec(total_rows, 1, height_ratios=height_ratios)
    
    # Define a color palette for different speakers
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    import numpy as np
    
    # Use tab10 for up to 10 speakers
    speaker_colors = cm.get_cmap('tab10', max(10, len(speakers)))
    
    # ---- CADENCE ROW ----
    ax_cadence = fig.add_subplot(gs[0])
    ax_cadence.set_xlim(0, max_time)
    
    # Get cadence values for better range calculation
    cadence_values = [v for _, v in moving_avg_data["cadence_ma"]]
    if cadence_values:
        # Find a more reasonable max value by excluding outliers
        # Use 90th percentile instead of maximum to make it tighter
        p90 = np.percentile(cadence_values, 90)
        min_val = max(0, min(cadence_values))  # Min won't go below 0
        
        # If there are values above p90, use p90 * 1.2 as max
        # This makes the y-axis tighter but still shows the important range
        if p90 * 1.2 < max(cadence_values):
            max_val = p90 * 1.2
        else:
            # If no significant outliers, use the original max plus small buffer
            max_val = max(cadence_values) * 1.05
    else:
        min_val, max_val = 0, 200  # Default range for WPM
    
    ax_cadence.set_ylim(min_val, max_val)
    ax_cadence.set_title("Speaker Timeline Analysis")
    ax_cadence.set_ylabel("Cadence (WPM)", fontweight='bold')
    
    # Only use vertical grid lines
    ax_cadence.grid(False)  # Remove all grid lines first
    ax_cadence.grid(True, axis='x', linestyle='--', alpha=0.7)  # Only add vertical grid lines
    
    # Plot combined cadence with moving average (thin line)
    ma_times = [t for t, _ in moving_avg_data["cadence_ma"]]
    ma_values = [v for _, v in moving_avg_data["cadence_ma"]]
    ax_cadence.plot(ma_times, ma_values, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Combined')
    
    # Plot each speaker's cadence
    for i, speaker in enumerate(speakers):
        if speaker in speaker_ma_metrics and speaker_ma_metrics[speaker]["cadence_ma"]:
            sp_times = [t for t, _ in speaker_ma_metrics[speaker]["cadence_ma"]]
            sp_values = [v for _, v in speaker_ma_metrics[speaker]["cadence_ma"]]
            line_color = speaker_colors(i % 10)
            ax_cadence.plot(sp_times, sp_values, color=line_color, linewidth=2, label=f'R{speaker}')
    
    ax_cadence.legend(loc='upper right')
    
    # ---- PAUSE ROW (HORIZONTAL LINES FOR LONG PAUSES) ----
    ax_pause = fig.add_subplot(gs[1])
    ax_pause.set_xlim(0, max_time)
    
    # Collect long pauses (>3s) information from consecutive segments ONLY
    # Use raw data, no moving averages
    long_pauses = []
    
    # Find all significant pauses per speaker by directly examining speech segment gaps
    for speaker in speakers:
        # Get segments for this speaker in order
        if speaker not in speaker_segments:
            continue
            
        segments = speaker_segments[speaker]
        segments = sorted(segments, key=lambda s: s.get("start", 0))
        
        # Create a timeline of all speaker activity to check for other speakers talking during gaps
        all_speaker_times = []
        for sp in speakers:
            if sp != speaker and sp in speaker_segments:
                for seg in speaker_segments[sp]:
                    all_speaker_times.append((seg.get("start", 0), seg.get("end", 0), sp))
        
        # Look for gaps between consecutive segments
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get("end", 0)
            curr_start = segments[i].get("start", 0)
            pause_duration = curr_start - prev_end
            
            # Check if this is a significant pause (> 3 seconds)
            if pause_duration > 3.0:
                # Check if other speakers are talking during this gap
                other_speaker_active = False
                for start, end, sp in all_speaker_times:
                    # If another speaker's segment overlaps with this gap
                    if max(start, prev_end) < min(end, curr_start):
                        other_speaker_active = True
                        break
                
                # Only count as pause if no other speakers are active during this time
                if not other_speaker_active:
                    # Store the pause with speaker, start time, end time
                    long_pauses.append({
                        "speaker": speaker,
                        "start": prev_end,
                        "end": curr_start,
                        "duration": pause_duration
                    })
    
    # Set up Y-axis for the pause plot to show speaker roles
    # Calculate a nice spacing for speaker roles on y-axis
    max_speaker = max(speakers) if speakers else 0
    min_speaker = min(speakers) if speakers else 0
    # Add buffer to top and bottom
    y_padding = 0.5
    ax_pause.set_ylim(min_speaker - y_padding, max_speaker + y_padding)
    
    # Set y-ticks at speaker role positions
    ax_pause.set_yticks(speakers)
    ax_pause.set_yticklabels([f"R{s}" for s in speakers])
    
    # Set thinner tick marks and lighter axis line for y-axis
    ax_pause.tick_params(axis='y', width=0.5, length=5)
    for spine in ax_pause.spines.values():
        spine.set_linewidth(0.5)
        
    ax_pause.set_ylabel("Silent Pauses (>3s) by Speaker", fontweight='bold')
    
    # Remove grid lines
    ax_pause.grid(False)
    
    # Add only vertical grid lines
    ax_pause.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Draw horizontal lines for each long pause
    for pause in long_pauses:
        speaker = pause["speaker"]
        start_time = pause["start"]
        end_time = pause["end"]
        duration = pause["duration"]
        
        # Get the index of this speaker in our list
        try:
            speaker_idx = speakers.index(speaker)
            color = speaker_colors(speaker_idx % 10)
        except:
            # Just use a default color if speaker not found
            color = 'gray'
        
        # Draw a horizontal line from start to end at speaker's y-position
        ax_pause.plot(
            [start_time, end_time], 
            [speaker, speaker], 
            color=color, 
            linewidth=2 + min(duration, 5) * 0.4,  # Thicker line for longer pauses
            alpha=0.8,
            solid_capstyle='round'
        )
        
        # Add duration text above the line for longer pauses
        if duration > 3.0:  # Only show text for all pauses (changed from 2.0)
            ax_pause.text(
                (start_time + end_time) / 2, 
                speaker + 0.15,  # y position: slightly above the line
                f"{duration:.1f}s",
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=8,
                color=color,
                fontweight='bold'
            )
    
    # Legend with explanation
    if long_pauses:
        from matplotlib.lines import Line2D
        legend_elements = []
        for speaker in speakers:
            try:
                speaker_idx = speakers.index(speaker)
                color = speaker_colors(speaker_idx % 10)
                legend_elements.append(
                    Line2D([0], [0], color=color, lw=2, label=f'R{speaker} pauses')
                )
            except:
                continue
                
        if legend_elements:
            ax_pause.legend(handles=legend_elements, loc='upper right')
    
    # Add tick marks for each second on x-axis
    ax_pause.set_xticks(range(seconds + 1))
    ax_pause.set_xticklabels([f"{i}s" for i in range(seconds + 1)])
    
    # ---- SENTIMENT ROW (Combined for all speakers) ----
    ax_sentiment = fig.add_subplot(gs[2])
    ax_sentiment.set_xlim(0, max_time)
    
    # Adjust y-axis limits for sentiment - showing from -1 to 1
    ax_sentiment.set_ylim(-1.05, 1.05)  
    ax_sentiment.set_ylabel("Sentiment Intensity\n(+pos/-neg)", fontweight='bold')
    
    # Add a horizontal line at y=0
    ax_sentiment.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Remove all grid lines first, then add only vertical ones
    ax_sentiment.grid(False)
    ax_sentiment.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Plot each speaker's sentiment with different line styles/colors
    for i, speaker in enumerate(speakers):
        # Use a distinct color for each speaker
        speaker_color = speaker_colors(i % 10)
        
        # Plot positive sentiment for this speaker
        if speaker in speaker_ma_metrics and "positive_sentiment_ma" in speaker_ma_metrics[speaker]:
            sp_pos_times = [t for t, _ in speaker_ma_metrics[speaker]["positive_sentiment_ma"]]
            sp_pos_values = [v for _, v in speaker_ma_metrics[speaker]["positive_sentiment_ma"]]
            if sp_pos_values:
                ax_sentiment.plot(sp_pos_times, sp_pos_values, color=speaker_color, 
                                linewidth=2, linestyle='-', alpha=0.7,
                                label=f'R{speaker} Positive')
        
        # Plot negative sentiment for this speaker
        if speaker in speaker_ma_metrics and "negative_sentiment_ma" in speaker_ma_metrics[speaker]:
            sp_neg_times = [t for t, _ in speaker_ma_metrics[speaker]["negative_sentiment_ma"]]
            sp_neg_values = [v for _, v in speaker_ma_metrics[speaker]["negative_sentiment_ma"]]
            if sp_neg_values:
                ax_sentiment.plot(sp_neg_times, sp_neg_values, color=speaker_color, 
                                linewidth=2, linestyle='--', alpha=0.7,
                                label=f'R{speaker} Negative')
    
    ax_sentiment.legend(loc='upper right')
    
    # ---- TENSE ROW (Combined for all speakers) ----
    ax_tense = fig.add_subplot(gs[3])
    ax_tense.set_xlim(0, max_time)
    
    # Adjust y-axis limits for tense - showing from -1 to 1
    ax_tense.set_ylim(-1.05, 1.05)  
    ax_tense.set_ylabel("Tense Intensity\n(+fut/-past)", fontweight='bold')
    
    # Add a horizontal line at y=0
    ax_tense.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Remove all grid lines first, then add only vertical ones
    ax_tense.grid(False)
    ax_tense.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # # Debug: Print available tense data for each speaker
    # print("\nAvailable tense data by speaker:")
    # for speaker in speakers:
    #     if speaker in speaker_ma_metrics:
    #         has_future = "future_tense_ma" in speaker_ma_metrics[speaker] and speaker_ma_metrics[speaker]["future_tense_ma"]
    #         has_past = "past_tense_ma" in speaker_ma_metrics[speaker] and speaker_ma_metrics[speaker]["past_tense_ma"]
    #         print(f"Speaker R{speaker}: Future data: {has_future}, Past data: {has_past}")
    
    # Plot each speaker's tense with different line styles/colors
    for i, speaker in enumerate(speakers):
        # Use the same speaker-specific color for consistency
        speaker_color = speaker_colors(i % 10)
        
        # Plot future tense for this speaker
        if speaker in speaker_ma_metrics and "future_tense_ma" in speaker_ma_metrics[speaker]:
            sp_future_times = [t for t, _ in speaker_ma_metrics[speaker]["future_tense_ma"]]
            sp_future_values = [v for _, v in speaker_ma_metrics[speaker]["future_tense_ma"]]
            if sp_future_values:
                ax_tense.plot(sp_future_times, sp_future_values, color=speaker_color, 
                            linewidth=2, linestyle='-', alpha=0.7,
                            label=f'R{speaker} Future')
        
        # Plot past tense for this speaker
        if speaker in speaker_ma_metrics and "past_tense_ma" in speaker_ma_metrics[speaker]:
            sp_past_times = [t for t, _ in speaker_ma_metrics[speaker]["past_tense_ma"]]
            sp_past_values = [v for _, v in speaker_ma_metrics[speaker]["past_tense_ma"]]
            if sp_past_values:
                ax_tense.plot(sp_past_times, sp_past_values, color=speaker_color, 
                            linewidth=2, linestyle='--', alpha=0.7,
                            label=f'R{speaker} Past')
    
    ax_tense.legend(loc='upper right')
    
    # ---- SPEAKER TIMELINES ----
    for i, speaker in enumerate(speakers):
        timeline_row_idx = 4 + i
        ax_speaker = fig.add_subplot(gs[timeline_row_idx])
        ax_speaker.set_xlim(0, max_time)
        ax_speaker.set_ylim(0, 1)
        
        # Use the same color for speaker timeline as used in the metric plots
        speaker_color = speaker_colors(i % 10)
        
        # Create a label with colored text
        from matplotlib.text import Text
        speaker_label = f"R{speaker}"
        ax_speaker.set_ylabel(speaker_label, color=speaker_color, fontweight='bold')
        
        # Draw segments for this speaker
        for seg in speaker_segments[speaker]:
            start_time = seg.get("start", 0)
            end_time = seg.get("end", 0)
            
            # Determine fill color based on sentiment if available
            facecolor = speaker_color
            alpha = 0.7
            
            # If segment has sentiment data, modify the color or add a marker
            if "sentiment" in seg:
                pos_senti = seg["sentiment"].get("pos_senti", 0)
                neg_senti = seg["sentiment"].get("neg_senti", 0)
                
                # Add sentiment indicators to the segment as small markers at the top edge
                mid_x = (start_time + end_time) / 2
                
                # Use intensity to determine marker size
                if pos_senti > 0.3:
                    marker_size = 30 * pos_senti
                    ax_speaker.scatter([mid_x], [0.8], color='green', s=marker_size, 
                                     marker='^', alpha=0.8, zorder=5)
                
                if neg_senti > 0.3:
                    marker_size = 30 * neg_senti
                    ax_speaker.scatter([mid_x], [0.8], color='red', s=marker_size, 
                                     marker='v', alpha=0.8, zorder=5)
            
            # Add a colored segment rectangle
            rect = patches.Rectangle(
                (start_time, 0.1), 
                end_time - start_time, 
                0.8, 
                linewidth=1, 
                edgecolor='black', 
                facecolor=facecolor, 
                alpha=alpha
            )
            ax_speaker.add_patch(rect)
            
            # Add duration text if segment is wide enough
            if end_time - start_time > 1.0:
                duration = seg.get("duration", end_time - start_time)
                ax_speaker.text(
                    (start_time + end_time) / 2, 
                    0.5, 
                    f"{duration:.1f}s", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white',
                    fontsize=8,
                    weight='bold'
                )
        
        # Hide y-ticks and only show x-axis labels on bottom plot
        ax_speaker.set_yticks([])
        if i < len(speakers) - 1:
            ax_speaker.set_xticklabels([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.show()
    ## after close from user, save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    os.makedirs("outputs", exist_ok=True)
    ## save figure as pickle
    with open("outputs/chart_data.pkl", "wb") as f:
        pickle.dump(fig, f)
    plt.close()
    
    print(f"Timeline chart saved to {output_path}")



# 사용 예시
if __name__ == "__main__":

    # is_linux = os.name == 'posix'
    is_linux = os.name == 'posix' and platform.system() != 'Darwin'
    print(f"is_linux: {is_linux}")

    # uuid_str= "e4978649-e7d3-4f5a-86f7-b111a2c71d52" #1min cut
    uuid_str= "c5a1d3a9-3cac-4d63-8bba-efdf5f979379" #35min all
    prefix_path = '/home/gq/workspace/simri/mavo_dev/uploads/'
    if is_linux:
        pass
    else:
        prefix_path = '/Users/beaver.baek/dev/simri/data/exp_data/data_250425_sentiment/uploads'
    
    path_sample = f'{prefix_path}/{uuid_str}/id[{uuid_str}]_consecutive_segments.json'
    with open(path_sample, "r", encoding="utf-8") as f:
        diarization_data = json.load(f)

    
    diarization_data = analyze_word_segments(diarization_data)
    
    ## display

    # 결과 저장 또는 출력
    with open("diarization_data.json", "w", encoding="utf-8") as f:
        json.dump(diarization_data, f, indent=2, ensure_ascii=False)

    # 심리 상태 시간 추이 비디오 생성
    
    # Extract UUID from the path to find corresponding audio file
    import re
    uuid_match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', path_sample)
    if uuid_match:
        uuid = uuid_match.group(1)
        audio_path = f'{prefix_path}/{uuid}/id[{uuid}]_merged.m4a'
        if os.path.exists(audio_path):
            print(f"Found audio file: {audio_path}")
        else:
            print(f"Audio file not found: {audio_path}")
            audio_path = None
    else:
        audio_path = None
        print("Could not extract UUID from path")
    
    # Get consecutive_segments and word_segments from diarization_data
    consecutive_segments = diarization_data.get("consecutive_segments", [])
    word_segments = diarization_data.get("word_segments", [])
    
    print(f"Processed {len(consecutive_segments)} consecutive segments and {len(word_segments)} word segments")
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    chart_data = make_psycho_timeline_chart_data(consecutive_segments, word_segments)

    with open("outputs/chart_data.json", "w", encoding="utf-8") as f:
        json.dump(chart_data, f, indent=2, ensure_ascii=False)

    render_psycho_timeline_chart(chart_data)
