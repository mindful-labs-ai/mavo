import json
import os
import re
from difflib import SequenceMatcher

## from 1_6

#####
# USER DIRECTION
# i've improved stt result using LLM.
# im adding timestamp info to my improved lines. the improved lines are in 'temp/tmp_improved_lines.json'
# refer the data and source code. implement further.
# keep in mind that the improved text is similar but not identical to transcribed words.
# for important steps, save as result temp/ with proper name, in json (indented with 2, ensure_ascii=False)
# the language of transcription word is korean and english mainly.
# if comparing words, please ignore white spaces and all special characters. that might improve the performance.
#####

def normalize_text(txt: str) -> str:
    """
    Lowercase, remove spaces/punctuation, keep only Korean (가-힣) + English letters + digits.
    """
    txt = txt.lower()
    txt = re.sub(r'[^0-9a-zA-Z가-힣]', '', txt)
    return txt

def match_ratio(a: str, b: str) -> float:
    """
    Returns similarity ratio in [0..1], using difflib.
    """
    return SequenceMatcher(None, a, b).ratio()

def get_best_chunk(normalized_line, transcription_words, start_index, MAX_CHUNK_SIZE=30, MATCH_THRESHOLD=0.6):
    """
    Same chunk-based matching from earlier, except increased MAX_CHUNK_SIZE to handle longer lines.
    We try chunk sizes from 1..MAX_CHUNK_SIZE, then pick the best match ratio above MATCH_THRESHOLD.
    """
    n = len(transcription_words)
    best_indices = []
    best_ratio = 0.0

    for chunk_size in range(1, MAX_CHUNK_SIZE + 1):
        if start_index + chunk_size > n:
            break
        chunk = transcription_words[start_index : start_index + chunk_size]
        chunk_joined = ''.join(w['norm_text'] for w in chunk)
        ratio = match_ratio(normalized_line, chunk_joined)
        if ratio > best_ratio:
            best_ratio = ratio
            best_indices = list(range(start_index, start_index + chunk_size))

    if best_ratio >= MATCH_THRESHOLD:
        return best_indices, best_ratio
    else:
        return [], 0.0
    

def get_improved_lines_with_ts(improved_lines, transcription_words):
    """
    Creates a list of objects:
      {
        "text": <the original improved line>,
        "start": <earliest matched word start time>,
        "end": <latest matched word end time>,
        "idx_words": [...],
        "words": [...]
      }

    We do a first-pass chunk-based matching approach with a large MAX_CHUNK_SIZE 
    for lines that have many words.
    """
    # Pre-normalize transcription words
    for w in transcription_words:
        w['norm_text'] = normalize_text(w['text'])
        w['used'] = False

    lines_with_ts = []
    next_transcript_idx = 0

    for line in improved_lines:
        norm_line = normalize_text(line)
        matched_indices, best_ratio = get_best_chunk(
            normalized_line=norm_line,
            transcription_words=transcription_words,
            start_index=next_transcript_idx,
            MAX_CHUNK_SIZE=30,
            MATCH_THRESHOLD=0.6
        )

        if matched_indices:
            for mi in matched_indices:
                transcription_words[mi]['used'] = True

            min_start = min(transcription_words[mi]['start'] for mi in matched_indices)
            max_end = max(transcription_words[mi]['end'] for mi in matched_indices)
            matched_words = [transcription_words[mi]['text'] for mi in matched_indices]

            line_info = {
                "text": line,
                "start": float(min_start),
                "end": float(max_end),
                "idx_words": matched_indices,
                "words": matched_words
            }

            next_transcript_idx = matched_indices[-1] + 1
        else:
            line_info = {
                "text": line,
                "start": None,
                "end": None,
                "idx_words": [],
                "words": []
            }

        lines_with_ts.append(line_info)

    #TODO: move some words to the next line if that is more plausible?
    postprocess_boundaries(lines_with_ts, transcription_words)

    # 누락된 타임스탬프 보간하기
    interpolate_missing_timestamps(lines_with_ts)
    
    return lines_with_ts


def postprocess_boundaries(lines_with_ts, transcription_words):
    """
    Minimal boundary-correction pass:
    - For each pair of consecutive lines (i, i+1):
      1) Attempt moving the last word of line i to the beginning of line i+1.
      2) Attempt moving the first word of line i+1 to the end of line i.

    In each case, if the combined match ratio for lines i and i+1 is improved, 
    we keep that single boundary move; otherwise, we revert. 
    This helps fix minor boundary misalignments (e.g. trailing words that belong to the next line).
    """

    def combined_match_ratio(line_info, line_text):
        """
        Returns a match ratio comparing:
          normalized(line_text) vs. concatenation of normalized words from line_info['words'].
        """
        norm_line = normalize_text(line_text)
        word_joined = ''.join(normalize_text(w) for w in line_info['words'])
        return match_ratio(norm_line, word_joined)

    def update_times_for_line(line_info):
        """
        If line_info has words, set 'start' to the earliest word's start, 
        and 'end' to the latest word's end.
        Otherwise set both None.
        """
        if line_info['idx_words']:
            # earliest word's start
            first_idx = line_info['idx_words'][0]
            line_info['start'] = float(transcription_words[first_idx]['start'])
            # latest word's end
            last_idx = line_info['idx_words'][-1]
            line_info['end'] = float(transcription_words[last_idx]['end'])
        else:
            line_info['start'] = None
            line_info['end'] = None

    for i in range(len(lines_with_ts) - 1):
        line_i = lines_with_ts[i]
        line_i1 = lines_with_ts[i+1]

        # If either line has no words, skip boundary checks
        if not line_i['words'] or not line_i1['words']:
            continue

        # ----- 1) Try moving last word of line i -> front of line i+1 -----
        old_ratio_i = combined_match_ratio(line_i, line_i['text'])
        old_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        old_sum = old_ratio_i + old_ratio_i1

        last_word_idx = line_i['idx_words'][-1]
        last_word = line_i['words'][-1]

        # Remove from line i
        line_i['idx_words'] = line_i['idx_words'][:-1]
        line_i['words'] = line_i['words'][:-1]

        # Add to the front of line i+1
        line_i1['idx_words'] = [last_word_idx] + line_i1['idx_words']
        line_i1['words'] = [last_word] + line_i1['words']

        # Update times
        update_times_for_line(line_i)
        update_times_for_line(line_i1)

        # Check new ratio
        new_ratio_i = combined_match_ratio(line_i, line_i['text'])
        new_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        new_sum = new_ratio_i + new_ratio_i1

        if new_sum <= old_sum:
            # revert
            line_i1['idx_words'].pop(0)
            line_i1['words'].pop(0)
            line_i['idx_words'].append(last_word_idx)
            line_i['words'].append(last_word)

            # revert times
            update_times_for_line(line_i)
            update_times_for_line(line_i1)
        else:
            # keep the change
            continue

        # ----- 2) Try moving first word of line i+1 -> end of line i -----
        if not line_i1['words']:
            continue

        old_ratio_i = combined_match_ratio(line_i, line_i['text'])
        old_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        old_sum = old_ratio_i + old_ratio_i1

        first_word_idx = line_i1['idx_words'][0]
        first_word = line_i1['words'][0]

        # Remove from line i+1
        line_i1['idx_words'] = line_i1['idx_words'][1:]
        line_i1['words'] = line_i1['words'][1:]

        # Add to line i
        line_i['idx_words'].append(first_word_idx)
        line_i['words'].append(first_word)

        # Update times
        update_times_for_line(line_i)
        update_times_for_line(line_i1)

        # Check new ratio
        new_ratio_i = combined_match_ratio(line_i, line_i['text'])
        new_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        new_sum = new_ratio_i + new_ratio_i1

        if new_sum <= old_sum:
            # revert
            line_i['idx_words'].pop()
            line_i['words'].pop()
            line_i1['idx_words'] = [first_word_idx] + line_i1['idx_words']
            line_i1['words'] = [first_word] + line_i1['words']

            # revert times
            update_times_for_line(line_i)
            update_times_for_line(line_i1)

def interpolate_missing_timestamps(lines_with_ts):
    """
    누락된 start 또는 end 타임스탬프를 보간하는 함수.
    보간된 경우 itp_start: True 또는 itp_end: True 필드를 추가합니다.
    """
    # 먼저 유효한 타임스탬프가 있는 라인들의 인덱스를 찾습니다
    valid_indices = []
    for i, line in enumerate(lines_with_ts):
        if line["start"] is not None and line["end"] is not None:
            valid_indices.append(i)
    
    if not valid_indices:
        return  # 유효한 타임스탬프가 없으면 보간할 수 없음
    
    for i, line in enumerate(lines_with_ts):
        # start가 없는 경우
        if line["start"] is None:
            # 이전/다음 유효한 라인 찾기
            prev_valid = None
            next_valid = None
            
            for idx in valid_indices:
                if idx < i:
                    prev_valid = idx
                elif idx > i:
                    next_valid = idx
                    break
            
            # 보간 수행
            if prev_valid is not None and next_valid is not None:
                # 이전과 다음 사이를 선형 보간
                prev_end = lines_with_ts[prev_valid]["end"]
                next_start = lines_with_ts[next_valid]["start"]
                ratio = (i - prev_valid) / (next_valid - prev_valid)
                line["start"] = prev_end + ratio * (next_start - prev_end)
                line["itp_start"] = True
            elif prev_valid is not None:
                # 이전의 end를 사용
                line["start"] = lines_with_ts[prev_valid]["end"]
                line["itp_start"] = True
            elif next_valid is not None:
                # 다음의 start를 사용
                line["start"] = lines_with_ts[next_valid]["start"]
                line["itp_start"] = True
        
        # end가 없는 경우
        if line["end"] is None:
            # 이전/다음 유효한 라인 찾기
            prev_valid = None
            next_valid = None
            
            for idx in valid_indices:
                if idx < i:
                    prev_valid = idx
                elif idx > i:
                    next_valid = idx
                    break
            
            # 보간 수행
            if prev_valid is not None and next_valid is not None:
                # 이전과 다음 사이를 선형 보간
                prev_end = lines_with_ts[prev_valid]["end"]
                next_start = lines_with_ts[next_valid]["start"]
                ratio = (i - prev_valid) / (next_valid - prev_valid)
                line["end"] = prev_end + ratio * (next_start - prev_end)
                line["itp_end"] = True
            elif prev_valid is not None:
                # 이전의 end를 사용
                line["end"] = lines_with_ts[prev_valid]["end"]
                line["itp_end"] = True
            elif next_valid is not None:
                # 다음의 start를 사용
                line["end"] = lines_with_ts[next_valid]["start"]
                line["itp_end"] = True

def postprocess_boundaries(lines_with_ts, transcription_words):
    """
    Minimal boundary-correction pass:
    - For each pair of consecutive lines (i, i+1):
      1) Attempt moving the last word of line i to the beginning of line i+1.
      2) Attempt moving the first word of line i+1 to the end of line i.

    In each case, if the combined match ratio for lines i and i+1 is improved, 
    we keep that single boundary move; otherwise, we revert. 
    This helps fix minor boundary misalignments (e.g. trailing words that belong to the next line).
    """

    def combined_match_ratio(line_info, line_text):
        """
        Returns a match ratio comparing:
          normalized(line_text) vs. concatenation of normalized words from line_info['words'].
        """
        norm_line = normalize_text(line_text)
        word_joined = ''.join(normalize_text(w) for w in line_info['words'])
        return match_ratio(norm_line, word_joined)

    def update_times_for_line(line_info):
        """
        If line_info has words, set 'start' to the earliest word's start, 
        and 'end' to the latest word's end.
        Otherwise set both None.
        """
        if line_info['idx_words']:
            # earliest word's start
            first_idx = line_info['idx_words'][0]
            line_info['start'] = float(transcription_words[first_idx]['start'])
            # latest word's end
            last_idx = line_info['idx_words'][-1]
            line_info['end'] = float(transcription_words[last_idx]['end'])
        else:
            line_info['start'] = None
            line_info['end'] = None

    for i in range(len(lines_with_ts) - 1):
        line_i = lines_with_ts[i]
        line_i1 = lines_with_ts[i+1]

        # If either line has no words, skip boundary checks
        if not line_i['words'] or not line_i1['words']:
            continue

        # ----- 1) Try moving last word of line i -> front of line i+1 -----
        old_ratio_i = combined_match_ratio(line_i, line_i['text'])
        old_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        old_sum = old_ratio_i + old_ratio_i1

        last_word_idx = line_i['idx_words'][-1]
        last_word = line_i['words'][-1]

        # Remove from line i
        line_i['idx_words'] = line_i['idx_words'][:-1]
        line_i['words'] = line_i['words'][:-1]

        # Add to the front of line i+1
        line_i1['idx_words'] = [last_word_idx] + line_i1['idx_words']
        line_i1['words'] = [last_word] + line_i1['words']

        # Update times
        update_times_for_line(line_i)
        update_times_for_line(line_i1)

        # Check new ratio
        new_ratio_i = combined_match_ratio(line_i, line_i['text'])
        new_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        new_sum = new_ratio_i + new_ratio_i1

        if new_sum <= old_sum:
            # revert
            line_i1['idx_words'].pop(0)
            line_i1['words'].pop(0)
            line_i['idx_words'].append(last_word_idx)
            line_i['words'].append(last_word)

            # revert times
            update_times_for_line(line_i)
            update_times_for_line(line_i1)
        else:
            # keep the change
            continue

        # ----- 2) Try moving first word of line i+1 -> end of line i -----
        if not line_i1['words']:
            continue

        old_ratio_i = combined_match_ratio(line_i, line_i['text'])
        old_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        old_sum = old_ratio_i + old_ratio_i1

        first_word_idx = line_i1['idx_words'][0]
        first_word = line_i1['words'][0]

        # Remove from line i+1
        line_i1['idx_words'] = line_i1['idx_words'][1:]
        line_i1['words'] = line_i1['words'][1:]

        # Add to line i
        line_i['idx_words'].append(first_word_idx)
        line_i['words'].append(first_word)

        # Update times
        update_times_for_line(line_i)
        update_times_for_line(line_i1)

        # Check new ratio
        new_ratio_i = combined_match_ratio(line_i, line_i['text'])
        new_ratio_i1 = combined_match_ratio(line_i1, line_i1['text'])
        new_sum = new_ratio_i + new_ratio_i1

        if new_sum <= old_sum:
            # revert
            line_i['idx_words'].pop()
            line_i['words'].pop()
            line_i1['idx_words'] = [first_word_idx] + line_i1['idx_words']
            line_i1['words'] = [first_word] + line_i1['words']

            # revert times
            update_times_for_line(line_i)
            update_times_for_line(line_i1)

def validate_improved_lines_with_ts(lines_with_ts):
    """
    Validates the improved lines with timestamps.
    1) Check that idx_words is monotonically increasing within each line.
    2) Ensure a word is not used multiple times (if that is desired).
    """
    used_word_indices = set()
    prev_max_idx = -1

    for i, line_info in enumerate(lines_with_ts):
        idx_words = line_info.get("idx_words", [])
        # Check monotonic within the line
        for j in range(len(idx_words) - 1):
            if idx_words[j] >= idx_words[j+1]:
                raise ValueError(f"Line {i} has non-monotonic idx_words: {idx_words}")

        # Check usage uniqueness
        for widx in idx_words:
            if widx in used_word_indices:
                raise ValueError(f"Word index {widx} reused in line {i}.")
            used_word_indices.add(widx)

        # Optionally ensure lines come in ascending order
        if idx_words:
            if min(idx_words) <= prev_max_idx:
                raise ValueError(
                    f"Line {i} index usage out of order: min idx {min(idx_words)} "
                    f"<= previous max {prev_max_idx}."
                )
            prev_max_idx = max(idx_words)

    print("[validate_improved_lines_with_ts] All checks passed.")

if __name__ == "__main__":
    #####
    # Main code
    # read improved lines from "temp/tmp_improved_lines.json"
    # read transcription words from "temp/tmp_transcription_words.json"
    # generate an alignment
    # run boundary post-processing
    # validate
    # save final results in JSON with 2-space indent, ensure_ascii=False
    #####

    # 1) Load data
    improved_lines_json_path = "temp/tmp_improved_lines.json"
    with open(improved_lines_json_path, "r", encoding="utf-8") as f:
        improved_lines = json.load(f)

    transcription_words_json_path = "temp/tmp_transcription_words.json"
    with open(transcription_words_json_path, "r", encoding="utf-8") as f:
        transcription_words = json.load(f)

    # 2) Basic alignment
    lines_with_ts = get_improved_lines_with_ts(improved_lines, transcription_words)

    # 3) Post-process boundary adjustments
    postprocess_boundaries(lines_with_ts, transcription_words)

    # 4) Validate
    validate_improved_lines_with_ts(lines_with_ts)

    # 5) Save results
    os.makedirs("temp", exist_ok=True)
    final_path = "temp/improved_lines_with_ts_charwise.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(lines_with_ts, f, indent=2, ensure_ascii=False)
    print(f"Final results saved to: {final_path}")
    print("Done.")