import os
import sys
import re
import json
import base64
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import httpx

# Support direct script execution and package import
_curr_dir = str(Path(__file__).resolve().parent)
_backend_dir = str(Path(__file__).resolve().parent.parent)
for _p in [_curr_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from backend.core.ffmpeg_utils import BINARIES, get_video_info
    from backend.core.gemini_asr import get_available_gemini_models, extract_json_data
except ImportError:
    from .ffmpeg_utils import BINARIES, get_video_info
    from .gemini_asr import get_available_gemini_models, extract_json_data


def detect_speech_intervals(video_path: str, silence_threshold_db: float = -32.0, min_silence_dur: float = 0.6) -> List[Tuple[float, float]]:
    """
    Use FFmpeg silencedetect to identify intervals containing active audio/speech in the video.
    Returns list of (speech_start, speech_end) tuples.
    """
    info = get_video_info(video_path)
    total_dur = float(info.get("duration", 0.0))
    if total_dur <= 0:
        return []

    cmd = [
        BINARIES["ffmpeg"],
        "-i", video_path,
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_dur}",
        "-f", "null",
        "-"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = res.stderr or ""
    except Exception as e:
        print(f"Error running silencedetect: {e}")
        return [(0.0, total_dur)]

    # Parse silence_start and silence_end
    silences = []
    cur_silence_start = None

    for line in output.splitlines():
        start_m = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_m:
            cur_silence_start = float(start_m.group(1))

        end_m = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_m and cur_silence_start is not None:
            silence_end = float(end_m.group(1))
            silences.append((cur_silence_start, silence_end))
            cur_silence_start = None

    if cur_silence_start is not None:
        silences.append((cur_silence_start, total_dur))

    # Invert silence intervals to obtain speech intervals
    speech_intervals = []
    prev_end = 0.0
    for s_start, s_end in silences:
        if s_start > prev_end + 0.3:
            speech_intervals.append((round(prev_end, 2), round(s_start, 2)))
        prev_end = max(prev_end, s_end)

    if prev_end < total_dur - 0.3:
        speech_intervals.append((round(prev_end, 2), round(total_dur, 2)))

    return speech_intervals


def find_uncovered_speech_gaps(
    existing_subtitles: List[Dict[str, Any]],
    speech_intervals: List[Tuple[float, float]],
    min_gap_sec: float = 2.0,
    total_video_duration: float = 0.0
) -> List[Tuple[float, float]]:
    """
    Find time intervals where speech was missed or skipped.
    Combines direct timeline subtitle gap detection with audio VAD intervals.
    Immune to continuous BGM noise!
    """
    gaps_found = set()

    # 1. Direct timeline subtitle gaps (always reliable)
    if existing_subtitles:
        sorted_subs = sorted(existing_subtitles, key=lambda s: float(s.get("start", 0.0)))
        if sorted_subs[0]["start"] >= min_gap_sec:
            gaps_found.add((0.0, round(float(sorted_subs[0]["start"]), 2)))

        for i in range(len(sorted_subs) - 1):
            cur_end = float(sorted_subs[i].get("end", 0.0))
            nxt_start = float(sorted_subs[i + 1].get("start", 0.0))
            if nxt_start - cur_end >= min_gap_sec:
                gaps_found.add((round(cur_end, 2), round(nxt_start, 2)))

        if total_video_duration > 0:
            last_end = float(sorted_subs[-1].get("end", 0.0))
            if total_video_duration - last_end >= min_gap_sec:
                gaps_found.add((round(last_end, 2), round(total_video_duration, 2)))
    elif total_video_duration > 0:
        gaps_found.add((0.0, round(total_video_duration, 2)))

    # 2. Add uncovered intervals from audio VAD if available
    for sp_start, sp_end in speech_intervals:
        dur = sp_end - sp_start
        if dur < min_gap_sec:
            continue
        covered_duration = 0.0
        for sub in existing_subtitles:
            overlap = max(0.0, min(sp_end, sub["end"]) - max(sp_start, sub["start"]))
            covered_duration += overlap
        if dur > 0 and (covered_duration / dur) < 0.35:
            gaps_found.add((round(sp_start, 2), round(sp_end, 2)))

    # Sort and split slices larger than 25s for peak LLM attention
    sorted_gaps = sorted(list(gaps_found), key=lambda x: x[0])
    refined = []
    for g_start, g_end in sorted_gaps:
        dur = g_end - g_start
        if dur <= 25.0:
            refined.append((g_start, g_end))
        else:
            cur = g_start
            while cur < g_end:
                nxt = min(g_end, round(cur + 24.0, 2))
                refined.append((cur, nxt))
                cur = nxt

    return refined


def smart_merge_short_subtitles(
    subtitles: List[Dict[str, Any]],
    max_merged_dur: float = 6.0,
    max_gap: float = 1.1,
    min_single_dur: float = 2.8
) -> List[Dict[str, Any]]:
    """
    Intelligently merge choppy, fragmented short phrases into natural sentences.
    Solves the problem where 1-second Chinese phrases become crowded, colliding Vietnamese subtitles.
    """
    if not subtitles:
        return []

    sorted_subs = sorted(subtitles, key=lambda s: float(s.get("start", 0.0)))
    merged = []

    for s in sorted_subs:
        if not merged:
            merged.append(dict(s))
            continue

        prev = merged[-1]
        gap = round(float(s.get("start", 0.0)) - float(prev.get("end", 0.0)), 2)
        prev_dur = round(float(prev.get("end", 0.0)) - float(prev.get("start", 0.0)), 2)
        curr_dur = round(float(s.get("end", 0.0)) - float(s.get("start", 0.0)), 2)

        # Merge conditions:
        # 1. Close together (gap <= max_gap)
        # 2. Combined duration stays within natural speaking limit (<= max_merged_dur)
        # 3. At least one of the fragments is too short for natural speech (< min_single_dur)
        can_merge = (gap <= max_gap) and (prev_dur + curr_dur + gap <= max_merged_dur) and (prev_dur < min_single_dur or curr_dur < min_single_dur)
        
        if can_merge:
            prev["end"] = round(float(s.get("end", prev["end"])), 2)
            
            p_text = prev.get("text", "").strip()
            s_text = s.get("text", "").strip()
            prev["text"] = f"{p_text}，{s_text}" if p_text and s_text else (p_text or s_text)
            
            p_tr = prev.get("translated_text", "").strip().rstrip(".,;!?")
            s_tr = s.get("translated_text", "").strip()
            prev["translated_text"] = f"{p_tr}, {s_tr}" if p_tr and s_tr else (p_tr or s_tr)
            
            # Invalidate cached single audio since text has changed
            prev["audio_file"] = ""
            prev["audio_url"] = ""
        else:
            merged.append(dict(s))

    for idx, item in enumerate(merged):
        item["id"] = idx + 1

    return merged


def smart_decollide_and_align_subtitles(
    subtitles: List[Dict[str, Any]],
    total_video_duration: float,
    min_breath_gap: float = 0.30,
    chars_per_second: float = 14.0,
    merge_short: bool = True
) -> List[Dict[str, Any]]:
    """
    Resolve all timing collisions, overlapping speech, and rushed cadence:
    1. Merge choppy short fragments into natural sentences with comfortable speaking time.
    2. Enforce a minimum breath buffer (0.30s) between adjacent dialogue blocks.
    3. Dynamically expand or adjust timestamps so fast/long Vietnamese sentences have enough speaking time.
    4. Ensure no two audio clips ever overlap or cut each other off.
    """
    if not subtitles:
        return []

    if merge_short:
        subtitles = smart_merge_short_subtitles(subtitles)

    # Sort subtitles by start time
    sorted_subs = sorted(subtitles, key=lambda s: float(s.get("start", 0.0)))
    total = len(sorted_subs)
    aligned = []

    for idx, sub in enumerate(sorted_subs):
        item = dict(sub)
        start = round(float(item.get("start", 0.0)), 2)
        end = round(float(item.get("end", start + 2.0)), 2)
        text_vn = item.get("translated_text") or item.get("text", "")
        
        # Estimate natural duration needed to pronounce this sentence
        char_count = len(text_vn.strip())
        natural_min_dur = max(1.2, char_count / chars_per_second)
        
        # Keep start anchored to original video dialogue timestamp!
        # If previous sub end is too close, trim previous sub end rather than pushing this sub into the future
        if aligned:
            prev = aligned[-1]
            if prev["end"] > start - min_breath_gap:
                prev["end"] = max(prev["start"] + 0.8, round(start - min_breath_gap, 2))

        # Check next anchor to not exceed boundary
        if idx + 1 < total:
            next_start = round(float(sorted_subs[idx + 1].get("start", 0.0)), 2)
            if next_start > start + 0.4:
                end = min(end, round(next_start - min_breath_gap, 2))
                # Expand end if there is ample dead silence before next speaker
                if end < start + natural_min_dur:
                    end = min(round(next_start - min_breath_gap, 2), round(start + natural_min_dur, 2))
        else:
            if end - start < natural_min_dur:
                end = round(start + natural_min_dur, 2)

        # Cap within video total duration
        if total_video_duration > 0:
            start = min(start, max(0.0, total_video_duration - 0.5))
            end = min(end, total_video_duration)
            if end <= start:
                end = min(total_video_duration, start + 1.0)

        item["start"] = round(start, 2)
        item["end"] = round(end, 2)
        aligned.append(item)

    # Re-index IDs sequentially
    for i, s in enumerate(aligned):
        s["id"] = i + 1

    return aligned


async def recover_missing_dialogue_with_ai(
    video_path: str,
    missing_gaps: List[Tuple[float, float]],
    api_key: str,
    target_language: str = "Vietnamese",
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    Use Gemini ASR with targeted high-gain audio extraction on detected missing gaps
    to recover dialogues that were skipped during the initial broad pass.
    """
    if not missing_gaps:
        return []

    effective_key = api_key.strip()
    if not effective_key:
        return []

    models = get_available_gemini_models(effective_key)
    candidate_model = "gemini-flash-lite-latest"
    for m in ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-2.0-flash", "gemini-flash-latest"]:
        if m in models:
            candidate_model = m
            break

    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    recovered_subs = []
    total_gaps = len(missing_gaps)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, (gap_start, gap_end) in enumerate(missing_gaps):
            gap_dur = gap_end - gap_start
            if gap_dur < 1.2:
                continue

            if progress_callback:
                progress_callback(
                    (idx / total_gaps) * 0.9,
                    f"AI đang bóc băng vét phân đoạn chưa nhận diện ({gap_start:.1f}s - {gap_end:.1f}s)..."
                )

            # Extract high-gain audio for whisper / soft dialogue recovery
            clip_path = temp_dir / f"gap_{idx}_{int(gap_start)}_{int(gap_end)}.mp3"
            cmd = [
                BINARIES["ffmpeg"], "-y",
                "-ss", f"{gap_start:.2f}",
                "-t", f"{gap_dur:.2f}",
                "-i", video_path,
                "-vn",
                "-af", "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=120,lowpass=f=7500,volume=1.5",
                "-acodec", "libmp3lame",
                "-ar", "24000",
                "-ac", "1",
                "-b:a", "96k",
                str(clip_path)
            ]
            await asyncio.to_thread(subprocess.run, cmd, capture_output=True)

            if not clip_path.exists() or os.path.getsize(str(clip_path)) < 100:
                continue

            with open(str(clip_path), "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            prompt = (
                "Bạn là chuyên gia bóc băng âm thanh siêu nhạy.\n"
                f"Đoạn âm thanh này dài {gap_dur:.2f} giây. Có thể chứa tiếng nói nhỏ, thì thầm, tiếng nói bị nhạc đè hoặc lời thoại ngắn.\n"
                "Hãy nghe thật kỹ và trả về danh sách các câu nói dưới dạng JSON:\n"
                '[\n  {"start": 0.0, "end": 2.1, "text": "<câu gốc>", "translated_text": "<dịch tiếng Việt>"}\n]\n'
                "Nếu là tiếng nhạc/tiếng ồn hoàn toàn không có người nói, trả về mảng rỗng: []\n"
                "Chỉ trả về duy nhất JSON hợp lệ."
            )

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate_model}:generateContent?key={effective_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"inline_data": {"mime_type": "audio/mp3", "data": b64}},
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.1
                }
            }

            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    res_json = resp.json()
                    parts = res_json.get("candidates", [])[0].get("content", {}).get("parts", [])
                    if parts:
                        parsed = extract_json_data(parts[0].get("text", ""))
                        if isinstance(parsed, list):
                            for itm in parsed:
                                raw = itm.get("text", "").strip()
                                trans = itm.get("translated_text", "").strip()
                                if raw or trans:
                                    s_start = round(gap_start + float(itm.get("start", 0.0)), 2)
                                    s_end = round(gap_start + float(itm.get("end", s_start + 1.5)), 2)
                                    recovered_subs.append({
                                        "start": s_start,
                                        "end": s_end,
                                        "text": raw or trans,
                                        "translated_text": trans or raw,
                                        "audio_file": None,
                                        "audio_url": None,
                                        "is_recovered": True
                                    })
            except Exception as e:
                print(f"Error recovering gap {idx}: {e}")

            # Clean up temp clip
            try:
                clip_path.unlink(missing_ok=True)
            except OSError:
                pass

    return recovered_subs


def apply_broadcast_voice_mastering(
    input_audio: str,
    output_audio: str,
    target_lufs: float = -16.0,
    clean_noise: bool = True
) -> bool:
    """
    VidLab Architecture Voice Clean-Up & Audio Mastering:
    Highpass filter + Gentle EQ + Compressor + EBU R128 Loudness Normalization to -16 LUFS.
    Ensures voice is punchy, broadcast-standard, and never clips or drops below audible levels.
    """
    filters = []
    # 1. Clean rumble & low-end mic hum
    filters.append("highpass=f=75")
    # 2. De-harsh high frequencies
    filters.append("lowpass=f=9500")
    # 3. Dynamic audio leveling & gentle vocal compression
    filters.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=80")
    # 4. Broadcast EBU R128 loudness normalization (-16 LUFS)
    filters.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=10")

    cmd = [
        BINARIES["ffmpeg"], "-y",
        "-i", input_audio,
        "-af", ",".join(filters),
        "-ar", "24000",
        "-b:a", "128k",
        output_audio
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in voice mastering: {e.stderr}")
        return False
