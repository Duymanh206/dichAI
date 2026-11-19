import os
import sys
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import edge_tts

# Support both direct script execution and package import
_curr_dir = str(Path(__file__).resolve().parent)
_backend_dir = str(Path(__file__).resolve().parent.parent)
for _p in [_curr_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .ffmpeg_utils import BINARIES, get_video_info, adjust_audio_speed
    from .smart_aligner import smart_decollide_and_align_subtitles, apply_broadcast_voice_mastering
except (ImportError, ValueError):
    from ffmpeg_utils import BINARIES, get_video_info, adjust_audio_speed
    from smart_aligner import smart_decollide_and_align_subtitles, apply_broadcast_voice_mastering

# Predefined available voices
AVAILABLE_VOICES = [
    {
        "id": "vi-VN-HoaiMyNeural",
        "name": "Hoài My (Nữ - Truyền cảm)",
        "engine": "Edge TTS",
        "lang": "vi-VN",
        "gender": "Female"
    },
    {
        "id": "vi-VN-NamMinhNeural",
        "name": "Nam Minh (Nam - Trầm ấm)",
        "engine": "Edge TTS",
        "lang": "vi-VN",
        "gender": "Male"
    },
    {
        "id": "capcut-de-thuong",
        "name": "Dễ Thương (Nữ CapCut)",
        "engine": "CapCut",
        "lang": "vi-VN",
        "gender": "Female"
    },
    {
        "id": "capcut-anh-dung",
        "name": "Anh Dũng (Nam CapCut)",
        "engine": "CapCut",
        "lang": "vi-VN",
        "gender": "Male"
    },
    {
        "id": "capcut-chi-mai",
        "name": "Chi Mai (Nữ CapCut)",
        "engine": "CapCut",
        "lang": "vi-VN",
        "gender": "Female"
    },
    {
        "id": "capcut-duy-bac",
        "name": "Duy Bắc (Nam Bắc)",
        "engine": "CapCut",
        "lang": "vi-VN",
        "gender": "Male"
    }
]

def apply_pronunciation_rules(text: str, rules: List[Dict[str, str]]) -> str:
    """Replace words according to custom pronunciation rules dictionary."""
    if not rules or not text:
        return text
    processed = text
    for rule in rules:
        origin = rule.get("from", "").strip()
        pronounce = rule.get("to", "").strip()
        if origin and pronounce:
            # Case-insensitive replace for standalone or substring
            pattern = re.compile(re.escape(origin), re.IGNORECASE)
            processed = pattern.sub(pronounce, processed)
    return processed

def clean_text_for_speech(text: str) -> str:
    """Clean text and trailing punctuations that can trigger Edge-TTS synthesis glitches."""
    if not text:
        return ""
    # Strip unnecessary trailing dots or repeated symbols
    cleaned = re.sub(r'[\.]{2,}', '.', text)
    cleaned = re.sub(r'[,]{2,}', ',', cleaned)
    cleaned = cleaned.strip()
    return cleaned

async def generate_single_tts(
    text: str,
    voice_id: str,
    output_path: str,
    target_duration: Optional[float] = None,
    auto_fit: bool = True
) -> bool:
    """Generate audio for a single sentence with natural tempo and CapCut acoustic profiles."""
    text_clean = clean_text_for_speech(text)
    if not text_clean:
        return False

    temp_raw = output_path + ".raw.mp3"
    vid = (voice_id or "").lower()
    is_male = any(m in vid for m in ["male", "nam", "dung", "bac"])

    # High-quality native Neural Edge-TTS voice mapping
    if "anh-dung" in vid:
        edge_voice = "vi-VN-NamMinhNeural"
        rate_str = "+4%"
        pitch_str = "-2Hz"
    elif "duy-bac" in vid:
        edge_voice = "vi-VN-NamMinhNeural"
        rate_str = "+7%"
        pitch_str = "+2Hz"
    elif "de-thuong" in vid:
        edge_voice = "vi-VN-HoaiMyNeural"
        rate_str = "+6%"
        pitch_str = "+3Hz"
    elif "chi-mai" in vid:
        edge_voice = "vi-VN-HoaiMyNeural"
        rate_str = "-2%"
        pitch_str = "-1Hz"
    elif is_male:
        edge_voice = "vi-VN-NamMinhNeural"
        rate_str = "+0%"
        pitch_str = "+0Hz"
    else:
        edge_voice = "vi-VN-HoaiMyNeural"
        rate_str = "+0%"
        pitch_str = "+0Hz"

    generated = False
    # 1. Primary: High-speed Neural Edge-TTS with native tone & rate
    try:
        communicate = edge_tts.Communicate(text_clean, edge_voice, rate=rate_str, pitch=pitch_str)
        await communicate.save(temp_raw)
        if os.path.exists(temp_raw) and os.path.getsize(temp_raw) > 200:
            generated = True
    except Exception as e:
        print(f"[-] Edge-TTS warning for '{text_clean[:20]}': {e}. Đang chuyển sang Google TTS fallback...")

    # 2. Secondary fallback: Google Translate TTS via non-blocking thread
    if not generated:
        try:
            from gtts import gTTS
            tts = gTTS(text=text_clean, lang='vi', slow=False)
            await asyncio.to_thread(tts.save, temp_raw)
            if os.path.exists(temp_raw) and os.path.getsize(temp_raw) > 200:
                generated = True
        except Exception as e2:
            print(f"[!] Lỗi TTS fallback cho '{text_clean[:20]}': {e2}")
            generated = False

    if not generated or not os.path.exists(temp_raw) or os.path.getsize(temp_raw) == 0:
        if os.path.exists(temp_raw):
            try:
                os.remove(temp_raw)
            except OSError:
                pass
        return False

    try:
        # Check generated audio duration
        info = get_video_info(temp_raw)
        gen_duration = info.get("duration", 0.0)
        
        filters = []
        # Gentle, natural speed compensation (strictly bounded between 0.92x and 1.18x)
        # Prevents jarring speed rushes ("lúc nhanh lúc chậm")
        if auto_fit and target_duration and target_duration > 0.5 and gen_duration > 0:
            speed_ratio = gen_duration / target_duration
            if speed_ratio > 1.04:
                # Speed up naturally (up to 1.35x) to fit the original dialogue slot
                factor = min(1.35, speed_ratio)
                filters.append(f"atempo={factor:.2f}")
            elif speed_ratio < 0.82:
                # Slight slow down to fill long slot
                factor = max(0.88, speed_ratio)
                filters.append(f"atempo={factor:.2f}")

        if filters:
            from .ffmpeg_utils import BINARIES
            filter_str = ",".join(filters)
            cmd = [
                BINARIES["ffmpeg"], "-y",
                "-i", temp_raw,
                "-af", filter_str,
                output_path
            ]
            import subprocess
            res = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
            if res.returncode == 0 and os.path.exists(output_path):
                try:
                    os.remove(temp_raw)
                except OSError:
                    pass
                return True

        # If no filter needed, rename temp to final output
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_raw, output_path)
        return True
    except Exception as e:
        print(f"Error post-processing TTS for '{text}': {e}")
        if os.path.exists(temp_raw) and not os.path.exists(output_path):
            try:
                os.rename(temp_raw, output_path)
                return True
            except OSError:
                pass
        return False

async def generate_all_subtitles_tts(
    subtitles: List[Dict[str, Any]],
    voice_id: str = "vi-VN-HoaiMyNeural",
    output_dir: str = "temp/tts",
    pronunciation_rules: Optional[List[Dict[str, str]]] = None,
    auto_fit: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> List[Dict[str, Any]]:
    """Generate TTS audio files for all subtitles with consistent natural pacing and auto duration sync."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Pre-align and de-collide subtitles before TTS to ensure no overlap and natural breath spacing
    subtitles = smart_decollide_and_align_subtitles(subtitles, total_video_duration=99999.0, min_breath_gap=0.30)
    total = len(subtitles)
    
    for idx, sub in enumerate(subtitles):
        text_to_speak = sub.get("translated_text") or sub.get("text", "")
        if not text_to_speak.strip():
            continue
            
        processed_text = apply_pronunciation_rules(text_to_speak, pronunciation_rules or [])
        out_filename = f"tts_sub_{sub['id']:04d}.mp3"
        out_path = os.path.abspath(os.path.join(output_dir, out_filename))
        
        # Determine actual available duration before next character speaks
        if idx + 1 < total and subtitles[idx + 1]["start"] > sub["start"]:
            avail_dur = max(sub["end"] - sub["start"], subtitles[idx + 1]["start"] - sub["start"] - 0.25)
        else:
            avail_dur = max(sub["end"] - sub["start"], 4.0)

        target_dur = avail_dur
        if progress_callback:
            progress_callback(
                idx / total,
                f"Đang tạo giọng đọc câu {idx + 1} / {total}: '{processed_text[:25]}...'"
            )
            
        cur_voice = sub.get("voice_id") or voice_id
        success = await generate_single_tts(
            processed_text,
            voice_id=cur_voice,
            output_path=out_path,
            target_duration=target_dur,
            auto_fit=auto_fit
        )
        if success:
            sub["audio_file"] = out_path
            # Dynamically sync subtitle end time with actual spoken duration so words are never cut off
            info = get_video_info(out_path)
            act_dur = float(info.get("duration", 0.0))
            if act_dur > 0:
                sub["audio_duration"] = round(act_dur, 2)
                # Keep subtitle visible for the entire duration of the spoken sentence
                sub["end"] = round(sub["start"] + act_dur, 2)
                # If this sentence duration overlaps with the next sentence, push next start with natural breath gap
                if idx + 1 < total:
                    min_gap = 0.20
                    if subtitles[idx + 1]["start"] < sub["end"] + min_gap:
                        diff = round((sub["end"] + min_gap) - subtitles[idx + 1]["start"], 2)
                        subtitles[idx + 1]["start"] = round(sub["end"] + min_gap, 2)
                        subtitles[idx + 1]["end"] = round(subtitles[idx + 1]["end"] + diff, 2)

        await asyncio.sleep(0.05)
            
    if progress_callback:
        progress_callback(1.0, f"Đã tạo hoàn tất {total} file âm thanh thuyết minh!")
        
    return subtitles

if __name__ == "__main__":
    print("[+] TTS Engine module loaded successfully!")
    print(f"[*] Available voices count: {len(AVAILABLE_VOICES)}")
