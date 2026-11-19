import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

def find_ffmpeg_bin() -> Dict[str, str]:
    """Locate ffmpeg and ffprobe executables."""
    ffmpeg_exe = shutil.which("ffmpeg")
    ffprobe_exe = shutil.which("ffprobe")
    
    # Fallback to known winget installation directory if not in PATH
    winget_pkg_dir = Path(r"C:\Users\FPT\AppData\Local\Microsoft\WinGet\Packages")
    if (not ffmpeg_exe or not ffprobe_exe) and winget_pkg_dir.exists():
        for p in winget_pkg_dir.rglob("ffmpeg.exe"):
            ffmpeg_exe = str(p)
            probe_candidate = p.parent / "ffprobe.exe"
            if probe_candidate.exists():
                ffprobe_exe = str(probe_candidate)
            break
            
    return {
        "ffmpeg": ffmpeg_exe or "ffmpeg",
        "ffprobe": ffprobe_exe or "ffprobe"
    }

BINARIES = find_ffmpeg_bin()

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get metadata about video file."""
    cmd = [
        BINARIES["ffprobe"],
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        data = json.loads(res.stdout)
        
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        
        duration = float(data.get("format", {}).get("duration", 0.0))
        if duration == 0.0 and video_stream:
            duration = float(video_stream.get("duration", 0.0))
            
        width = int(video_stream.get("width", 1920)) if video_stream else 1920
        height = int(video_stream.get("height", 1080)) if video_stream else 1080
        
        # Calculate fps
        r_frame_rate = video_stream.get("r_frame_rate", "30/1") if video_stream else "30/1"
        try:
            num, den = map(int, r_frame_rate.split("/"))
            fps = round(num / den, 2) if den != 0 else 30.0
        except Exception:
            fps = 30.0
            
        return {
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "has_audio": audio_stream is not None,
            "filename": os.path.basename(video_path)
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {
            "duration": 0.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "has_audio": True,
            "filename": os.path.basename(video_path)
        }

def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """Extract audio track as 16kHz WAV or MP3."""
    cmd = [
        BINARIES["ffmpeg"],
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False

def extract_frame_at(video_path: str, time_sec: float, output_image_path: str) -> bool:
    """Extract a single frame image at timestamp."""
    cmd = [
        BINARIES["ffmpeg"],
        "-y",
        "-ss", str(max(0.0, time_sec)),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_image_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def generate_filmstrip(video_path: str, output_dir: str, count: int = 12) -> List[str]:
    """Generate a sequence of thumbnail frames for the timeline video track filmstrip."""
    info = get_video_info(video_path)
    dur = float(info.get("duration", 0.0))
    if dur <= 0:
        return []
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    urls = []
    step = dur / max(1, count)
    stem = Path(video_path).stem
    for i in range(count):
        t = i * step + (step * 0.3)
        img_name = f"fs_{stem}_{i}.jpg"
        img_path = out_dir / img_name
        if not img_path.exists():
            extract_frame_at(video_path, t, str(img_path))
        if img_path.exists():
            norm = str(img_path).replace("\\", "/")
            if "/temp/" in norm:
                urls.append(norm[norm.index("/temp/"):])
            else:
                urls.append(f"/temp/{img_name}")
    return urls

def adjust_audio_speed(input_path: str, output_path: str, speed_factor: float) -> bool:
    """Adjust audio speed using ffmpeg atempo filter without changing pitch."""
    # atempo supports values between 0.5 and 2.0. If outside, chain filters.
    filters = []
    factor = speed_factor
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    filters.append(f"atempo={factor:.4f}")
    
    filter_str = ",".join(filters)
    cmd = [
        BINARIES["ffmpeg"],
        "-y",
        "-i", input_path,
        "-filter:a", filter_str,
        "-vn",
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adjusting audio speed: {e}")
        return False
