import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# Support both direct script execution and package import
_curr_dir = str(Path(__file__).resolve().parent)
_backend_dir = str(Path(__file__).resolve().parent.parent)
for _p in [_curr_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .ffmpeg_utils import BINARIES, get_video_info
except (ImportError, ValueError):
    from ffmpeg_utils import BINARIES, get_video_info

def generate_ass_subtitle_file(
    subtitles: List[Dict[str, Any]],
    output_ass_path: str,
    font_name: str = "Arial",
    font_size: int = 26,
    primary_color: str = "#FFE128",
    outline_color: str = "#000000",
    outline_width: int = 2,
    margin_v: int = 40,
    alignment: int = 2,
    pos_x_pct: Optional[float] = None,
    pos_y_pct: Optional[float] = None,
    video_width: int = 1920,
    video_height: int = 1080
):
    """
    Generate an Advanced SubStation Alpha (.ass) file with precise typography styling.
    Color hex format in ASS is &HAABBGGRR.
    Supports exact (X, Y) pixel positioning from percentage coords (pos_x_pct, pos_y_pct).
    """
    def hex_to_ass_color(hex_str: str) -> str:
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 6:
            r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6]
            return f"&H00{b}{g}{r}&"
        return "&H00FFFFFF&"

    ass_primary = hex_to_ass_color(primary_color)
    ass_outline = hex_to_ass_color(outline_color)
    
    def sec_to_ass_time(sec: float) -> str:
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        seconds = int(sec % 60)
        centisecs = int(round((sec - int(sec)) * 100))
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centisecs:02d}"

    # Scale typography relative to standard 576p height
    scale_factor = max(0.6, video_height / 576.0)
    ass_fontsize = max(16, int(round(font_size * scale_factor)))
    ass_out_width = max(1, int(round(outline_width * scale_factor)))

    header = f"""[Script Info]
Title: AutoSub Studio Export
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{ass_fontsize},{ass_primary},&H000000FF,{ass_outline},&H90000000,-1,0,0,0,100,100,0,0,1,{ass_out_width},1,{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Calculate explicit pixel position if percentage coords are specified
    pos_tag = ""
    if pos_x_pct is not None and pos_y_pct is not None:
        try:
            px = int(round(video_width * float(pos_x_pct)))
            py = int(round(video_height * float(pos_y_pct)))
            # \an5 places center of text box at (px, py)
            pos_tag = f"{{\\an5\\pos({px},{py})}}"
        except (ValueError, TypeError):
            pos_tag = ""

    events = []
    for sub in subtitles:
        text = sub.get("translated_text") or sub.get("text", "")
        if not text.strip():
            continue
        start_str = sec_to_ass_time(sub["start"])
        end_str = sec_to_ass_time(sub["end"])
        clean_text = text.replace("\n", "\\N")
        events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{pos_tag}{clean_text}")

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

def build_filter_complex(
    subtitles: List[Dict[str, Any]],
    video_width: int,
    video_height: int,
    blur_box: Optional[Dict[str, Any]] = None,
    color_filter: str = "none",
    ass_path: Optional[str] = None
) -> str:
    """Build FFmpeg video filter chain for de-sub blur, color grade, and subtitle burning."""
    filters = []
    current_stream = "[0:v]"
    
    # 1. Old Subtitle Blur / De-sub (FFmpeg Boxblur or Mosaic)
    if blur_box and blur_box.get("enabled", True):
        # Coordinates can be normalized (0.0-1.0) or pixel
        if "x_pct" in blur_box:
            bx = int(blur_box["x_pct"] * video_width)
            by = int(blur_box["y_pct"] * video_height)
            bw = int(blur_box["w_pct"] * video_width)
            bh = int(blur_box["h_pct"] * video_height)
        else:
            bx = int(blur_box.get("x", video_width * 0.1))
            by = int(blur_box.get("y", video_height * 0.77))
            bw = int(blur_box.get("w", video_width * 0.8))
            bh = int(blur_box.get("h", video_height * 0.14))

        # Padding adjustment
        pad_t = int(blur_box.get("pad_top", 0))
        pad_b = int(blur_box.get("pad_bottom", 0))
        pad_l = int(blur_box.get("pad_left", 0))
        pad_r = int(blur_box.get("pad_right", 0))

        bx = max(0, bx - pad_l)
        by = max(0, by - pad_t)
        bw = min(video_width - bx, bw + pad_l + pad_r)
        bh = min(video_height - by, bh + pad_t + pad_b)
        
        # Build timeline enable expression based on subtitle timings
        if blur_box.get("sync_subtitles", True) and subtitles:
            time_conditions = [f"between(t,{max(0, s['start']-0.1):.2f},{s['end']+0.1:.2f})" for s in subtitles]
            enable_expr = ":enable='" + "+".join(time_conditions) + "'"
        else:
            enable_expr = ""
            
        blur_type = blur_box.get("type", "blur") # 'blur', 'mosaic', or 'box'
        blur_rad = int(blur_box.get("blur_radius", 36))
        blur_opacity = float(blur_box.get("opacity", 0.85))

        if blur_type == "mosaic":
            filters.append(
                f"{current_stream}split=2[base][crop_in]; "
                f"[crop_in]crop={bw}:{bh}:{bx}:{by},scale=iw/24:ih/24,scale={bw}:{bh}:flags=neighbor[pix]; "
                f"[base][pix]overlay={bx}:{by}{enable_expr}[v_blur]"
            )
            current_stream = "[v_blur]"
        elif blur_type == "box":
            # Solid black banner / inpainting strip with customizable opacity
            filters.append(
                f"{current_stream}drawbox=x={bx}:y={by}:w={bw}:h={bh}:color=black@{blur_opacity:.2f}:t=fill{enable_expr}[v_blur]"
            )
            current_stream = "[v_blur]"
        else:
            # True Gaussian Blur (eliminates original subtitles cleanly without boxblur chroma limitations)
            sigma = max(4.0, min(64.0, blur_rad / 2.0))
            filters.append(
                f"{current_stream}split=2[base][crop_in]; "
                f"[crop_in]crop={bw}:{bh}:{bx}:{by},gblur=sigma={sigma:.1f}:steps=3,drawbox=x=0:y=0:w={bw}:h={bh}:color=black@{blur_opacity:.2f}:t=fill[blurred]; "
                f"[base][blurred]overlay={bx}:{by}{enable_expr}[v_blur]"
            )
            current_stream = "[v_blur]"

    # 2. Cinematic Color Filters
    if color_filter == "cinematic":
        filters.append(f"{current_stream}colorbalance=rs=0.08:gs=0.02:bs=-0.06,eq=contrast=1.12:saturation=1.15[v_col]")
        current_stream = "[v_col]"
    elif color_filter == "vivid":
        filters.append(f"{current_stream}eq=contrast=1.18:saturation=1.35:brightness=0.02[v_col]")
        current_stream = "[v_col]"
    elif color_filter == "warm":
        filters.append(f"{current_stream}colorbalance=rs=0.12:gs=0.04:bs=-0.08,eq=saturation=1.1[v_col]")
        current_stream = "[v_col]"
    elif color_filter == "cool":
        filters.append(f"{current_stream}colorbalance=rs=-0.08:gs=0.01:bs=0.12,eq=contrast=1.05[v_col]")
        current_stream = "[v_col]"
    elif color_filter == "vintage":
        filters.append(f"{current_stream}curves=vintage,eq=saturation=0.82:contrast=1.1[v_col]")
        current_stream = "[v_col]"

    # 3. Burn Subtitle
    if ass_path and os.path.exists(ass_path):
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
        filters.append(f"{current_stream}subtitles='{escaped_ass}'[v_out]")
        current_stream = "[v_out]"
    else:
        if filters:
            last_filter = filters[-1]
            if not last_filter.endswith("[v_out]"):
                filters[-1] = last_filter.rsplit("[", 1)[0] + "[v_out]"
        else:
            filters.append(f"{current_stream}null[v_out]")

    return "; ".join(filters)

def render_project_video(
    video_path: str,
    subtitles: List[Dict[str, Any]],
    output_path: str,
    blur_box: Optional[Dict[str, Any]] = None,
    color_filter: str = "none",
    sub_style: Optional[Dict[str, Any]] = None,
    isolate_bgm: bool = True,
    original_audio_vol: float = 0.8,
    tts_audio_vol: float = 1.25,
    burn_subtitles: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> bool:
    """Full render pipeline with video filters, BGM vocal-separation, audio mixing, and subtitle burn."""
    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    video_info = get_video_info(video_path)
    vid_w = video_info.get("width", 1920)
    vid_h = video_info.get("height", 1080)

    # 1. Generate Subtitle ASS File (only if burning subtitles is enabled)
    ass_file = None
    if burn_subtitles and subtitles:
        if progress_callback:
            progress_callback(0.10, "Đang thiết lập kiểu chữ và tạo file phụ đề ASS...")
        ass_file = str(temp_dir / f"render_{Path(output_path).stem}.ass")
        style = sub_style or {}
        generate_ass_subtitle_file(
            subtitles,
            ass_file,
            font_name=style.get("font_name", "Arial"),
            font_size=style.get("font_size", 26),
            primary_color=style.get("color", "#FFE128"),
            outline_color=style.get("outline_color", "#000000"),
            outline_width=style.get("outline_width", 2),
            margin_v=style.get("margin_v", 40),
            alignment=style.get("alignment", 2),
            pos_x_pct=style.get("pos_x_pct"),
            pos_y_pct=style.get("pos_y_pct"),
            video_width=vid_w,
            video_height=vid_h
        )

    # 2. Build Audio Inputs & Filter Complex
    if progress_callback:
        progress_callback(0.25, "Đang đồng bộ timeline các đoạn lồng tiếng AI...")
    cmd = [BINARIES["ffmpeg"], "-y", "-i", video_path]
    
    tts_inputs = []
    base_dir = Path(__file__).resolve().parent.parent.parent
    for sub in subtitles:
        audio_file = sub.get("audio_file")
        if not audio_file and sub.get("audio_url"):
            url_part = sub["audio_url"].lstrip("/").replace("/", os.sep)
            for cand in [base_dir / url_part, Path(url_part), temp_dir / Path(url_part).name]:
                if cand.exists():
                    audio_file = str(cand)
                    break
        if audio_file and os.path.exists(audio_file):
            tts_inputs.append((sub["start"], audio_file))
            
    for _, a_file in tts_inputs:
        cmd.extend(["-i", a_file])
        
    audio_filters = []
    if tts_inputs:
        # Step 1: Delay each TTS segment to its exact subtitle start timestamp
        tts_streams = []
        for idx, (start_time, _) in enumerate(tts_inputs):
            input_idx = idx + 1
            delay_ms = int(round(start_time * 1000))
            audio_filters.append(f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[tts_{idx}]")
            tts_streams.append(f"[tts_{idx}]")
            
        # Step 2: Combine all TTS segments into one master voiceover track with normalize=0
        if len(tts_streams) == 1:
            audio_filters.append(f"{tts_streams[0]}volume={tts_audio_vol}[a_voice]")
        else:
            audio_filters.append(f"{''.join(tts_streams)}amix=inputs={len(tts_streams)}:normalize=0:dropout_transition=0,volume={tts_audio_vol}[a_voice]")
            
        # Step 3: Mix background audio with loud & clear AI voiceover
        if isolate_bgm:
            # Vocal Cancellation filter: remove center channel (dialogue), boost side channels (stereo BGM & SFX)
            # Ensure stereo channel layout before stereotools so mono or surround never crash
            audio_filters.append(
                f"[0:a]aformat=channel_layouts=stereo,stereotools=mlev=0.015625:slev=1.45,volume={original_audio_vol:.2f}[a_bg]"
            )
        else:
            audio_filters.append(f"[0:a]volume={original_audio_vol:.2f}[a_bg]")
            
        audio_filters.append(f"[a_bg][a_voice]amix=inputs=2:duration=first:normalize=0:dropout_transition=0[a_out]")
    else:
        # No TTS, keep original audio (optionally remove dialogue if requested)
        if isolate_bgm:
            audio_filters.append(f"[0:a]aformat=channel_layouts=stereo,stereotools=mlev=0.015625:slev=1.45,volume={original_audio_vol:.2f}[a_out]")
        else:
            audio_filters.append(f"[0:a]volume=1.0[a_out]")
    
    # Video Filter Complex
    v_filter = build_filter_complex(subtitles, vid_w, vid_h, blur_box, color_filter, ass_file)
    full_filter_complex = f"{v_filter}; {'; '.join(audio_filters)}"
    
    cmd.extend([
        "-filter_complex", full_filter_complex,
        "-map", "[v_out]",
        "-map", "[a_out]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ])
    
    if progress_callback:
        progress_callback(0.40, "Đang xử lý FFmpeg: Xóa sub cũ, đốt phụ đề & hòa âm lồng tiếng...")
        
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if progress_callback:
            progress_callback(1.0, f"Đã xuất video thành công: {os.path.basename(output_path)}")
        return True
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr or ""
        print(f"Error rendering video: {stderr_text}")
        lines = [l.strip() for l in stderr_text.splitlines() if l.strip()]
        real_errors = [
            l for l in lines
            if not l.startswith("ffmpeg version")
            and not l.startswith("built with")
            and not l.startswith("configuration:")
            and not l.startswith("libav")
            and not l.startswith("Input #")
            and not l.startswith("Output #")
            and not l.startswith("Stream #")
            and not l.startswith("Stream mapping:")
            and not l.startswith("Press [q]")
            and not l.startswith("frame=")
            and not l.startswith("size=")
            and not l.startswith("Copyright (c)")
        ]
        last_err = " | ".join(real_errors[-3:]) if real_errors else (stderr_text[-300:] if stderr_text else "Unknown error")
        if progress_callback:
            progress_callback(-1.0, f"Lỗi render: {last_err}")
        return False

if __name__ == "__main__":
    print("[+] Render Engine module loaded successfully!")
    print(f"[*] FFmpeg binary: {BINARIES['ffmpeg']}")
    print(f"[*] FFprobe binary: {BINARIES['ffprobe']}")
