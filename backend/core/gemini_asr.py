import os
import sys
import json
import base64
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import httpx

# Support both direct script execution and package import
_curr_dir = str(Path(__file__).resolve().parent)
_backend_dir = str(Path(__file__).resolve().parent.parent)
for _p in [_curr_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from backend.core.ffmpeg_utils import BINARIES, get_video_info
except ImportError:
    from .ffmpeg_utils import BINARIES, get_video_info

def get_available_gemini_models(api_key: str) -> List[str]:
    """Query Google API directly to get the list of models that this exact key supports."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    m["name"].replace("models/", "")
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                return models
            else:
                err_body = resp.text
                print(f"[!] Lỗi khi truy vấn danh sách model ({resp.status_code}): {err_body}")
                return []
    except Exception as e:
        print(f"[!] Lỗi kết nối Google API: {e}")
        return []

def extract_json_data(text: str) -> Any:
    """
    Safely extract and parse JSON from LLM output, ignoring extra trailing text,
    markdown wrapping, or commentary following the JSON array/object.
    """
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    # 1. Try finding outer brackets [ ... ]
    first_bracket = clean_text.find('[')
    last_bracket = clean_text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        try:
            return json.loads(clean_text[first_bracket:last_bracket + 1])
        except Exception:
            pass

    # 2. Try raw_decode from first bracket to safely ignore trailing extra data
    if first_bracket != -1:
        try:
            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(clean_text[first_bracket:])
            return data
        except Exception:
            pass

    # 3. Try finding outer braces { ... }
    first_brace = clean_text.find('{')
    last_brace = clean_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(clean_text[first_brace:last_brace + 1])
        except Exception:
            pass

    # 4. Try raw_decode from first brace
    if first_brace != -1:
        try:
            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(clean_text[first_brace:])
            return data
        except Exception:
            pass

    # 5. Direct load fallback
    return json.loads(clean_text)

def transcribe_with_gemini_sync(
    video_path: str,
    api_key: str = "",
    target_language: str = "Vietnamese",
    character_profiles: str = "",
    style_prompt: str = "",
    model_name: str = "gemini-2.0-flash",
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    Extract audio and call Gemini API via direct REST with base64 inline audio and auto-model detection.
    """
    effective_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not effective_key:
        raise ValueError("Chưa có Gemini API Key! Vui lòng nhập API Key ở cột trái.")

    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.15, "Đang kiểm tra API Key và tự động dò mô hình Gemini khả dụng...")

    # 1. Dynamically discover valid models for this user's key
    valid_models = get_available_gemini_models(effective_key)
    print(f"[*] Các mô hình khả dụng cho API Key: {valid_models}")

    # Build priority list prioritizing fast, working, free-tier models
    preferred_models = [
        "gemini-flash-lite-latest",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-flash-latest"
    ]
    if model_name and "2.5" not in model_name and "tts" not in model_name:
        preferred_models.insert(0, model_name)
    
    # Filter out models that cannot be used for transcription (TTS, image-only, deprecated 2.5)
    candidates = []
    if valid_models:
        # First add preferred models if they are valid
        for pm in preferred_models:
            if pm in valid_models and pm not in candidates:
                candidates.append(pm)
        # Then add any remaining flash/pro models, excluding non-speech/TTS models
        for vm in valid_models:
            is_blacklisted = any(b in vm for b in ["tts", "image", "preview-tts", "lyria", "robotics", "computer-use", "banana", "2.5"])
            if not is_blacklisted and ("flash" in vm or "pro" in vm):
                if vm not in candidates:
                    candidates.append(vm)
    else:
        candidates = ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-2.0-flash"]

    if not candidates:
        raise ValueError("API Key của bạn không có quyền truy cập vào bất kỳ mô hình Gemini nào. Vui lòng kiểm tra lại khóa tại aistudio.google.com!")

    info = get_video_info(video_path)
    total_duration = float(info.get("duration", 0.0))
    if total_duration <= 0:
        total_duration = 300.0

    # 2. Divide video audio into high-focus chunks
    # 45s chunks with 5s overlap provides peak LLM audio attention, preventing skipped dialogue in the middle
    chunk_size = 45.0
    overlap = 5.0

    if total_duration <= 45.0:
        chunks = [(0.0, total_duration)]
    else:
        chunks = []
        cur_start = 0.0
        while cur_start < total_duration:
            cur_end = min(total_duration, cur_start + chunk_size)
            chunks.append((cur_start, cur_end))
            if cur_end >= total_duration:
                break
            cur_start = cur_end - overlap

    num_chunks = len(chunks)
    all_raw_subs = []

    # 3. Process each chunk
    with httpx.Client(timeout=120.0) as http_client:
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_dur = chunk_end - chunk_start
            
            if progress_callback:
                pct = 0.20 + (chunk_idx / num_chunks) * 0.75
                progress_callback(
                    pct,
                    f"Đang bóc băng & dịch đoạn {chunk_idx + 1}/{num_chunks} ({chunk_start:.0f}s - {chunk_end:.0f}s)..."
                )

            chunk_audio_path = temp_dir / f"asr_{Path(video_path).stem}_c{chunk_idx}.mp3"
            cmd = [
                BINARIES["ffmpeg"],
                "-y",
                "-ss", f"{chunk_start:.2f}",
                "-t", f"{chunk_dur:.2f}",
                "-i", video_path,
                "-vn",
                "-af", "dynaudnorm=f=75:g=15,highpass=f=100,lowpass=f=8000",
                "-acodec", "libmp3lame",
                "-ar", "24000",
                "-ac", "1",
                "-b:a", "96k",
                str(chunk_audio_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            with open(chunk_audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            # System instructions
            glossary_section = (
                f"\n\nHỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:\n{character_profiles}"
                if character_profiles else ""
            )
            style_section = f"\nPHONG CÁCH DỊCH: {style_prompt}" if style_prompt else ""

            system_instruction = (
                "Bạn là chuyên gia bóc băng phụ đề (ASR) và dịch thuật phim ảnh chuyên nghiệp hàng đầu.\n"
                "Nhiệm vụ: Nghe kỹ từng giây của đoạn âm thanh và trích xuất TOÀN BỘ từng câu thoại của mọi nhân vật, TUYỆT ĐỐI KHÔNG ĐƯỢC BỎ SÓT BẤT KỲ CÂU NÀO DÙ LÀ NHỎ NHẤT.\n"
                "Bao gồm: lời thoại chính, lời thoại phụ, tiếng thì thầm, nói nhỏ, tiếng gọi giật, câu thoại bị nhạc nền hoặc hiệu ứng đè, câu cảm thán ngắn.\n"
                "Chia nhỏ câu tự nhiên (khoảng 1.5s đến 4.5s mỗi câu) tương ứng với từng khoảng ngắt hơi của diễn viên.\n"
                f"Đồng thời dịch từng câu sang {target_language} mạch lạc, tự nhiên, chuẩn văn phong đời thường, giữ nguyên sắc thái cảm xúc.\n"
                f"{style_section}{glossary_section}\n\n"
                "QUY TẮC BẮT BUỘC:\n"
                "1. Trả về DUY NHẤT một mảng JSON các object theo mẫu:\n"
                "[\n"
                '  {"id": 1, "start": 0.0, "end": 2.35, "text": "<câu gốc tiếng Trung/Anh>", "translated_text": "<bản dịch tiếng Việt>"},\n'
                '  {"id": 2, "start": 2.5, "end": 4.80, "text": "<câu gốc>", "translated_text": "<bản dịch tiếng Việt>"}\n'
                "]\n"
                "2. 'start' và 'end' tính bằng giây (số thực float, ví dụ 1.25) tương đối từ 0.0 đến độ dài đoạn audio này.\n"
                "3. Bắt buộc rà soát từ đầu đến cuối đoạn âm thanh để không bỏ sót bất kỳ phân đoạn nói chuyện nào.\n"
                "4. Tuyệt đối không thêm bất kỳ văn bản giải thích nào ngoài mảng JSON."
            )

            request_payload = {
                "system_instruction": {
                    "parts": [{"text": system_instruction}]
                },
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "audio/mp3",
                                    "data": audio_b64
                                }
                            },
                            {
                                "text": f"Hãy bóc băng và dịch toàn bộ các câu thoại trong đoạn audio dài {chunk_dur:.1f} giây này mà KHÔNG ĐƯỢC BỎ SÓT bất kỳ câu nào."
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.15,
                    "maxOutputTokens": 8192
                }
            }

            response_json = None
            last_error = None
            for model in candidates:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={effective_key}"
                try:
                    resp = http_client.post(url, json=request_payload)
                    if resp.status_code == 200:
                        response_json = resp.json()
                        break
                    else:
                        last_error = f"{resp.status_code}: {resp.text}"
                except Exception as req_err:
                    last_error = str(req_err)

            if not response_json:
                print(f"[!] Đoạn {chunk_idx + 1} gặp lỗi: {last_error}")
                continue

            candidates_list = response_json.get("candidates", [])
            if not candidates_list:
                continue

            text_part = candidates_list[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            data = extract_json_data(text_part)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break

            if isinstance(data, list):
                for item in data:
                    rel_start = float(item.get("start", 0.0))
                    rel_end = float(item.get("end", rel_start + 1.5))
                    abs_start = round(chunk_start + rel_start, 2)
                    abs_end = round(chunk_start + rel_end, 2)
                    if abs_end <= abs_start:
                        abs_end = round(abs_start + 1.5, 2)

                    raw_text = item.get("text", "").strip()
                    trans_text = item.get("translated_text", "").strip()
                    if raw_text or trans_text:
                        all_raw_subs.append({
                            "start": abs_start,
                            "end": abs_end,
                            "text": raw_text,
                            "translated_text": trans_text,
                            "chunk": chunk_idx
                        })

    # 4. Smart Overlap Deduplication and Merging (only across chunk boundaries)
    all_raw_subs.sort(key=lambda s: float(s["start"]))
    final_subtitles = []
    for s in all_raw_subs:
        if not final_subtitles:
            final_subtitles.append(s)
            continue
        last = final_subtitles[-1]
        time_diff = abs(s["start"] - last["start"])
        overlap_time = max(0.0, min(s["end"], last["end"]) - max(s["start"], last["start"]))
        
        # Check if they are referring to the same sentence across chunk boundaries
        s_text = (s.get("text", "") or s.get("translated_text", "")).strip().lower()
        last_text = (last.get("text", "") or last.get("translated_text", "")).strip().lower()
        
        is_same = False
        # Deduplication only applies to overlapping boundary between different chunks
        if s.get("chunk") != last.get("chunk"):
            if time_diff < 3.5:
                if s_text == last_text and s_text:
                    is_same = True
                elif s_text and last_text and (s_text in last_text or last_text in s_text) and len(min(s_text, last_text, key=len)) > 6:
                    is_same = True
                elif overlap_time > 1.2 and (s_text[:8] == last_text[:8] or s_text[-8:] == last_text[-8:]):
                    is_same = True

        if is_same:
            # Keep the longer, more complete version
            if len(s.get("translated_text", "")) + len(s.get("text", "")) > len(last.get("translated_text", "")) + len(last.get("text", "")):
                final_subtitles[-1] = s
            continue
            
        final_subtitles.append(s)

    # Merge short, adjacent fragments into natural dialogue sentences
    try:
        from .smart_aligner import smart_merge_short_subtitles
    except (ImportError, ValueError):
        from smart_aligner import smart_merge_short_subtitles
    
    final_subtitles = smart_merge_short_subtitles(final_subtitles)

    # Assign final sequential IDs
    for idx, s in enumerate(final_subtitles):
        s["id"] = idx + 1
        s["audio_file"] = ""
        s["confidence"] = 0.99

    if progress_callback:
        progress_callback(1.0, f"Hoàn tất: Đã nhận dạng và dịch trọn vẹn {len(final_subtitles)} câu thoại!")

    return final_subtitles

if __name__ == "__main__":
    print("[+] Gemini ASR module loaded successfully!")
    print(f"[*] FFmpeg binary: {BINARIES['ffmpeg']}")
