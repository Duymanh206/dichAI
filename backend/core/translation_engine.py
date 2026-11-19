import json
import os
import re
import sys
from pathlib import Path
import httpx
from typing import List, Dict, Any, Optional, Callable

# Support both direct script execution and package import
_curr_dir = str(Path(__file__).resolve().parent)
_backend_dir = str(Path(__file__).resolve().parent.parent)
for _p in [_curr_dir, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .gemini_asr import get_available_gemini_models, extract_json_data
except (ImportError, ValueError):
    from gemini_asr import get_available_gemini_models, extract_json_data

PROMPT_STYLES = {
    "default": (
        "Bạn là chuyên gia dịch thuật video chuyên nghiệp. Dịch các câu phụ đề sau sang tiếng Việt tự nhiên, "
        "mạch lạc, chuẩn văn phong đời thường, giữ nguyên ý nghĩa và ngữ cảnh phim."
    ),
    "co_trang": (
        "Bối cảnh: Phim Cổ Trang / Cung Đấu Trung Quốc. Dịch sang tiếng Việt mang âm hưởng Hán Việt tinh tế, "
        "chuẩn mực xưng hô cung đình (hoàng thượng, thái hậu, thần thiếp, bản vương, nô tài, thảo dân, v.v.)."
    ),
    "kiem_hiep": (
        "Bối cảnh: Phim Kiếm Hiệp / Tiên Hiệp. Dịch chuẩn văn phong giang hồ võ lâm (sư huynh, chưởng môn, tại hạ, "
        "các hạ, đạo hữu, bản tọa, chiêu thức, chân khí, tu vi, đan điền, v.v.)."
    ),
    "xuyen_khong": (
        "Bối cảnh: Phim Xuyên Không / Hệ Thống. Dịch khéo léo kết hợp giữa xưng hô cổ trang và ngôn từ hiện đại "
        "khi nhân vật suy nghĩ nội tâm hoặc giao tiếp với hệ thống."
    ),
    "hien_dai": (
        "Bối cảnh: Phim Đô Thị / Hiện Đại. Dịch khẩu ngữ tự nhiên, gần gũi, xưng hô linh hoạt theo ngữ cảnh đời sống "
        "(anh/em, cậu/tớ, mày/tao, bác/cháu)."
    ),
    "hai_huoc": (
        "Bối cảnh: Phim Hài Hước / Hoạt Hình (SpongeBob, Tom & Jerry...). Dịch dí dỏm, biểu cảm, bắt trend nhẹ nhàng, "
        "tạo tiếng cười tự nhiên phù hợp với khán giả Việt Nam."
    ),
    "tom_tat": (
        "Bối cảnh: Video Kể Chuyện / Tóm Tắt Phim (Review phim). Dịch dưới góc nhìn người kể chuyện (thuyết minh viên), "
        "câu từ súc tích, lôi cuốn, xưng hô 'cậu ấy, cô ta, gã này, lúc này, bất ngờ là...'"
    )
}

async def translate_subtitles_batch(
    subtitles: List[Dict[str, Any]],
    provider: str = "gemini",
    api_key: str = "",
    model: str = "gemini-2.5-flash",
    base_url: str = "",
    style_key: str = "default",
    custom_system_prompt: str = "",
    character_profiles: str = "",
    target_language: str = "Vietnamese",
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    Translates subtitles in context-aware batches while preserving IDs and timing.
    """
    if not subtitles:
        return []
        
    style_prompt = custom_system_prompt or PROMPT_STYLES.get(style_key, PROMPT_STYLES["default"])
    char_context = f"\n\nHồ sơ nhân vật & xưng hô cần tuân thủ:\n{character_profiles}" if character_profiles else ""
    
    system_instruction = (
        f"{style_prompt}{char_context}\n\n"
        f"Mục tiêu dịch: Chuyển toàn bộ các câu thoại sang ngôn ngữ: {target_language}.\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Trả về DUY NHẤT một mảng JSON định dạng: [{\"id\": <id>, \"translation\": \"<câu dịch>\"}, ...]\n"
        "2. Không thêm bất kỳ lời giải thích, ghi chú hay định dạng markdown thừa ngoài JSON.\n"
        "3. Dịch đúng ngữ cảnh xuyên suốt giữa các câu liên tiếp, không dịch thô cứng máy móc."
    )

    batch_size = 25
    total_subs = len(subtitles)
    
    for start_idx in range(0, total_subs, batch_size):
        end_idx = min(start_idx + batch_size, total_subs)
        batch = subtitles[start_idx:end_idx]
        
        if progress_callback:
            progress_callback(
                start_idx / total_subs,
                f"Đang dịch câu {start_idx + 1} - {end_idx} / {total_subs}..."
            )
            
        input_payload = [{"id": s["id"], "text": s["text"]} for s in batch]
        user_prompt = f"Dịch danh sách các câu thoại sau sang {target_language}:\n" + json.dumps(input_payload, ensure_ascii=False)
        
        response_text = ""
        
        # 1. Google Gemini Provider
        if provider == "gemini":
            effective_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not effective_key:
                raise ValueError("Chưa cấu hình Gemini API Key!")
                
            payload = {
                "system_instruction": {
                    "parts": [{"text": system_instruction}]
                },
                "contents": [
                    {"parts": [{"text": user_prompt}]}
                ],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                    "maxOutputTokens": 8192
                }
            }
            
            # Dynamic fallback with priority for user's functional models
            discovered = get_available_gemini_models(effective_key)
            candidates = []
            if model and "2.5" not in model and "1.5" not in model:
                candidates.append(model)
            for m in ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]:
                if m in discovered and m not in candidates:
                    candidates.append(m)
            for m in discovered:
                if not any(b in m for b in ["tts", "image", "preview-tts", "lyria", "robotics", "computer-use", "banana", "2.5", "1.5"]):
                    if m not in candidates:
                        candidates.append(m)
            if not candidates:
                candidates = ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-2.0-flash"]
            
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                for cand in candidates:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cand}:generateContent?key={effective_key}"
                    resp = await http_client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                        print(f"[+] Dịch thuật thành công với mô hình: {cand}")
                        break
                if not response_text:
                    raise RuntimeError("Gemini API không thể hoàn tất dịch thuật với các mô hình khả dụng.")
            
        # 2. OpenAI / DeepSeek / Groq / Local OpenAI-compatible
        else:
            api_endpoint = base_url.rstrip("/") + "/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
            if provider == "deepseek" and not base_url:
                api_endpoint = "https://api.deepseek.com/v1/chat/completions"
                model = model or "deepseek-chat"
            elif provider == "groq" and not base_url:
                api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
                model = model or "llama-3.3-70b-versatile"
                
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"} if "gemini" not in provider else None
            }
            
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                r = await http_client.post(api_endpoint, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                response_text = data["choices"][0]["message"]["content"]
                
        # Parse translated JSON
        try:
            parsed = extract_json_data(response_text)
            if isinstance(parsed, dict):
                # sometimes models wrap as {"translations": [...]}
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                        
            translation_map = {item["id"]: item.get("translation", "") for item in parsed if "id" in item}
            
            for s in batch:
                if s["id"] in translation_map:
                    s["translated_text"] = translation_map[s["id"]]
        except Exception as parse_err:
            print(f"Error parsing translation output: {parse_err}, response was: {response_text}")
            # Fallback: keep original if failed
            for s in batch:
                if not s.get("translated_text"):
                    s["translated_text"] = s["text"]

    if progress_callback:
        progress_callback(1.0, f"Đã dịch hoàn tất {total_subs} câu phụ đề!")
        
    return subtitles
