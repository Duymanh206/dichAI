import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from typing import List, Dict, Any, Optional, Callable
import difflib

# Initialize RapidOCR singleton
_ocr_instance: Optional[RapidOCR] = None

def get_ocr() -> RapidOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = RapidOCR()
    return _ocr_instance

def is_text_similar(t1: str, t2: str, threshold: float = 0.7) -> bool:
    """Check if two strings are similar enough to belong to the same subtitle sentence."""
    t1_clean = "".join(t1.split())
    t2_clean = "".join(t2.split())
    if not t1_clean and not t2_clean:
        return True
    if not t1_clean or not t2_clean:
        return False
    if t1_clean == t2_clean:
        return True
    len_ratio = min(len(t1_clean), len(t2_clean)) / max(len(t1_clean), len(t2_clean))
    if len_ratio < 0.5:
        return False
    matcher = difflib.SequenceMatcher(None, t1_clean, t2_clean)
    return matcher.ratio() >= threshold

def extract_subtitles_from_video(
    video_path: str,
    crop_box: Optional[Dict[str, Any]] = None,
    fps_sample: float = 1.8,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    High-speed subtitle OCR using sequential frame grabbing and visual change detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
        
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / video_fps if video_fps > 0 else 0
    
    # Calculate crop coordinates in pixels
    if crop_box:
        if "x_pct" in crop_box:
            x = int(crop_box["x_pct"] * video_width)
            y = int(crop_box["y_pct"] * video_height)
            w = int(crop_box["w_pct"] * video_width)
            h = int(crop_box["h_pct"] * video_height)
        else:
            x = int(crop_box.get("x", 0))
            y = int(crop_box.get("y", int(video_height * 0.75)))
            w = int(crop_box.get("w", video_width))
            h = int(crop_box.get("h", int(video_height * 0.2)))
    else:
        x = int(video_width * 0.05)
        y = int(video_height * 0.75)
        w = int(video_width * 0.90)
        h = int(video_height * 0.20)
        
    x = max(0, min(x, video_width - 10))
    y = max(0, min(y, video_height - 10))
    w = max(10, min(w, video_width - x))
    h = max(10, min(h, video_height - y))

    frame_interval = max(1, int(video_fps / fps_sample))
    ocr = get_ocr()

    raw_detections = []
    prev_crop_gray = None
    last_text = ""
    last_score = 0.0
    
    frame_idx = 0
    while cap.isOpened():
        # Sequential grab is 20-50x faster than seeking
        if frame_idx % frame_interval != 0:
            grabbed = cap.grab()
            if not grabbed:
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        time_sec = frame_idx / video_fps
        if progress_callback and total_duration > 0 and (frame_idx % (frame_interval * 4) == 0):
            progress_callback(min(0.95, time_sec / total_duration), f"Đang quét OCR: {time_sec:.1f}s / {total_duration:.1f}s")
            
        crop_img = frame[y:y+h, x:x+w]
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Check if crop has changed from previous sample
        is_same_frame = False
        if prev_crop_gray is not None and prev_crop_gray.shape == crop_gray.shape:
            diff = cv2.absdiff(crop_gray, prev_crop_gray)
            if np.mean(diff) < 3.0: # Very small difference -> identical scene/sub
                is_same_frame = True
                
        prev_crop_gray = crop_gray
        
        if is_same_frame and last_text:
            raw_detections.append((time_sec, last_text, last_score))
        else:
            # Run OCR only when there's visual change
            ocr_result, _ = ocr(crop_img)
            if ocr_result:
                texts = [item[1] for item in ocr_result if item[1] and len(item[1].strip()) > 0]
                if texts:
                    combined_text = " ".join(texts).strip()
                    avg_score = float(np.mean([item[2] for item in ocr_result]))
                    raw_detections.append((time_sec, combined_text, avg_score))
                    last_text = combined_text
                    last_score = avg_score
                else:
                    last_text = ""
            else:
                last_text = ""
        
        frame_idx += 1
        if total_frames > 0 and frame_idx >= total_frames:
            break
            
    cap.release()
    
    if progress_callback:
        progress_callback(0.98, "Đang tối ưu và gom cụm câu phụ đề...")
        
    grouped_subs: List[Dict[str, Any]] = []
    if not raw_detections:
        return grouped_subs
        
    current_start = raw_detections[0][0]
    current_end = raw_detections[0][0] + (1.0 / fps_sample)
    current_text = raw_detections[0][1]
    scores = [raw_detections[0][2]]
    step_duration = 1.0 / fps_sample
    
    for i in range(1, len(raw_detections)):
        t_sec, text, score = raw_detections[i]
        time_gap = t_sec - current_end
        
        if is_text_similar(current_text, text) and time_gap <= 0.9:
            current_end = t_sec + step_duration
            scores.append(score)
            if len(text) > len(current_text):
                current_text = text
        else:
            sub_len = current_end - current_start
            if sub_len >= 0.3 and len(current_text.strip()) > 0:
                grouped_subs.append({
                    "id": len(grouped_subs) + 1,
                    "start": round(current_start, 2),
                    "end": round(min(total_duration, current_end), 2),
                    "text": current_text.strip(),
                    "translated_text": "",
                    "audio_file": "",
                    "confidence": round(float(np.mean(scores)), 2)
                })
            current_start = t_sec
            current_end = t_sec + step_duration
            current_text = text
            scores = [score]
            
    if len(current_text.strip()) > 0:
        grouped_subs.append({
            "id": len(grouped_subs) + 1,
            "start": round(current_start, 2),
            "end": round(min(total_duration, current_end), 2),
            "text": current_text.strip(),
            "translated_text": "",
            "audio_file": "",
            "confidence": round(float(np.mean(scores)), 2)
        })

    if progress_callback:
        progress_callback(1.0, f"Hoàn thành OCR: Tìm thấy {len(grouped_subs)} đoạn phụ đề.")
        
    return grouped_subs
