"""Gemini Robotics detection using direct API with dual-SDK support.

Prefers google-genai (from google import genai). If unavailable,
attempts google-generativeai (import google.generativeai as genai).
SDK imports are deferred to call time to avoid ImportError during import.
Falls back to YOLOv8 when Gemini APIs are unavailable.
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import json
import io
import os

# Preferred model order (user-requested model first)
MODEL_CANDIDATES = (
    os.getenv("GEMINI_MODEL", "gemini-robotics-er-1.5-preview"),
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-pro-vision",
)

# Optional debug logging
_DEF_DEBUG = os.getenv("GEMINI_DEBUG")

def _dbg(msg: str) -> None:
    if _DEF_DEBUG:
        print(f"[gemini] {msg}")

# Flag indicating if new SDK is detected at import time (best-effort)
USE_GOOGLE_GENAI = False
try:  # best-effort probe only; we'll import again at call time
    from google import genai as _probe_genai  # type: ignore  # noqa: F401
    USE_GOOGLE_GENAI = True
except Exception:
    USE_GOOGLE_GENAI = False


def _detect_with_yolo_fallback(
    image: Image.Image, instruction: str, max_items: int
) -> List[Dict[str, Any]]:
    """Fallback to YOLOv8 when Gemini is unavailable."""
    try:
        from ultralytics import YOLO
        import numpy as np
    except Exception as e:
        _dbg(f"YOLOv8 fallback unavailable: {e}")
        return []

    _dbg("Using YOLOv8 fallback for detection")

    try:
        # Load YOLOv8 model
        model_path = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
        if not os.path.exists(model_path):
            _dbg(f"YOLOv8 model not found at {model_path}")
            return []

        model = YOLO(model_path)

        # Run inference
        results = model(image, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                if i >= max_items:
                    break

                # Get class
                cls = int(box.cls[0])
                class_name = result.names[cls]

                # Filter based on instruction if it mentions specific objects
                instruction_lower = instruction.lower()
                if "person" in instruction_lower or "people" in instruction_lower or "hombre" in instruction_lower or "mujer" in instruction_lower:
                    if class_name != "person":
                        continue

                # Get bounding box in xyxy format
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Get image dimensions
                h, w = image.size[1], image.size[0]

                # Convert to normalized 0-1000 format [y1, x1, y2, x2]
                norm_box = [
                    int(y1 / h * 1000),
                    int(x1 / w * 1000),
                    int(y2 / h * 1000),
                    int(x2 / w * 1000)
                ]

                conf = float(box.conf[0])

                detections.append({
                    "box_2d": norm_box,
                    "label": f"{class_name} ({conf:.2f})"
                })

        _dbg(f"YOLOv8 found {len(detections)} detections")
        return detections

    except Exception as e:
        _dbg(f"YOLOv8 fallback error: {e}")
        return []


def _detect_with_google_genai(
    api_key: str, image: Image.Image, instruction: str, max_items: int
) -> List[Dict[str, Any]]:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        _dbg("google-genai not available")
        return []

    client = genai.Client(api_key=api_key)

    prompt = f"""
    Detect no more than {max_items} {instruction} in the image.
    Return bounding boxes with descriptive labels.
    Answer in JSON: [{{"box_2d": <box>, "label": <label>}}, ...]
    Box format: [y1, x1, y2, x2] where (y1,x1) is top-left and (y2,x2) is bottom-right.
    All coordinates normalized to 0-1000.
    """

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    # Try preferred models in order
    for model_name in MODEL_CANDIDATES:
        _dbg(f"google-genai: trying model {model_name}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes.getvalue(), mime_type="image/png"
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as e:
            _dbg(f"google-genai call failed: {e}")
            response = None
        if not response:
            continue
        try:
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text)
            _dbg(f"google-genai: got detections ({len(data) if isinstance(data, list) else 'dict'})")
            return data if isinstance(data, list) else data.get("objects", [])
        except Exception as e:
            _dbg(f"google-genai parse failed: {e}")
            continue

    _dbg("google-genai: no detections")
    return []


def _detect_with_generativeai(
    api_key: str, image: Image.Image, instruction: str, max_items: int
) -> List[Dict[str, Any]]:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        _dbg("google-generativeai not available")
        return []

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        _dbg(f"google-generativeai configure failed: {e}")
        return []

    # Check for new API (GenerativeModel)
    ModelCls = getattr(genai, "GenerativeModel", None)
    if ModelCls is not None:
        # New API with GenerativeModel
        prompt = f"""
        You are an assistant that returns ONLY JSON, no prose.
        In the given image, detect no more than {max_items} {instruction}.
        Return JSON list: [{{"box_2d": [y1, x1, y2, x2], "label": "..."}}, ...]
        Box format: [y1, x1, y2, x2] where (y1,x1) is top-left and (y2,x2) is bottom-right.
        All coordinates must be normalized to 0-1000 integers.
        """

        # Try preferred models in order
        for model_name in MODEL_CANDIDATES:
            _dbg(f"google-generativeai: trying model {model_name}")
            try:
                model = ModelCls(model_name)
            except Exception as e:
                _dbg(f"model init failed: {e}")
                model = None
            if not model:
                continue
            try:
                response = model.generate_content(
                    [prompt, image],
                    generation_config={"response_mime_type": "application/json", "temperature": 0.0},
                )
            except Exception as e:
                _dbg(f"generate_content failed: {e}")
                continue
            try:
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                data = json.loads(text)
                if isinstance(data, dict) and "objects" in data:
                    _dbg(f"google-generativeai: got detections (dict)")
                    return data["objects"]
                if isinstance(data, list):
                    _dbg(f"google-generativeai: got detections ({len(data)})")
                    return data
            except Exception as e:
                _dbg(f"parse failed: {e}")
                continue
    else:
        # Old API - try legacy methods
        _dbg("google-generativeai: using legacy API (no GenerativeModel)")

        # Check if we have the legacy vision models
        try:
            models = genai.list_models()
            _dbg(f"Available models: {[m.name for m in models]}")

            # Legacy API doesn't support vision well, return empty
            _dbg("Legacy API doesn't support vision tasks properly")
            return []
        except Exception as e:
            _dbg(f"list_models failed: {e}")
            return []

    _dbg("google-generativeai: no detections")
    return []


def _detect_with_rest_api(
    api_key: str, image: Image.Image, instruction: str, max_items: int
) -> Optional[List[Dict[str, Any]]]:
    """Detect objects using Gemini REST API directly.
    
    Returns:
        List of detections if successful (can be empty).
        None if API call failed.
    """
    import requests
    import base64
    
    _dbg("Using Gemini REST API")
    
    # Convert image to base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    
    prompt = f"""
    Detect no more than {max_items} {instruction} in the image.
    Return bounding boxes with descriptive labels.
    Answer in JSON: [{{"box_2d": <box>, "label": <label>}}, ...]
    Box format: [y1, x1, y2, x2] where (y1,x1) is top-left and (y2,x2) is bottom-right.
    All coordinates normalized to 0-1000.
    """
    
    # Try preferred models
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro-vision"]:
        _dbg(f"REST API: trying model {model_name}")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        
        data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            
            if response.status_code != 200:
                _dbg(f"REST API error {response.status_code}: {response.text}")
                continue
                
            result_json = response.json()
            
            # Extract text
            try:
                text = result_json['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                _dbg("REST API: Unexpected response format")
                continue
            
            # Clean markdown
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            text = text.strip()
            
            # Parse JSON
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                _dbg(f"REST API: Invalid JSON response: {text}")
                continue
            
            if isinstance(data, dict) and "objects" in data:
                return data["objects"]
            if isinstance(data, list):
                return data
            
            # If we got valid JSON but not list or dict with objects, return empty list?
            # Or maybe it's an error?
            _dbg(f"REST API: Unexpected JSON structure: {type(data)}")
            return []
                
        except Exception as e:
            _dbg(f"REST API exception: {e}")
            continue
            
    return None


def detect_objects(
    api_key: str, image: Image.Image, instruction: str, max_items: int = 10
) -> List[Dict[str, Any]]:
    """Detect objects using Gemini APIs with YOLOv8 fallback.

    Args:
        api_key: Google AI API key
        image: PIL Image to analyze
        instruction: e.g., "people", "bottles", or full natural instruction
        max_items: Maximum objects to detect

    Returns:
        List like [{"box_2d": [y1, x1, y2, x2], "label": "person"}, ...]
    """
    # Try REST API first (most robust)
    res = _detect_with_rest_api(api_key, image, instruction, max_items)
    if res is not None:
        return res

    # Try Gemini SDKs (legacy)
    if USE_GOOGLE_GENAI:
        _dbg("using google-genai SDK path")
        res = _detect_with_google_genai(api_key, image, instruction, max_items)
        if res:
            return res

    _dbg("using google-generativeai SDK path")
    res = _detect_with_generativeai(api_key, image, instruction, max_items)
    if res:
        return res

    # Fallback to YOLOv8 if Gemini is unavailable
    _dbg("Gemini unavailable, falling back to YOLOv8")
    return _detect_with_yolo_fallback(image, instruction, max_items)
