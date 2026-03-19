"""Use Gemini to classify instruction and determine target object class."""

import os
from typing import Optional, List
import json


# YOLO COCO classes mapping
YOLO_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13,
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'backpack': 24,
    'umbrella': 25,
    'handbag': 26,
    'tie': 27,
    'suitcase': 28,
    'frisbee': 29,
    'skis': 30,
    'snowboard': 31,
    'sports ball': 32,
    'kite': 33,
    'baseball bat': 34,
    'baseball glove': 35,
    'skateboard': 36,
    'surfboard': 37,
    'tennis racket': 38,
    'bottle': 39,
    'wine glass': 40,
    'cup': 41,
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'banana': 46,
    'apple': 47,
    'sandwich': 48,
    'orange': 49,
    'broccoli': 50,
    'carrot': 51,
    'hot dog': 52,
    'pizza': 53,
    'donut': 54,
    'cake': 55,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,
    'bed': 59,
    'dining table': 60,
    'toilet': 61,
    'tv': 62,
    'laptop': 63,
    'mouse': 64,
    'remote': 65,
    'keyboard': 66,
    'cell phone': 67,
    'microwave': 68,
    'oven': 69,
    'toaster': 70,
    'sink': 71,
    'refrigerator': 72,
    'book': 73,
    'clock': 74,
    'vase': 75,
    'scissors': 76,
    'teddy bear': 77,
    'hair drier': 78,
    'toothbrush': 79
}


def classify_instruction(api_key: str, instruction: str) -> Optional[List[int]]:
    """Use Gemini to determine which YOLO classes to track based on instruction.
    
    Args:
        api_key: Gemini API key
        instruction: Natural language instruction (e.g., "el auto rojo")
    
    Returns:
        List of YOLO class IDs to track, or None for all classes
    """
    # Create prompt with YOLO classes
    class_list = ', '.join(YOLO_CLASSES.keys())
    
    prompt = f"""
Given the instruction: "{instruction}"

Determine which of these YOLO object classes are relevant:
{class_list}

Return ONLY a JSON object with this format:
{{"classes": ["class1", "class2", ...]}}

Rules:
- If the instruction mentions a specific object type (car, person, dog, etc.), return only that class
- If it's a general description without a specific object type, return null for all classes
- Use the exact class names from the list above
- Return ONLY JSON, no explanation

Examples:
"el auto rojo" → {{"classes": ["car"]}}
"la persona con lentes" → {{"classes": ["person"]}}
"el perro marrón" → {{"classes": ["dog"]}}
"autos y motos" → {{"classes": ["car", "motorcycle"]}}
"cualquier cosa roja" → {{"classes": null}}
"""

    try:
        # Use direct REST API to avoid library version issues
        import requests
        
        # Use gemini-2.0-flash which is available
        model_name = "gemini-2.0-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code != 200:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            return None
            
        result_json = response.json()
        
        # Extract text from response
        try:
            text = result_json['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            print("Unexpected Gemini response format")
            return None
        
        # Clean markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Parse JSON
        data = json.loads(text)
        
        if data.get('classes') is None:
            print("Gemini suggests: Track ALL object types")
            return None
        
        class_names = data.get('classes', [])
        
        if not class_names:
            print("No specific class found, tracking ALL")
            return None
        
        # Convert class names to IDs
        class_ids = []
        for name in class_names:
            if name in YOLO_CLASSES:
                class_ids.append(YOLO_CLASSES[name])
        
        if class_ids:
            class_names_str = ', '.join(class_names)
            print(f"Gemini determined target classes: {class_names_str} (IDs: {class_ids})")
            return class_ids
        else:
            print("No valid YOLO classes found, tracking ALL")
            return None
            
    except Exception as e:
        print(f"Error classifying instruction: {e}")
        print("Defaulting to track ALL object types")
        return None


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    test_instructions = [
        "el auto rojo",
        "la persona con lentes",
        "el perro marrón",
        "autos y motos rojos",
        "cualquier cosa que se mueva"
    ]
    
    for instruction in test_instructions:
        print(f"\nTesting: '{instruction}'")
        result = classify_instruction(api_key, instruction)
        print(f"   Result: {result}")
