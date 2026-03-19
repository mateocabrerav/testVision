"""Gemini Robotics vision system."""

from PIL import Image
from .gemini_detector import detect_objects
from .bbox_drawer import draw_points


def _clean_instruction(instr: str) -> str:
    """Normalize instruction text for better model understanding."""
    if not instr:
        return ""
    s = instr.strip()
    if s.lower().startswith("instruction:"):
        s = s.split(":", 1)[1].strip()
    return s


def process_image(api_key: str, image: Image.Image, instruction: str) -> Image.Image:
    """
    Process image with Gemini Robotics.

    Args:
        api_key: Google AI API key
        image: PIL Image
        instruction: What to find (e.g., "people", "bottles")

    Returns:
        Image with detection points
    """
    instruction = _clean_instruction(instruction)
    detections = detect_objects(api_key, image, instruction)
    return draw_points(image, detections) if detections else image
