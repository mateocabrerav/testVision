"""Quick start - Gemini Robotics detection."""

import os
from dotenv import load_dotenv
from PIL import Image
from src.gemini_vision_system import process_image


def main() -> None:
    """Run a minimal Gemini Robotics detection example."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY in .env file")
        return

    image = Image.open("your_image.jpg")
    result = process_image(api_key, image, "people")
    result.save("output.jpg")
    print("Saved output.jpg")


if __name__ == "__main__":
    main()
