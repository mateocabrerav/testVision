"""Example usage of Gemini Robotics."""

import os
from dotenv import load_dotenv
from PIL import Image
from src.gemini_vision_system import process_image


def main():
    """Demonstrate Gemini vision system usage."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: Set GEMINI_API_KEY in .env file")
        exit(1)

    # Example 1: Detect people
    image = Image.open("input_image.jpg")
    result = process_image(api_key, image, "people")
    result.save("output_people.jpg")
    print("✓ People detected: output_people.jpg")

    # Example 2: Detect bottles
    result = process_image(api_key, image, "bottles")
    result.save("output_bottles.jpg")
    print("✓ Bottles detected: output_bottles.jpg")


if __name__ == "__main__":
    main()
