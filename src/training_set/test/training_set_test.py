"""Test for training set generation."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training_set.generate_training_set import create_training_set


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not set")
        sys.exit(1)
    
    # Example usage
    input_image = str(project_root / "src" / "gemini" / "test" / "test_image.png")
    instruction = "mujeres en la imagen"
    
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    if len(sys.argv) > 2:
        instruction = sys.argv[2]
    
    create_training_set(api_key, input_image, instruction)
