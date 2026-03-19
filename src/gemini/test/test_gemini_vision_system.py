"""Test for Gemini Robotics vision system with real image."""

import pytest
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gemini.gemini_vision_system import process_image


class TestGeminiVisionSystem:
    """Test suite for Gemini vision system with real API calls."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment or skip."""
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY not set")
        return key

    @pytest.fixture
    def test_image(self):
        """Load requested test image: test_image.png."""
        image_path = os.path.join(os.path.dirname(__file__), "test_image.png")
        if not os.path.exists(image_path):
            pytest.skip(f"Test image not found: {image_path}")
        return Image.open(image_path)

    def test_select_curly_haired_woman(self, api_key, test_image):
        """Instruction: selecciona el hombre de camiza azul."""
        result = process_image(
             api_key, test_image, "Instruction: selecciona solamente al hombre sin barba, trata de que no haya ninguna persona mas en el recuadro"
        )

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

        output_path = os.path.join(
            os.path.dirname(__file__), "output_tst_image_result.png"
        )
        result.save(output_path)
        print(f"\n✓ Result saved to: {output_path}")


if __name__ == "__main__":
    """Run test directly without pytest."""
    test = TestGeminiVisionSystem()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not set in environment")
        sys.exit(1)
    
    # Load test image
    image_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    if not os.path.exists(image_path):
        print(f"Error: Test image not found: {image_path}")
        sys.exit(1)
    
    test_image = Image.open(image_path)
    
    # Run test
    print("Running test: selecciona a la mujer de cabello risado")
    print(f"Image: {image_path}")
    
    try:
        result = process_image(
            api_key, test_image, "Instruction: selecciona a la mujer de cabello risado"
        )
        
        output_path = os.path.join(
            os.path.dirname(__file__), "output_tst_image_result.png"
        )
        result.save(output_path)
        print(f"Success! Result saved to: {output_path}")
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)
