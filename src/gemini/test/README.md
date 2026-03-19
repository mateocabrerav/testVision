# Test Image for Gemini Vision System

Place your test image here as `test_image.png`.

This image should contain a woman with curly hair for the test:
"selecciona a la mujer de cabello rizado"

## Test Output

When you run the test, the result will be saved as `output_test_result.png` in this folder.
"""Test for Gemini Robotics vision system with real image."""

import pytest
import os
from PIL import Image
from src.gemini_vision_system import process_image


class TestGeminiVisionSystem:
    """Test suite for Gemini vision system with real API calls."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY not set")
        return key

    @pytest.fixture
    def test_image(self):
        """Load test image."""
        image_path = os.path.join(
            os.path.dirname(__file__), "test_image.png"
        )
        if not os.path.exists(image_path):
            pytest.skip(f"Test image not found: {image_path}")
        return Image.open(image_path)

    def test_select_curly_haired_woman(self, api_key, test_image):
        """Test: selecciona a la mujer de cabello rizado."""
        result = process_image(
            api_key, test_image, "selecciona a la mujer de cabello rizado"
        )
        
        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        
        output_path = os.path.join(
            os.path.dirname(__file__), "output_test_result.png"
        )
        result.save(output_path)
        print(f"\n✓ Result saved to: {output_path}")

