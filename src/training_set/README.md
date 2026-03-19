# Training Set Generator

Automatically generate augmented training datasets from Gemini Robotics detections.

## Features

This script detects objects using Gemini Robotics API and creates a diverse training set with **16 augmentations per detection**:

### Transformations Applied:
- **Rotation**: ±15°, ±30°
- **Flipping**: Horizontal & Vertical
- **Brightness**: Darker (0.7x) & Brighter (1.3x)
- **Contrast**: Lower (0.7x) & Higher (1.3x)
- **Blur**: Gaussian blur
- **Noise**: Random gaussian noise
- **Zoom**: Zoom in (1.2x) & Zoom out (0.8x)
- **Color**: Saturation & Desaturation
- **Original**: Unmodified image

## Usage

### Basic Usage:
```bash
python training_set/generate_training_set.py
```

### With Custom Image:
```bash
python training_set/generate_training_set.py path/to/image.png "people wearing hats"
```

### From Python:
```python
from training_set.generate_training_set import create_training_set

create_training_set(
    api_key="your_gemini_api_key",
    input_image_path="test/gemini/test_image.png",
    instruction="people in the image",
    output_dir="training_set/output"
)
```

## Output Structure

```
training_set/output/
├── person/
│   ├── det0_original.png
│   ├── det0_rotate_15.png
│   ├── det0_flip_h.png
│   ├── det0_bright_1.3.png
│   └── ... (16 images per detection)
├── woman_with_curly_hair/
│   └── ... (16 images)
└── ...
```

Each detected object gets its own folder with all augmentations.

## Environment Setup

Ensure `GEMINI_API_KEY` is set in your `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

## Requirements

- Python 3.11+
- PIL/Pillow
- OpenCV (cv2)
- NumPy
- python-dotenv
- google-genai

All dependencies are in the project's `requirements.txt`.
