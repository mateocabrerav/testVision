"""Generate augmented training set from Gemini Robotics detections."""

import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gemini.gemini_vision_system import process_image


def extract_bbox_from_detection(image: Image.Image, box_2d: list) -> Image.Image:
    """Extract bounding box region from image."""
    h, w = image.size[1], image.size[0]
    y1, x1, y2, x2 = box_2d
    px1, py1 = int(x1 * w / 1000), int(y1 * h / 1000)
    px2, py2 = int(x2 * w / 1000), int(y2 * h / 1000)
    return image.crop((px1, py1, px2, py2))


def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image by angle degrees."""
    return img.rotate(angle, expand=True, fillcolor=(0, 0, 0))


def apply_flip(img: Image.Image, horizontal: bool = True) -> Image.Image:
    """Flip image horizontally or vertically."""
    return img.transpose(Image.FLIP_LEFT_RIGHT if horizontal else Image.FLIP_TOP_BOTTOM)


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness (factor: 0.5=darker, 1.5=brighter)."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast (factor: 0.5=less, 1.5=more)."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_blur(img: Image.Image, radius: int = 2) -> Image.Image:
    """Apply gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius))


def apply_noise(img: Image.Image, intensity: int = 25) -> Image.Image:
    """Add gaussian noise."""
    img_array = np.array(img)
    noise = np.random.normal(0, intensity, img_array.shape).astype(np.uint8)
    noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_zoom(img: Image.Image, factor: float) -> Image.Image:
    """Zoom in/out (factor: 0.8=zoom out, 1.2=zoom in)."""
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    if factor > 1.0:  # Zoom in - crop center
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    else:  # Zoom out - pad
        new_img = Image.new('RGB', (w, h), (0, 0, 0))
        new_img.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
        return new_img


def apply_color_jitter(img: Image.Image, saturation: float = 1.3) -> Image.Image:
    """Adjust color saturation."""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(saturation)


def generate_augmentations(image: Image.Image) -> dict:
    """Generate all augmentation variants."""
    return {
        'original': image,
        'rotate_15': apply_rotation(image, 15),
        'rotate_neg_15': apply_rotation(image, -15),
        'rotate_30': apply_rotation(image, 30),
        'flip_h': apply_flip(image, horizontal=True),
        'flip_v': apply_flip(image, horizontal=False),
        'bright_1.3': apply_brightness(image, 1.3),
        'bright_0.7': apply_brightness(image, 0.7),
        'contrast_1.3': apply_contrast(image, 1.3),
        'contrast_0.7': apply_contrast(image, 0.7),
        'blur': apply_blur(image, 2),
        'noise': apply_noise(image, 20),
        'zoom_in_1.2': apply_zoom(image, 1.2),
        'zoom_out_0.8': apply_zoom(image, 0.8),
        'saturate_1.5': apply_color_jitter(image, 1.5),
        'desaturate_0.5': apply_color_jitter(image, 0.5),
    }


def create_training_set(
    api_key: str,
    input_image_path: str,
    instruction: str,
    output_dir: str = "training_set/output"
) -> None:
    """
    Create augmented training set from detected objects.
    
    Args:
        api_key: Gemini API key
        input_image_path: Path to input image
        instruction: Detection instruction
        output_dir: Output directory for training images
    """
    # Load and process image
    print(f"Processing image: {input_image_path}")
    print(f"Instruction: {instruction}")
    
    # Clean up previous output
    if os.path.exists(output_dir):
        print(f"Removing previous training set from: {output_dir}")
        shutil.rmtree(output_dir)
    
    image = Image.open(input_image_path)
    
    # Get detections from Gemini
    from src.gemini.gemini_detector import detect_objects
    detections = detect_objects(api_key, image, instruction)
    
    if not detections:
        print("No detections found!")
        return
    
    print(f"Found {len(detections)} detection(s)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each detection
    for idx, detection in enumerate(detections):
        label = detection.get('label', 'unknown')
        box_2d = detection.get('box_2d') or detection.get('box')
        
        if not box_2d:
            print(f"Skipping detection {idx}: no bounding box")
            continue
        
        print(f"\nProcessing detection {idx}: {label}")
        
        # Extract bounding box region
        bbox_img = extract_bbox_from_detection(image, box_2d)
        
        # Create label directory
        label_dir = os.path.join(output_dir, label.replace(" ", "_"))
        os.makedirs(label_dir, exist_ok=True)
        
        # Generate augmentations
        augmentations = generate_augmentations(bbox_img)
        
        # Save all variants
        for aug_name, aug_img in augmentations.items():
            filename = f"det{idx}_{aug_name}.png"
            filepath = os.path.join(label_dir, filename)
            aug_img.save(filepath)
        
        print(f"   Saved {len(augmentations)} augmented images to {label_dir}")
    
    print(f"\nTraining set created successfully in: {output_dir}")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)
    
    # Example usage
    input_image = "test/gemini/test_image.png"
    instruction = "people in the image"
    
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    if len(sys.argv) > 2:
        instruction = sys.argv[2]
    
    create_training_set(api_key, input_image, instruction)

