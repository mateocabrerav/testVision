"""Test selective person tracker with training set."""

import os
import sys
from pathlib import Path

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.person_tracker.selective_person_tracker import SelectivePersonTracker
from src.realsense_camera import get_realsense_camera


if __name__ == '__main__':
    # Configuration
    training_set_dir = 'training_set/output/person_(0.84)'  # Directory with your training images
    yolo_model = 'yolov8n.pt'  # YOLOv8 model (will auto-download if not exists)
    
    # Try to use RealSense camera first
    print("🔍 Checking for RealSense camera...")
    rs_camera = get_realsense_camera()
    
    if rs_camera:
        print("✅ RealSense camera found! Using it for Color + Depth tracking.")
        source = rs_camera
    else:
        print("⚠️ RealSense not found. Falling back to standard webcam.")
        source = 2  # Camera index 2 (working webcam on this system)
    
    # Check if training set exists
    if not os.path.exists(training_set_dir):
        print(f"❌ Training set not found at: {training_set_dir}")
        print("Run generate_training_set.py first to create training data")
        sys.exit(1)
    
    print("🚀 Starting Selective Person Tracker")
    print(f"📁 Training set: {training_set_dir}")
    print(f"🎥 Source: {source}")
    print("\nPress 'q' to quit\n")
    
    # Create tracker
    tracker = SelectivePersonTracker(
        model_path=yolo_model,
        training_set_dir=training_set_dir,
        match_threshold=0.6,  # Adjust: higher = stricter matching
        device='cuda:0',  # Change to 'cpu' if no GPU
        conf=0.5,  # YOLO confidence threshold
    )
    
    # Run tracking
    tracker.detect_and_track(source=source, show=True)
