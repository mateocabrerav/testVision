# Selective Person Tracker

Track only specific people that match your training set images, ignoring all others.

## How It Works

1. **Load Training Set**: Loads reference images from your training set
2. **Detect All People**: Uses YOLOv8 to detect all people in video
3. **Feature Matching**: Compares each detected person with training set using color histograms
4. **Selective Tracking**: Only tracks and counts people that match (similarity > threshold)

## Features

✅ **Selective Tracking** - Only tracks people matching training set
✅ **Visual Feedback** - Green boxes for matches, red for non-matches
✅ **Similarity Scores** - Shows match confidence for each person
✅ **Result Logging** - Saves tracking results with timestamps
✅ **Real-time Display** - Shows matched person count on video

## Usage

### Basic Usage:
```bash
python test/tracker_test/test_selective_tracker.py
```

### From Python:
```python
from src.person_tracker.selective_person_tracker import SelectivePersonTracker

tracker = SelectivePersonTracker(
    model_path='yolov8n.pt',
    training_set_dir='src/training_set/output/people',
    match_threshold=0.6,  # Adjust based on results
    device='cuda:0'
)

tracker.detect_and_track(source=0)  # 0 = webcam
```

### Parameters:

- `training_set_dir`: Path to your training images
- `match_threshold`: Similarity threshold (0.0-1.0, higher = stricter)
  - `0.5`: Loose matching (may include false positives)
  - `0.6-0.7`: Balanced (recommended)
  - `0.8+`: Very strict (may miss some matches)
- `device`: `'cuda:0'` for GPU, `'cpu'` for CPU
- `conf`: YOLO detection confidence (0.0-1.0)

## Visual Output

- **Green Box**: Person matches training set ✅
- **Red Box**: Person does NOT match ❌
- **Label**: Shows ID and similarity score

## Results

Results are saved to `results/` with:
- Timestamp
- Frame number
- Matched person count
- Track IDs
- Similarity scores

## Workflow

1. **Generate Training Set**:
   ```bash
   python src/training_set/generate_training_set.py
   ```

2. **Run Selective Tracker**:
   ```bash
   python test/tracker_test/test_selective_tracker.py
   ```

## Adjusting Accuracy

If you get:
- **Too many false positives**: Increase `match_threshold` (e.g., 0.7 → 0.8)
- **Missing matches**: Decrease `match_threshold` (e.g., 0.7 → 0.6)
- **Need more training data**: Run training set generator on more images

## Requirements

- YOLOv8 (ultralytics)
- OpenCV
- NumPy
- CUDA (optional, for GPU acceleration)
