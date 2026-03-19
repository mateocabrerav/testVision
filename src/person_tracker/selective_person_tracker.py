"""Track only specific objects that match training set images.

This tracker can work with any object type (persons, cars, animals, etc.)
by using visual similarity matching against a training set.
"""

from ultralytics import YOLO
from datetime import datetime
import os
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import logging
import threading
from .tracking_callback import process_tracking_data

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

# Load environment variables
load_dotenv()

# Set OpenCV to use DirectShow backend on Windows for better webcam compatibility
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '1'


def get_best_device():
    """Auto-detect best available device (CUDA > CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            arch_string = f"sm_{cap[0]}{cap[1]}"
            arch_list = torch.cuda.get_arch_list()
            
            # The Jetson Orin is sm_87. Using generic torch on ARM64 lacks sm_87 support.
            if arch_list and arch_string not in arch_list and not any(a.startswith("sm_8") for a in arch_list):
                 # Fallback if no Ampere architecture is matched (actually sm_86/80 might run or not, let's just use try)
                 pass
            
            # A more robust check: YOLO will crash if PyTorch compiled arches don't have sm_87
            if "sm_87" not in arch_list and arch_string == "sm_87":
                log.warning(f"PyTorch lacks {arch_string} support (Jetson Orin). Falling back to CPU.")
                return 'cpu'

            device = 'cuda:0'
            gpu_name = torch.cuda.get_device_name(0)
            log.info(f"Using GPU: {gpu_name}")
            return device
    except:
        pass
    
    log.info("Using CPU")
    return 'cpu'
class SelectivePersonTracker:
    """Track only objects matching the training set.
    
    Can track any object type (persons, cars, animals, etc.) based on
    visual similarity with reference images from a training set.
    """
    
    def __init__(
        self,
        model_path: str,
        training_set_dir: str,
        result_dir: str = 'results/',
        tracker_config: str = "bytetrack.yaml",
        conf: float = 0.5,
        device: str = 'cuda:0',
        iou: float = 0.5,
        img_size: tuple = (480, 640),  # Reduced size for speed
        match_threshold: float = 0.7,
        skip_frames: int = 2,  # Process every Nth frame
        auto_improve: bool = False,
        max_training_images: int = 200,
        improvement_interval: int = 10,  # seconds
        target_classes: list = None  # None = all classes, [0] = person only, etc.
    ):
        self.model = YOLO(model_path)
        self.result_dir = result_dir
        self.tracker_config = tracker_config
        self.conf = conf
        self.device = device
        self.iou = iou
        self.img_size = img_size
        self.match_threshold = match_threshold
        self.skip_frames = skip_frames
        self.feature_cache = {}  # Cache computed features
        self.training_set_dir = training_set_dir
        self.target_classes = target_classes  # None = all classes
        
        # Use FP16 on CUDA for ~2x faster inference on Jetson
        self.half = 'cuda' in device
        
        # Auto-improvement settings
        self.auto_improve = auto_improve
        self.max_training_images = max_training_images
        self.improvement_interval = improvement_interval
        self.last_improvement_time = 0
        self.training_image_count = 0
        self.gemini_capture_count = 0  # Track how many Gemini captures done
        self.max_gemini_captures = 4  # Use Gemini for first 4 captures
        self.current_instruction = None  # Store current tracking instruction
        self.is_improving = False  # Flag for async improvement
        
        # Load reference features from training set
        self.reference_features = self._load_training_set_features(training_set_dir)
        self.training_image_count = len(self.reference_features)
        log.info(f"Loaded {self.training_image_count} reference images from training set")
    
    def reload_training_set(self):
        """Reload training set features and clear cache."""
        log.info("Reloading training set...")
        self.reference_features = self._load_training_set_features(self.training_set_dir)
        self.training_image_count = len(self.reference_features)
        self.feature_cache.clear()
        log.info(f"Reloaded {self.training_image_count} reference images | mean_ref={'yes' if self.mean_reference is not None else 'no'}")
    
    def restart_auto_improvement(self, instruction: str = None):
        """Restart auto-improvement process from scratch."""
        import time
        log.info("Restarting auto-improvement process...")
        self.auto_improve = True
        # Set in the past so improvement fires on the very next frame check
        self.last_improvement_time = time.time() - self.improvement_interval
        self._improve_started_at = 0.0  # watchdog: 0 means not running
        self.feature_cache.clear()
        self.gemini_capture_count = 0
        self.current_instruction = instruction
        log.info(f"Auto-improvement restarted | instruction='{instruction}'")
        log.info(f"Phase 1: Gemini for first {self.max_gemini_captures} captures (fires immediately then every {self.improvement_interval}s)")
        log.info(f"Phase 2: Direct detection until {self.max_training_images} images")
    
    def _generate_training_with_gemini(self, frame, instruction: str):
        """Generate training set from frame using Gemini detection."""
        import time
        import tempfile
        from training_set.generate_training_set import create_training_set
        
        log.info(f"Gemini capture {self.gemini_capture_count + 1}/{self.max_gemini_captures} starting | instruction='{instruction}'")
        
        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, frame)
        log.info(f"Saved temp frame to {temp_path} | shape={frame.shape}")
        
        try:
            # Get Gemini API key
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                log.error("GEMINI_API_KEY not found in environment")
                return 0
            
            # Use Gemini to detect and create training set (appends to existing)
            from gemini.gemini_detector import detect_objects
            from PIL import Image
            from training_set.generate_training_set import generate_augmentations, extract_bbox_from_detection
            
            pil_image = Image.open(temp_path)
            log.info(f"Calling Gemini detect_objects API...")
            t0 = time.time()
            detections = detect_objects(api_key, pil_image, instruction)
            elapsed = time.time() - t0
            log.info(f"Gemini API returned in {elapsed:.1f}s | detections={len(detections) if detections else 0}")
            
            if not detections:
                log.warning("Gemini found no detections")
                return 0
            
            # Process first detection only
            detection = detections[0]
            box_2d = detection.get('box_2d') or detection.get('box')
            log.info(f"First detection: label='{detection.get('label')}' box={box_2d}")
            
            if not box_2d:
                log.warning("No bounding box found in detection")
                return 0
            
            # Extract and augment
            bbox_img = extract_bbox_from_detection(pil_image, box_2d)
            augmentations = generate_augmentations(bbox_img)
            
            # Save to gemini subdirectory
            label_dir = os.path.join(self.training_set_dir, f"gemini_capture_{self.gemini_capture_count + 1}")
            os.makedirs(label_dir, exist_ok=True)
            
            saved_count = 0
            for aug_name, aug_img in augmentations.items():
                filename = f"{aug_name}.png"
                filepath = os.path.join(label_dir, filename)
                aug_img.save(filepath)
                saved_count += 1
            
            log.info(f"Saved {saved_count} Gemini-generated images to {label_dir}")
            self.gemini_capture_count += 1
            return saved_count
            
        except Exception as e:
            log.error(f"Gemini training generation failed: {e}", exc_info=True)
            return 0
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _generate_augmented_training_images(self, crop_image, instruction: str):
        """Generate augmented training images from a crop."""
        import time
        from PIL import Image
        from training_set.generate_training_set import generate_augmentations
        
        log.info(f"Generating augmented images from direct detection crop")
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        
        # Generate augmentations
        augmentations = generate_augmentations(pil_image)
        
        # Create label directory
        label_dir = os.path.join(self.training_set_dir, "auto_improved")
        os.makedirs(label_dir, exist_ok=True)
        
        # Count existing images
        existing_count = len(list(Path(label_dir).glob("*.png")))
        log.info(f"Existing images in auto_improved: {existing_count}")
        
        # Save augmented images
        saved_count = 0
        for aug_name, aug_img in augmentations.items():
            if existing_count + saved_count >= self.max_training_images:
                break
            
            timestamp = int(time.time() * 1000)
            filename = f"improve_{timestamp}_{aug_name}.png"
            filepath = os.path.join(label_dir, filename)
            aug_img.save(filepath)
            saved_count += 1
        
        log.info(f"Saved {saved_count} new augmented images to {label_dir}")
        return saved_count

    def _load_training_set_features(self, training_set_dir: str) -> list:
        """Load and compute features from training set images."""
        features = []
        training_path = Path(training_set_dir)
        
        if not training_path.exists():
            log.warning(f"Training set directory not found: {training_set_dir}")
            return features
        
        # Load all images from training set
        png_files = list(training_path.rglob("*.png"))
        log.info(f"Found {len(png_files)} .png files in {training_set_dir}")
        for img_file in png_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                hist = self._compute_histogram_features(img)
                features.append(hist)
            else:
                log.warning(f"Failed to read image: {img_file}")

        # Pre-compute mean reference for fast O(1) quick-reject
        if features:
            self.mean_reference = np.mean(features, axis=0).astype(np.float32)
        else:
            self.mean_reference = None

        return features

    def _compute_histogram_features(self, img: np.ndarray) -> np.ndarray:
        """Compute spatial color histogram (upper/lower body split)."""
        h, w = img.shape[:2]
        if h > 10 and w > 10:
            img = img[int(h*0.05):int(h*0.95), int(w*0.1):int(w*0.9)]

        img_resized = cv2.resize(img, (64, 128))
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        # Split into upper body (shirt) and lower body (pants)
        mid = hsv.shape[0] // 2
        parts = []
        for region in (hsv[:mid], hsv[mid:]):
            hist_hs = cv2.calcHist([region], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist_hs, hist_hs)
            parts.append(hist_hs.flatten())

        return np.concatenate(parts)

    def _matches_training_set(self, person_crop: np.ndarray, track_id: int = None) -> float:
        """Two-stage matching: quick-reject via mean reference, then full comparison."""
        if len(self.reference_features) == 0:
            return 0.0

        person_hist = self._compute_histogram_features(person_crop).astype(np.float32)

        # Stage 1: Quick-reject against the pre-computed mean reference (O(1))
        if self.mean_reference is not None:
            mean_sim = cv2.compareHist(person_hist, self.mean_reference, cv2.HISTCMP_CORREL)
            if mean_sim < self.match_threshold - 0.15:
                # Clearly not the target — skip expensive full comparison
                if track_id is not None:
                    self.feature_cache[track_id] = mean_sim
                return mean_sim

        # Stage 2: Full comparison against all reference images
        max_similarity = 0.0
        for ref_hist in self.reference_features:
            similarity = cv2.compareHist(
                person_hist,
                ref_hist.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            max_similarity = max(max_similarity, similarity)

        # Cache with decay to allow ID recovery
        if track_id is not None:
            if track_id in self.feature_cache:
                max_similarity = max(max_similarity, self.feature_cache[track_id] * 0.95)
            self.feature_cache[track_id] = max_similarity

        return max_similarity

    def create_result_file(self) -> str:
        """Create timestamped result file."""
        folder_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_file_path = os.path.join(self.result_dir, folder_name + ".txt")
        os.makedirs(self.result_dir, exist_ok=True)
        with open(result_file_path, 'w') as file:
            file.write(f"Selective Person Tracking - {folder_name}\n")
            file.write(f"Match threshold: {self.match_threshold}\n\n")
        return result_file_path

    def detect_and_track(self, source, show: bool = True, logger=None, frame_callback=None):
        """
        Detect and track only people matching training set.
        
        Args:
            source: Video source (file path, camera index, URL, or RealSenseCamera object)
            show: Display video with detections
            logger: Optional logger for output
            frame_callback: Optional callback function(frame) called on each frame
        """
        # Check if source is a camera object (RealSense or OpenCV fallback)
        is_camera_object = False
        # Duck typing: any object with read() + release() is a camera
        if hasattr(source, 'read') and hasattr(source, 'release') and callable(source.read):
            is_camera_object = True

        # If source is a camera index, configure it properly
        if isinstance(source, int):
            import cv2
            cap = cv2.VideoCapture(source)  # No DirectShow on Linux/Jetson
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            # Test if camera works
            ret, _ = cap.read()
            cap.release()
            
            if not ret:
                log.error(f"Cannot access camera {source}")
                return

        # If camera object, handle frame reading manually
        if is_camera_object:
            log.info("Using camera object for tracking")
            self._track_with_realsense(source, show, logger, frame_callback)
            return

        # Continue with regular YOLO tracking for non-RealSense sources
        result_file = self.create_result_file()
        matched_person_count = 0
        previous_count = 0
        matched_ids = set()
        last_annotated_frame = None  # Store last processed frame with annotations
        
        # Auto-improvement tracking
        import time
        self.last_improvement_time = time.time()
        best_match_crop = None  # Store best matching crop for improvement
        
        # Store current target classes to detect changes
        current_target_classes = self.target_classes

        # Build tracking parameters with current classes
        def get_track_params():
            params = {
                'source': source,
                'show': False,
                'stream': True,
                'tracker': self.tracker_config,
                'conf': self.conf,
                'device': self.device,
                'iou': self.iou,
                'stream_buffer': True,
                'imgsz': self.img_size,
                'verbose': False,
                'half': self.half
            }
            
            # Add classes filter only if specified
            if self.target_classes is not None:
                params['classes'] = self.target_classes
                class_names = [self.model.names.get(c, str(c)) for c in self.target_classes]
                log.info(f"Tracking classes: {class_names} (IDs: {self.target_classes})")
            else:
                log.info("Tracking ALL detectable objects")
            
            return params
        
        results = self.model.track(**get_track_params())

        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            
            # Update frame callback if provided
            if frame_callback:
                frame_callback(frame)
            
            # Skip frames for speed
            if frame_idx % self.skip_frames != 0:
                # Show last annotated frame instead of raw frame to avoid flickering
                if show:
                    display_frame = last_annotated_frame if last_annotated_frame is not None else frame
                    cv2.imshow("Selective Object Tracker", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                if show:
                    cv2.imshow("Selective Object Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                last_annotated_frame = frame
                continue

            # Process each detected object (can be any class)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Extract person crop
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    continue
                
                # Check if this person matches training set (with caching)
                similarity = self._matches_training_set(person_crop, track_id)
                
                if similarity >= self.match_threshold:
                    # This person matches!
                    if track_id and track_id not in matched_ids:
                        matched_ids.add(track_id)
                        matched_person_count = len(matched_ids)
                        
                        if matched_person_count != previous_count:
                            previous_count = matched_person_count
                            msg = f"Frame {frame_idx}: Matched person count: {matched_person_count} (ID: {track_id}, similarity: {similarity:.2f})"
                            
                            with open(result_file, 'a') as f:
                                f.write(msg + "\n")
                            
                            if logger:
                                logger.info(msg)
                            else:
                                print(msg)
                    
                    # Store best match for auto-improvement
                    if self.auto_improve and self.training_image_count < self.max_training_images:
                        if best_match_crop is None or similarity > 0.8:
                            best_match_crop = person_crop.copy()
                    
                    # Get class name if available
                    class_name = self.model.names[int(box.cls[0])] if hasattr(box, 'cls') else "object"
                    
                    # Draw green box for matched object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"MATCH {class_name} ID:{track_id} ({similarity:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Get class name if available
                    class_name = self.model.names[int(box.cls[0])] if hasattr(box, 'cls') else "object"
                    
                    # Draw red box for non-matched object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    label = f"{class_name} ID:{track_id} ({similarity:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Auto-improvement: check if it's time to improve training set
            if self.auto_improve and self.training_image_count < self.max_training_images:
                current_time = time.time()
                if current_time - self.last_improvement_time >= self.improvement_interval:
                    log.info(f"[improve] {self.improvement_interval}s passed | images={self.training_image_count}/{self.max_training_images}")
                    
                    saved_count = 0
                    
                    # Use Gemini for first 4 captures
                    if self.gemini_capture_count < self.max_gemini_captures:
                        if self.current_instruction:
                            saved_count = self._generate_training_with_gemini(frame, self.current_instruction)
                        else:
                            log.warning("[improve] No instruction available for Gemini")
                    # After 4 Gemini captures, use direct detections
                    elif best_match_crop is not None:
                        saved_count = self._generate_augmented_training_images(best_match_crop, "matched_object")
                    else:
                        log.warning("[improve] No best_match_crop and Gemini phase done")
                    
                    # Reload training set
                    if saved_count > 0:
                        self.reload_training_set()
                        
                        if self.gemini_capture_count < self.max_gemini_captures:
                            log.info(f"[improve] Gemini progress: {self.gemini_capture_count}/{self.max_gemini_captures}")
                        else:
                            log.info(f"[improve] Detection-based progress: {self.training_image_count}/{self.max_training_images}")
                        
                        if self.training_image_count >= self.max_training_images:
                            log.info(f"[improve] Reached max images ({self.max_training_images}) -- stopped")
                            self.auto_improve = False
                    
                    self.last_improvement_time = current_time
                    best_match_crop = None  # Reset for next cycle
            
            # Add info overlay
            info_text = f"Matched: {matched_person_count} | Frame: {frame_idx}"
            if self.auto_improve:
                info_text += f" | Training: {self.training_image_count}/{self.max_training_images}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Store annotated frame
            last_annotated_frame = frame
            
            # Display frame
            if show:
                cv2.imshow("Selective Object Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Final summary
        summary = f"\n=== Summary ===\nTotal matched objects: {matched_person_count}\nMatched IDs: {matched_ids}"
        with open(result_file, 'a') as f:
            f.write(summary)
        
        log.info(summary)

    def _track_with_realsense(self, camera, show: bool = True, logger=None, frame_callback=None):
        """Track objects using RealSense camera with manual frame processing.

        Args:
            camera: RealSenseCamera instance
            show: Display video with detections
            logger: Optional logger for output
            frame_callback: Optional callback function(frame) called on each frame
        """
        import time

        result_file = self.create_result_file()
        matched_person_count = 0
        previous_count = 0
        matched_ids = set()
        frame_idx = 0

        # Auto-improvement tracking
        self.last_improvement_time = time.time()
        if not hasattr(self, '_improve_started_at'):
            self._improve_started_at = 0.0
        best_match_crop = None
        last_annotated_frame = None

        # Store best match data for callback
        best_match_data = None
        IMPROVE_STUCK_TIMEOUT = 120  # seconds before watchdog kills a hung thread

        log.info("Starting RealSense tracking loop -- press 'q' to quit")
        last_status_log = time.time()
        status_log_interval = 5  # Log status every 5 seconds
        last_test_log = time.time()

        try:
            while True:
                try:
                    ret, frame = camera.read()

                    # Periodic alive log (every 30s to reduce I/O overhead)
                    if time.time() - last_test_log > 30.0:
                        log.info(f"[debug-loop] Thread alive at frame_idx {frame_idx} | has_frame={ret}")
                        last_test_log = time.time()

                    if not ret or frame is None:
                        log.warning("Failed to read frame from RealSense camera")
                        time.sleep(0.1)
                        continue

                    # Update frame callback if provided
                    if frame_callback:
                        frame_callback(frame)

                    # Skip frames for performance
                    if frame_idx % self.skip_frames != 0:
                        if show:
                            display_frame = last_annotated_frame if last_annotated_frame is not None else frame
                            cv2.imshow("Selective Object Tracker", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        frame_idx += 1
                        continue

                    # Run YOLO detection on this frame
                    # Do NOT pass classes=self.target_classes dynamically to ByteTrack, it can cause CUDA deadlocks or tracker hangs
                    results = self.model.track(
                        source=frame,
                        persist=True,
                        tracker=self.tracker_config,
                        conf=self.conf,
                        device=self.device,
                        iou=self.iou,
                        imgsz=self.img_size,
                        verbose=False,
                        half=self.half
                    )

                    # Variables for overlay
                    best_similarity_for_overlay = -1.0
                    overlay_info = "Searching..."
                    best_match_data = None
                    has_boxes = False

                    if results:
                        result = results[0]
                        boxes = result.boxes
                        has_boxes = boxes is not None and len(boxes) > 0

                    if has_boxes:
                        # 1. First pass to find the absolute best matched object
                        detected_objects = []
                        best_sim = -1.0
                        best_obj_idx = -1

                        # Process each detected object
                        for box in boxes:
                            # Manual class filtering to avoid Ultralytics ByteTrack dynamic classes bug
                            cls_id = int(box.cls[0]) if hasattr(box, 'cls') else None
                            if self.target_classes is not None and cls_id is not None and cls_id not in self.target_classes:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            track_id = int(box.id[0]) if box.id is not None else None

                            # Extract crop
                            person_crop = frame[y1:y2, x1:x2]

                            if person_crop.size == 0:
                                continue

                            # Check if matches training set
                            similarity = self._matches_training_set(person_crop, track_id)

                            # Calculate metrics
                            height, width = frame.shape[:2]
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            x_pct = ((cx - width/2) / (width/2)) * 100
                            box_area = (x2 - x1) * (y2 - y1)
                            frame_area = width * height
                            area_pct = (box_area / frame_area) * 100
                            
                            obj_data = {
                                'box': (x1, y1, x2, y2),
                                'track_id': track_id,
                                'similarity': similarity,
                                'x_pct': x_pct,
                                'area_pct': area_pct,
                                'cls_id': cls_id,
                                'crop': person_crop.copy()
                            }
                            detected_objects.append(obj_data)

                            if similarity > best_sim:
                                best_sim = similarity
                                best_obj_idx = len(detected_objects) - 1

                        # 2. Second pass: visualize and process ONLY the absolute best match
                        for i, obj in enumerate(detected_objects):
                            x1, y1, x2, y2 = obj['box']
                            similarity = obj['similarity']
                            track_id = obj['track_id']
                            x_pct = obj['x_pct']
                            area_pct = obj['area_pct']
                            cls_id = obj['cls_id']
                            
                            is_best_target = (i == best_obj_idx) and (similarity >= self.match_threshold)

                            if is_best_target:
                                if track_id and track_id not in matched_ids:
                                    matched_ids.add(track_id)
                                    msg = f"Frame {frame_idx}: Target locked (ID: {track_id}, similarity: {similarity:.2f})"

                                    with open(result_file, 'a') as f:
                                        f.write(msg + "\n")

                                    if logger:
                                        logger.info(msg)
                                    else:
                                        log.info(msg)

                                # Store best match for auto-improvement
                                if self.auto_improve and self.training_image_count < self.max_training_images:
                                    best_match_crop = obj['crop']

                                class_name = self.model.names[cls_id] if cls_id is not None else "object"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Thicker green box for target
                                label = f"TARGET ID:{track_id} ({similarity:.2f})"
                                cv2.putText(frame, label, (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                best_similarity_for_overlay = similarity
                                overlay_info = f"Target: X:{x_pct:.0f}% Area:{area_pct:.0f}%"
                                best_match_data = (x_pct, area_pct)
                            else:
                                class_name = self.model.names[cls_id] if cls_id is not None else "object"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1) # Thin red box for non-targets
                                label = f"ID:{track_id} {class_name} ({similarity:.2f})"
                                cv2.putText(frame, label, (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        # Call external callback if we have a valid target
                        if best_match_data is not None:
                            process_tracking_data(*best_match_data)

                    # -- Auto-improvement runs regardless of detections --
                    current_time = time.time()
                    if self.auto_improve and self.training_image_count < self.max_training_images:
                        elapsed = current_time - self.last_improvement_time
                        remaining = self.improvement_interval - elapsed

                        # Periodic status log
                        if current_time - last_status_log >= status_log_interval:
                            boxes_count = len(boxes) if has_boxes else 0
                            phase = "gemini" if self.gemini_capture_count < self.max_gemini_captures else "detection"
                            log.info(
                                f"[status] frame={frame_idx} | boxes={boxes_count} | "
                                f"refs={self.training_image_count}/{self.max_training_images} | "
                                f"phase={phase} ({self.gemini_capture_count}/{self.max_gemini_captures} gemini) | "
                                f"next_improve_in={remaining:.0f}s | "
                                f"improving={self.is_improving} | "
                                f"best_crop={'yes' if best_match_crop is not None else 'no'}"
                            )
                            last_status_log = current_time

                        if elapsed >= self.improvement_interval:
                            # Watchdog: reset is_improving if thread appears stuck
                            if self.is_improving and self._improve_started_at > 0:
                                stuck_for = current_time - self._improve_started_at
                                if stuck_for > IMPROVE_STUCK_TIMEOUT:
                                    log.warning(f"[improve] Improvement thread stuck for {stuck_for:.0f}s -- force-resetting is_improving")
                                    self.is_improving = False
                                    self._improve_started_at = 0.0

                            if not self.is_improving:
                                log.info(
                                    f"[improve] firing | "
                                    f"images={self.training_image_count}/{self.max_training_images} | "
                                    f"gemini={self.gemini_capture_count}/{self.max_gemini_captures} | "
                                    f"instruction={'yes' if self.current_instruction else 'NONE'} | "
                                    f"best_crop={'yes' if best_match_crop is not None else 'no'}"
                                )

                                self._improve_started_at = current_time
                                threading.Thread(
                                    target=self._async_improve,
                                    args=(frame.copy(), self.current_instruction, best_match_crop.copy() if best_match_crop is not None else None)
                                ).start()

                                self.last_improvement_time = current_time
                                best_match_crop = None
                            else:
                                stuck_for = current_time - self._improve_started_at
                                log.info(f"[improve] Skipped -- previous improvement still running ({stuck_for:.0f}s)")

                    # Add info overlay
                    cv2.putText(frame, overlay_info, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    info_text_2 = f"Matched: {matched_person_count}"
                    if self.auto_improve:
                        info_text_2 += f" | Train: {self.training_image_count}/{self.max_training_images}"
                    cv2.putText(frame, info_text_2, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    last_annotated_frame = frame

                    if show:
                        cv2.imshow("Selective Object Tracker", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    frame_idx += 1

                except Exception as loop_e:
                    log.error(f"Error executing frame {frame_idx}: {loop_e}", exc_info=True)
                    time.sleep(1) # Prevent tight crash loop
                    frame_idx += 1

        except KeyboardInterrupt:
            log.info("Tracking stopped by user")
        finally:
            summary = f"\n=== Summary ===\nTotal matched objects: {matched_person_count}\nMatched IDs: {matched_ids}"
            with open(result_file, 'a') as f:
                f.write(summary)

            log.info(summary)
    
    def _async_improve(self, frame, instruction, best_match_crop):
        """Run auto-improvement in background thread."""
        self.is_improving = True
        self._improve_started_at = time.time()
        log.info("[async_improve] Background improvement thread started")
        try:
            saved_count = 0
            
            # Use Gemini for first 4 captures
            if self.gemini_capture_count < self.max_gemini_captures:
                if instruction:
                    log.info(f"[async_improve] Using Gemini (capture {self.gemini_capture_count + 1}/{self.max_gemini_captures})")
                    saved_count = self._generate_training_with_gemini(frame, instruction)
                else:
                    log.warning("[async_improve] No instruction available for Gemini -- skipping")
            # After 4 Gemini captures, use direct detections
            elif best_match_crop is not None:
                log.info("[async_improve] Using direct detection crop for augmentation")
                saved_count = self._generate_augmented_training_images(best_match_crop, "matched_object")
            else:
                log.warning("[async_improve] No best_match_crop available and Gemini phase done -- nothing to improve")
            
            log.info(f"[async_improve] Generated {saved_count} new images")
            
            # Reload training set if changes made
            if saved_count > 0:
                self.reload_training_set()
                
                if self.gemini_capture_count < self.max_gemini_captures:
                    log.info(f"[async_improve] Gemini progress: {self.gemini_capture_count}/{self.max_gemini_captures} captures | total_images={self.training_image_count}")
                else:
                    log.info(f"[async_improve] Detection-based progress: {self.training_image_count}/{self.max_training_images} images")
                
                if self.training_image_count >= self.max_training_images:
                    log.info(f"[async_improve] Reached maximum training images ({self.max_training_images}) -- auto-improvement stopped")
                    self.auto_improve = False
            else:
                log.warning("[async_improve] No images saved -- training set unchanged")
                    
        except Exception as e:
            log.error(f"[async_improve] Error in background improvement: {e}", exc_info=True)
        finally:
            self.is_improving = False
            self._improve_started_at = 0.0
            log.info("[async_improve] Background improvement thread finished")


if __name__ == '__main__':
    # Example usage - General object tracking
    source = 0  # Webcam
    
    print("=" * 50)
    print("Selective Object Tracker")
    print("=" * 50)
    print("\nExamples:")
    print("  • target_classes=None        → Track ALL objects")
    print("  • target_classes=[0]         → Track persons only")
    print("  • target_classes=[2, 3, 5]   → Track cars, motorcycles, buses")
    print("  • target_classes=[16, 17]    → Track dogs and cats")
    print("\nCommon YOLO classes:")
    print("  0=person, 2=car, 3=motorcycle, 5=bus, 7=truck")
    print("  16=dog, 17=cat, 24=backpack, 26=handbag, 67=cell phone")
    print("=" * 50 + "\n")
    
    tracker = SelectivePersonTracker(
        model_path='yolov8n.pt',
        training_set_dir='training_set/output',
        match_threshold=0.6,
        device=get_best_device(),
        img_size=(480, 640),
        skip_frames=2,
        conf=0.6,
        target_classes=None,  # Track EVERYTHING
        auto_improve=False
    )
    
    tracker.detect_and_track(source=source, show=True)
