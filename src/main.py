"""Entry point: Firebase listener + parallel person tracker."""

import os
import sys
import threading
import signal
import time
import logging
import cv2

# Force unbuffered stdout so prints appear immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
from pathlib import Path
from dotenv import load_dotenv
from firebase_admin import db

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from person_tracker.selective_person_tracker import SelectivePersonTracker, get_best_device
from training_set.generate_training_set import create_training_set
from gemini.classify_instruction import classify_instruction
from PIL import Image
from realsense_camera import get_realsense_camera
from firebase_app import initialize_firebase_app


# Ensure project root is on sys.path so relative imports work the same as elsewhere
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

# Global flag for shutdown and current frame
shutdown_event = threading.Event()
current_frame = None
frame_lock = threading.Lock()
tracker_instance = None
tracker_lock = threading.Lock()


def on_instruction_change(event):
    """Callback for Realtime Database 'instruction' node changes.

    Captures current frame and generates new training set.
    """
    # Start processing in background to avoid blocking listener
    threading.Thread(target=_process_instruction, args=(event,)).start()


def _process_instruction(event):
    """Process instruction change in background thread."""
    try:
        instruction = event.data
        log.info(f"Firebase event received | data: {instruction}")

        if not instruction:
            log.info("Empty instruction, skipping")
            return

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            log.error("GEMINI_API_KEY not found in .env")
            return

        target_classes = classify_instruction(api_key, instruction)
        log.info(f"Gemini classified target_classes: {target_classes}")

        with tracker_lock:
            if tracker_instance is None:
                log.warning("Tracker not ready yet")
                return
            tracker_instance.target_classes = target_classes
            tracker_instance.set_instruction(instruction)
            clip_mode = tracker_instance.clip_active
            log.info(f"Tracker updated | mode={'CLIP' if clip_mode else 'histogram'} | classes={target_classes}")

        if clip_mode:
            return  # CLIP needs no training set — done

        # Histogram fallback: generate a training set via Gemini
        log.info("Waiting for camera frame (histogram fallback)...")
        deadline = time.time() + 15
        while time.time() < deadline:
            with frame_lock:
                if current_frame is not None:
                    break
            time.sleep(0.2)
        else:
            log.error("Timed out waiting for camera frame (15s), skipping training set generation")
            return

        with frame_lock:
            frame_copy = current_frame.copy()

        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame_copy)
        try:
            create_training_set(
                api_key=api_key,
                input_image_path=temp_image_path,
                instruction=instruction,
                output_dir='training_set/output',
            )
        except Exception as create_err:
            log.warning(f"create_training_set failed: {create_err} -- will rely on auto-improvement")
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        with tracker_lock:
            if tracker_instance is not None:
                tracker_instance.reload_training_set()
                tracker_instance.restart_auto_improvement(instruction=instruction)
                log.info(
                    f"Tracker ready | images={tracker_instance.training_image_count} | "
                    f"max={tracker_instance.max_training_images}"
                )

    except Exception as e:
        log.error(f"Error in _process_instruction: {e}", exc_info=True)


def run_tracker():
    """Run camera feed + selective person tracker.

    The camera window is always open. Detection overlays appear once
    the tracker finishes loading in the background.
    """
    import time

    log.info("Starting selective person tracker with RealSense D415...")

    camera = get_realsense_camera(width=640, height=480, fps=30)

    if camera is None:
        log.error("Failed to initialize RealSense D415 camera")
        log.error("Check USB connection, run: lsusb | grep Intel")
        return

    log.info("RealSense D415 camera ready")

    global current_frame

    def update_frame(frame):
        """Update global current frame for Firebase callback."""
        global current_frame
        with frame_lock:
            current_frame = frame

    # Initialize tracker in background so camera feed is never blocked
    tracker_ready = threading.Event()
    tracker_error = [None]

    def init_tracker():
        global tracker_instance
        try:
            tracker = SelectivePersonTracker(
                model_path='yolov8n.pt',
                training_set_dir='training_set/output',
                match_threshold=0.70,
                device=get_best_device(),
                img_size=(320, 320),
                skip_frames=2,
                conf=0.50,
                auto_improve=True,
                max_training_images=200,
                improvement_interval=10,
                target_classes=[0]
            )
            with tracker_lock:
                tracker_instance = tracker
            log.info("Tracker initialized and ready")
        except Exception as e:
            tracker_error[0] = e
            log.error(f"Tracker init error: {e}", exc_info=True)
        finally:
            tracker_ready.set()

    threading.Thread(target=init_tracker, daemon=True).start()

    log.info("Camera preview started (tracker loading in background)...")
    yolo_started = False
    try:
        while not shutdown_event.is_set():
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            update_frame(frame)

            # Once tracker is ready, hand off to its detection loop
            if tracker_ready.is_set() and tracker_error[0] is None and not yolo_started:
                yolo_started = True
                log.info("Handing off to tracker detection loop...")
                try:
                    tracker_instance.detect_and_track(
                        source=camera, show=True, frame_callback=update_frame
                    )
                    break
                except Exception as e:
                    log.error(f"Tracker detection error: {e}", exc_info=True)
                    yolo_started = False
                    tracker_ready.clear()
                continue

            # Raw preview while tracker is not active
            status = "Initializing tracker..."
            if tracker_error[0] is not None:
                status = f"Tracker error: {tracker_error[0]}"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Selective Object Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        log.info("Tracker stopped by user")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        log.info("Tracker cleanup complete")

def main():
    load_dotenv()

    # Get project root directory (parent of src/)
    project_root = Path(__file__).parent.parent

    # Resolve Firebase credentials path relative to project root
    FIREBASE_CREDS = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not FIREBASE_CREDS:
        # Default to project root
        FIREBASE_CREDS = project_root / "firebase_credentials.json"
    else:
        # If provided in .env, make it absolute relative to project root
        FIREBASE_CREDS = project_root / FIREBASE_CREDS if not os.path.isabs(FIREBASE_CREDS) else FIREBASE_CREDS

    FIREBASE_CREDS = str(FIREBASE_CREDS)  # Convert Path to string
    FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")

    if not os.path.exists(FIREBASE_CREDS):
        log.error(f"Firebase credentials not found: {FIREBASE_CREDS}")
        return

    if not FIREBASE_DB_URL:
        log.error("FIREBASE_DB_URL not set in environment")
        return

    firebase_app = initialize_firebase_app(FIREBASE_CREDS, FIREBASE_DB_URL)

    log.info("Firebase initialized")
    
    # Start tracker in parallel thread
    tracker_thread = threading.Thread(target=run_tracker, daemon=True)
    tracker_thread.start()
    log.info("Tracker thread started")
    
    log.info("Listening to Firebase 'instruction' node")
    ref = db.reference('instruction', app=firebase_app)
    
    # Get initial value with retry logic
    current_value = None
    max_retries = 5
    for attempt in range(max_retries):
        try:
            current_value = ref.get()
            log.info(f"Current instruction value: '{current_value}'")
            break
        except Exception as e:
            log.warning(f"Firebase connection attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                log.error("Failed to connect to Firebase")
                shutdown_event.set()
                return
    
    # Attach listener directly
    try:
        ref.listen(on_instruction_change)
        log.info("Listener attached -- change 'instruction' in Firebase to trigger processing")
    except Exception as e:
        log.error(f"Failed to attach listener: {e}")
        shutdown_event.set()
        return
    
    # Keep main thread alive
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down...")
        shutdown_event.set()
        time.sleep(1)  # Give threads time to cleanup
        print("Shutdown complete")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    shutdown_event.set()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
