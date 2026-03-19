"""Firebase listener for dynamic object tracking."""

import os
import sys
from pathlib import Path
from datetime import datetime
import threading
import time
from dotenv import load_dotenv
from PIL import Image
import mss
from firebase_admin import db
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training_set.generate_training_set import create_training_set
from src.person_tracker.selective_person_tracker import SelectivePersonTracker
from src.firebase_app import initialize_firebase_app


def get_best_device():
    """Auto-detect best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch_string = f"sm_{cap[0]}{cap[1]}"
        arch_list = torch.cuda.get_arch_list()

        # The Jetson Orin is sm_87. Using generic torch on ARM64 lacks sm_87 support.
        if "sm_87" not in arch_list and arch_string == "sm_87":
            print(f"⚠️ PyTorch lacks {arch_string} support (Jetson Orin). Falling back to CPU.")
            return 'cpu'

        device = 'cuda:0'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 Using GPU: {gpu_name}")
    else:
        device = 'cpu'
        print(f"💻 Using CPU (GPU not available)")
    return device


class FirebaseTrackerController:
    """Control object tracking via Firebase instructions."""
    
    def __init__(
        self,
        firebase_creds_path: str,
        firebase_db_url: str,
        gemini_api_key: str,
        yolo_model: str = 'yolov8n.pt',
        device: str = None  # Auto-detect if None
    ):
        self.gemini_api_key = gemini_api_key
        self.yolo_model = yolo_model
        self.device = device if device else get_best_device()
        self.current_tracker = None
        self.tracking_thread = None
        self.stop_tracking = threading.Event()
        self.last_instruction = None
        
        # Initialize Firebase
        print("🔥 Initializing Firebase...")
        self.firebase_app = initialize_firebase_app(
            firebase_creds_path,
            firebase_db_url,
        )
        print("✅ Firebase connected")
        
    def capture_screenshot(self, output_path: str) -> str:
        """Capture screenshot and save to file."""
        print(f"📸 Capturing screenshot...")
        
        with mss.mss() as sct:
            # Capture primary monitor
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            
            # Convert to PIL Image and save
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            img.save(output_path)
            
        print(f"✅ Screenshot saved: {output_path}")
        return output_path
    
    def process_instruction(self, instruction: str):
        """Process new instruction: screenshot → training set → tracking."""
        print(f"\n{'='*60}")
        print(f"🎯 New instruction received: {instruction}")
        print(f"{'='*60}\n")
        
        # Stop current tracking if running
        if self.tracking_thread and self.tracking_thread.is_alive():
            print("⏸️  Stopping current tracking...")
            self.stop_tracking.set()
            self.tracking_thread.join(timeout=5)
            self.stop_tracking.clear()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"temp_screenshots/screenshot_{timestamp}.png"
        os.makedirs("temp_screenshots", exist_ok=True)
        
        try:
            # Step 1: Capture screenshot
            self.capture_screenshot(screenshot_path)
            
            # Step 2: Generate training set from screenshot
            print("\n📦 Generating training set...")
            create_training_set(
                api_key=self.gemini_api_key,
                input_image_path=screenshot_path,
                instruction=instruction,
                output_dir="training_set/output<"
            )
            
            # Step 3: Start tracking in separate thread
            print("\n🎥 Starting object tracking...")
            self.tracking_thread = threading.Thread(
                target=self._run_tracking,
                args=(instruction,),
                daemon=True
            )
            self.tracking_thread.start()
            
            print("✅ Pipeline completed successfully!")
            
        except Exception as e:
            print(f"❌ Error processing instruction: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_tracking(self, instruction: str):
        """Run tracking in background thread."""
        try:
            tracker = SelectivePersonTracker(
                model_path=self.yolo_model,
                training_set_dir='training_set/output',
                match_threshold=0.6,
                device=self.device,
                img_size=(480, 640),  # Optimized size for speed
                skip_frames=2,  # Process every 2nd frame
                conf=0.6  # Slightly higher confidence for fewer false positives
            )
            
            # Run tracking with stop event check
            self.current_tracker = tracker
            tracker.detect_and_track(
                source=0,  # Webcam
                show=True,
                logger=None
            )
            
        except Exception as e:
            print(f"❌ Tracking error: {e}")
    
    def on_instruction_change(self, event):
        """Firebase listener callback."""
        if event.data is None:
            return
        
        instruction = event.data
        
        # Skip if same as last instruction
        if instruction == self.last_instruction:
            return
        
        self.last_instruction = instruction
        
        # Process in separate thread to not block Firebase listener
        process_thread = threading.Thread(
            target=self.process_instruction,
            args=(instruction,),
            daemon=True
        )
        process_thread.start()
    
    def start_listening(self):
        """Start listening to Firebase instruction node."""
        print("\n👂 Listening for Firebase 'instruction' changes...")
        print("📝 Update the 'instruction' node to trigger tracking\n")
        
        # Listen to instruction node
        ref = db.reference('instruction', app=self.firebase_app)
        ref.listen(self.on_instruction_change)
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            if self.tracking_thread and self.tracking_thread.is_alive():
                self.stop_tracking.set()


if __name__ == "__main__":
    load_dotenv()
    
    # Configuration
    FIREBASE_CREDS = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
    FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Validate configuration
    if not os.path.exists(FIREBASE_CREDS):
        print(f"❌ Firebase credentials not found: {FIREBASE_CREDS}")
        print("📝 Download your Firebase service account key and save it as firebase_credentials.json")
        sys.exit(1)
    
    if not FIREBASE_DB_URL:
        print("❌ FIREBASE_DB_URL not set in .env file")
        print("📝 Add: FIREBASE_DB_URL=https://your-project.firebaseio.com")
        sys.exit(1)
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set in .env file")
        sys.exit(1)
    
    # Start controller (auto-detects GPU)
    controller = FirebaseTrackerController(
        firebase_creds_path=FIREBASE_CREDS,
        firebase_db_url=FIREBASE_DB_URL,
        gemini_api_key=GEMINI_API_KEY
        # device will be auto-detected (GPU if available, else CPU)
    )
    
    controller.start_listening()
