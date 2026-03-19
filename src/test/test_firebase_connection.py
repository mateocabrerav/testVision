"""Test Firebase listener without actual tracking."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from firebase_admin import db

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.firebase_app import initialize_firebase_app


def test_firebase_connection():
    """Test Firebase connection and instruction listener."""
    load_dotenv()
    
    FIREBASE_CREDS = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")
    FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
    
    if not os.path.exists(FIREBASE_CREDS):
        print(f"❌ Firebase credentials not found: {FIREBASE_CREDS}")
        return False
    
    if not FIREBASE_DB_URL:
        print("❌ FIREBASE_DB_URL not set in .env")
        return False
    
    try:
        # Initialize Firebase
        print("🔥 Connecting to Firebase...")
        firebase_app = initialize_firebase_app(FIREBASE_CREDS, FIREBASE_DB_URL)
        print("✅ Firebase connected successfully!")
        
        # Test reading instruction
        ref = db.reference('instruction', app=firebase_app)
        instruction = ref.get()
        
        if instruction:
            print(f"📝 Current instruction: {instruction}")
        else:
            print("⚠️  No instruction found. Creating default...")
            ref.set("test instruction")
            print("✅ Default instruction created")
        
        # Listen for changes
        print("\n👂 Listening for instruction changes...")
        print("📝 Go to Firebase Console and change the 'instruction' value")
        print("   Press Ctrl+C to stop\n")
        
        def on_change(event):
            print(f"🔔 Instruction changed to: {event.data}")
        
        ref.listen(on_change)
        
        # Keep alive
        import time
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_firebase_connection()

