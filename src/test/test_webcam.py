"""Test webcam access and find available cameras."""

import cv2
import sys

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️  pyrealsense2 not available - RealSense detection disabled")

print("="*60)
print("CAMERA DETECTION TEST")
print("="*60)

# Test RealSense cameras first
if REALSENSE_AVAILABLE:
    print("\n🔍 Searching for RealSense cameras...\n")

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) > 0:
        print(f"✅ Found {len(devices)} RealSense device(s):")
        for i, dev in enumerate(devices):
            print(f"   [{i}] {dev.get_info(rs.camera_info.name)}")
            print(f"       Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"       Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

        # Test RealSense stream
        print("\n🎥 Testing RealSense camera stream...")
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            pipeline.start(config)
            print("✅ RealSense pipeline started successfully!")

            # Grab a few frames
            for _ in range(5):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame and depth_frame:
                    print("✅ Receiving color and depth frames")
                    break

            pipeline.stop()
            print("✅ RealSense camera is working perfectly!")

        except Exception as e:
            print(f"❌ RealSense test failed: {e}")
    else:
        print("❌ No RealSense devices found")
        print("\n💡 Troubleshooting:")
        print("   1. Check USB connection (use USB 3.0 port for best performance)")
        print("   2. Check permissions: sudo usermod -a -G video $USER")
        print("   3. Run: rs-enumerate-devices to see RealSense devices")

# Test regular webcams
print(f"\n{'='*60}")
print("\n🔍 Searching for regular webcams...\n")

available_cameras = []

for i in range(5):  # Test indices 0-4
    print(f"Testing camera index {i}...", end=" ")
    cap = cv2.VideoCapture(i)  # Remove Windows-specific backend

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✅ Working! Resolution: {w}x{h}")
            available_cameras.append(i)
        else:
            print("❌ Can't read frames")
        cap.release()
    else:
        print("❌ Can't open")

print(f"\n{'='*60}")

if available_cameras:
    print(f"✅ Found {len(available_cameras)} working camera(s): {available_cameras}")
    print(f"\n💡 Use camera index: {available_cameras[0]} (recommended)")
    
    # Test with a quick video display
    print(f"\n🎥 Testing camera {available_cameras[0]} with live preview...")
    print("   Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(available_cameras[0])

    # Set properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    while frame_count < 100:  # Show 100 frames
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            cv2.putText(frame, f"Frame: {frame_count} - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Webcam Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"❌ Failed to read frame {frame_count}")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed!")
    
else:
    print("ℹ️  No regular webcams found (RealSense cameras need pyrealsense2)")

print(f"\n{'='*60}")
