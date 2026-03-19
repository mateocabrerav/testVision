"""RealSense camera wrapper with OpenCV V4L2 fallback for Jetson."""

import os
import glob
import numpy as np
import cv2
from typing import Tuple


class OpenCVCamera:
    """OpenCV V4L2 camera (used when pyrealsense2 cannot claim the device).

    On Jetson (L4T), the uvcvideo kernel driver owns the RealSense, so
    pyrealsense2 cannot open it.  We use plain V4L2 via OpenCV instead.
    """

    def __init__(self, device_index: int, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def set(self, prop_id, value):
        self.cap.set(prop_id, value)


def _find_realsense_v4l2_index() -> list[int] | None:
    """Find the RealSense color V4L2 device indices.

    The D435i exposes two USB interfaces as separate sysfs groups:
      - Depth/IR group (4 nodes) — NOT color
      - Color group (2 nodes) — actual RGB stream

    Returns color-group nodes first so the caller tries them before
    depth/IR nodes.
    """
    # Map each RealSense video index to its sysfs sibling count
    rs_nodes: dict[int, int] = {}
    for vdir in sorted(glob.glob('/sys/class/video4linux/video*')):
        name_file = os.path.join(vdir, 'name')
        if not os.path.exists(name_file):
            continue
        with open(name_file) as f:
            name = f.read().strip()
        if 'RealSense' not in name:
            continue
        idx = int(os.path.basename(vdir).replace('video', ''))
        # Count siblings (nodes sharing the same parent device)
        parent = os.path.realpath(os.path.join(vdir, 'device'))
        v4l_dir = os.path.join(parent, 'video4linux')
        sibling_count = len(os.listdir(v4l_dir)) if os.path.isdir(v4l_dir) else 1
        rs_nodes[idx] = sibling_count

    if not rs_nodes:
        return None

    # Sort: color group (fewer siblings, typically 2) before depth (4)
    sorted_indices = sorted(rs_nodes, key=lambda i: (rs_nodes[i], i))
    print(f"sysfs: RealSense nodes (color-first): {sorted_indices}")
    return sorted_indices


def _try_opencv_fallback(width: int, height: int, fps: int):
    """Open the RealSense camera via OpenCV V4L2.

    Tries env override first, then probes only sysfs-confirmed RealSense
    nodes.  Keeps the capture open once a valid color frame is received.
    """
    idx_env = os.getenv("REALSENSE_VIDEO_INDEX")
    if idx_env:
        candidates = [int(idx_env)]
    else:
        candidates = _find_realsense_v4l2_index() or []

    if not candidates:
        print("Could not locate RealSense V4L2 device")
        return None

    for idx in candidates:
        cam = OpenCVCamera(idx, width, height, fps)
        if not cam.isOpened():
            cam.release()
            continue
        ret, frame = cam.read()
        if not (ret and frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3):
            cam.release()
            continue
        # Reject depth/IR frames (channels nearly identical = grayscale)
        ch_diff = float(np.abs(frame[:, :, 0].astype(np.int16) - frame[:, :, 1].astype(np.int16)).mean())
        if ch_diff < 3.0:
            print(f"OpenCV V4L2: /dev/video{idx} looks like depth/IR (ch_diff={ch_diff:.1f}), skipping")
            cam.release()
            continue
        print(f"OpenCV V4L2: /dev/video{idx} color stream ready ({frame.shape[1]}x{frame.shape[0]})")
        return cam

    print("OpenCV V4L2: no RealSense node delivered color frames")
    return None


def _try_realsense_sdk(width: int, height: int, fps: int):
    """Try to open the camera via pyrealsense2 SDK (lazy import)."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed, skipping SDK path")
        return None

    try:
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            print("pyrealsense2: no devices found")
            return None
    except Exception as e:
        print(f"pyrealsense2: device query failed ({e})")
        return None

    class _RSCamera:
        """Thin wrapper returned when pyrealsense2 SDK works."""

        def __init__(self):
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.align = None
            self.is_opened = False

        def open(self, w, h, f):
            self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            # Warm up
            for _ in range(30):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=1000)
                except Exception:
                    pass
            self.is_opened = True

        def read(self):
            if not self.is_opened:
                return False, None
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned = self.align.process(frames)
                color = aligned.get_color_frame()
                if not color:
                    return False, None
                return True, np.asanyarray(color.get_data())
            except Exception:
                return False, None

        def release(self):
            if self.is_opened:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.is_opened = False

        def isOpened(self):
            return self.is_opened

        def set(self, prop_id, value):
            pass

    cam = _RSCamera()
    try:
        cam.open(width, height, fps)
        print(f"✅ RealSense SDK: {width}x{height} @ {fps}fps")
        return cam
    except Exception as e:
        print(f"pyrealsense2 pipeline failed: {e}")
        return None


def get_realsense_camera(width: int = 640, height: int = 480, fps: int = 30):
    """Return an initialized camera.

    Strategy (Jetson-safe):
      1. OpenCV V4L2 via sysfs lookup (fast, no SDK needed)
      2. pyrealsense2 SDK (lazy import, only if V4L2 fails)

    Args:
        width: Frame width
        height: Frame height
        fps: Frames per second

    Returns:
        Camera with read()/release()/isOpened(), or None
    """
    # Fast path — OpenCV V4L2 (always works on Jetson)
    cam = _try_opencv_fallback(width, height, fps)
    if cam:
        return cam

    # Slow path — try native pyrealsense2 SDK
    print("V4L2 fallback failed, trying pyrealsense2 SDK...")
    cam = _try_realsense_sdk(width, height, fps)
    if cam:
        return cam

    print("No camera available")
    return None
