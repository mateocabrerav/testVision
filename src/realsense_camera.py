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
    Optionally opens the depth V4L2 node for real depth measurements.
    """

    def __init__(self, device_index: int, depth_index: int | None = None,
                 width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._depth_cap = None
        self._depth_frame: np.ndarray | None = None
        if depth_index is not None:
            self._depth_cap = self._init_depth_cap(depth_index, width, height, fps)

    @staticmethod
    def _init_depth_cap(idx: int, w: int, h: int, fps: int):
        """Open RealSense depth V4L2 node for raw 16-bit depth reading."""
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Keep raw 16-bit depth data
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"Depth V4L2: /dev/video{idx} failed to open")
            cap.release()
            return None
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Depth V4L2: /dev/video{idx} failed to read")
            cap.release()
            return None
        print(f"Depth V4L2: /dev/video{idx} depth ready (shape={frame.shape}, dtype={frame.dtype})")
        return cap

    def read(self):
        ret, frame = self.cap.read()
        if ret and self._depth_cap is not None:
            dep_ret, dep_raw = self._depth_cap.read()
            if dep_ret and dep_raw is not None:
                try:
                    if dep_raw.dtype == np.uint16:
                        self._depth_frame = dep_raw
                    else:
                        # Y16 via V4L2: uint8 array with doubled width, reinterpret as uint16 depth (mm)
                        self._depth_frame = np.ascontiguousarray(dep_raw).view(np.uint16)
                except Exception:
                    self._depth_frame = None
        return ret, frame

    def get_distance_at(self, cx: int, cy: int, radius: int = 2) -> float:
        """Return median depth in meters at (cx, cy) using a patch of valid pixels."""
        if self._depth_frame is None:
            return 0.0
        h, w = self._depth_frame.shape[:2]
        patch = self._depth_frame[
            max(0, cy - radius):min(h, cy + radius + 1),
            max(0, cx - radius):min(w, cx + radius + 1),
        ]
        valid = patch[patch > 0]
        return round(float(np.median(valid)) / 1000.0, 2) if valid.size > 0 else 0.0

    def release(self):
        self.cap.release()
        if self._depth_cap is not None:
            self._depth_cap.release()

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


def _find_depth_v4l2_index() -> int | None:
    """Find the RealSense depth V4L2 node index via sysfs device name."""
    for vdir in sorted(glob.glob('/sys/class/video4linux/video*')):
        name_file = os.path.join(vdir, 'name')
        if not os.path.exists(name_file):
            continue
        with open(name_file) as f:
            name = f.read().strip()
        if 'RealSense' not in name or 'Infrared' in name or 'RGB' in name:
            continue
        if 'Depth' not in name:
            continue
        parent = os.path.realpath(os.path.join(vdir, 'device'))
        v4l_dir = os.path.join(parent, 'video4linux')
        sibling_count = len(os.listdir(v4l_dir)) if os.path.isdir(v4l_dir) else 1
        if sibling_count < 3:  # Must be in the depth/IR interface group
            continue
        idx = int(os.path.basename(vdir).replace('video', ''))
        print(f"sysfs: RealSense depth node: /dev/video{idx} ({name})")
        return idx
    return None


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

    depth_index = _find_depth_v4l2_index()
    for idx in candidates:
        cam = OpenCVCamera(idx, depth_index=depth_index, width=width, height=height, fps=fps)
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
        depth_ok = cam._depth_cap is not None
        print(f"OpenCV V4L2: /dev/video{idx} color stream ready ({frame.shape[1]}x{frame.shape[0]}) [depth={'ok' if depth_ok else 'unavailable'}]")
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
            self._last_depth: np.ndarray | None = None
            self._depth_scale: float = 0.001  # default: 1mm per depth unit

        def open(self, w, h, f):
            self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
            profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            try:
                self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            except Exception:
                pass  # keep default 0.001
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
                depth = aligned.get_depth_frame()
                if depth:
                    self._last_depth = np.asanyarray(depth.get_data())
                return True, np.asanyarray(color.get_data())
            except Exception:
                return False, None

        def get_distance_at(self, cx: int, cy: int, radius: int = 2) -> float:
            """Return median depth in meters at (cx, cy) using aligned depth frame."""
            if self._last_depth is None:
                return 0.0
            h, w = self._last_depth.shape[:2]
            patch = self._last_depth[
                max(0, cy - radius):min(h, cy + radius + 1),
                max(0, cx - radius):min(w, cx + radius + 1),
            ]
            valid = patch[patch > 0]
            return round(float(np.median(valid)) * self._depth_scale, 2) if valid.size > 0 else 0.0

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
