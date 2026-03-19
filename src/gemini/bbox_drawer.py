"""Draw bounding boxes on images from Gemini Robotics detections."""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple


def _normalize_box(box: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """Normalize [y1,x1,y2,x2] that may be in 0-1, 0-1000, or pixels.

    Returns integer pixel coordinates clamped to image bounds.
    """
    y1, x1, y2, x2 = [float(v) for v in box]
    maxv = max(abs(y1), abs(x1), abs(y2), abs(x2))

    if maxv <= 1.5:  # normalized 0..1
        py1, px1, py2, px2 = y1 * h, x1 * w, y2 * h, x2 * w
    elif maxv <= 1000.0:  # normalized 0..1000
        scale_x, scale_y = w / 1000.0, h / 1000.0
        py1, px1, py2, px2 = y1 * scale_y, x1 * scale_x, y2 * scale_y, x2 * scale_x
    else:  # assume pixels
        py1, px1, py2, px2 = y1, x1, y2, x2

    # Convert to ints and clamp
    px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
    px1, py1 = max(0, min(w - 1, px1)), max(0, min(h - 1, py1))
    px2, py2 = max(0, min(w - 1, px2)), max(0, min(h - 1, py2))
    if px2 < px1:
        px1, px2 = px2, px1
    if py2 < py1:
        py1, py2 = py2, py1
    return px1, py1, px2, py2


def draw_points(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """
    Draw detection bounding boxes on image.

    Args:
        image: PIL Image
        detections: List from Gemini [{"box": [y1, x1, y2, x2], "label": "..."}]
                   or legacy [{"point": [y, x], "label": "..."}]

    Returns:
        PIL Image with drawn bounding boxes
    """
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    for det in detections or []:
        # Accept various keys
        box = det.get("box") or det.get("box_2d") or det.get("bbox")
        if box and len(box) == 4:
            px1, py1, px2, py2 = _normalize_box(box, w, h)
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
            label = str(det.get("label", "object"))
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                img, (px1, py1 - label_h - 4), (px1 + label_w, py1), (0, 255, 0), -1
            )
            cv2.putText(
                img,
                label,
                (px1, max(0, py1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        elif "point" in det:
            y, x = det["point"]
            # Support 0..1 or 0..1000 or pixels
            maxp = max(abs(float(y)), abs(float(x)))
            if maxp <= 1.5:
                px, py = int(float(x) * w), int(float(y) * h)
            elif maxp <= 1000.0:
                px, py = int(float(x) * w / 1000.0), int(float(y) * h / 1000.0)
            else:
                px, py = int(float(x)), int(float(y))
            cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(
                img,
                str(det.get("label", "point")),
                (min(w - 1, px + 10), py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
