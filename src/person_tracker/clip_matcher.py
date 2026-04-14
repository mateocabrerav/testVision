"""Zero-shot image-text matching via CLIP (HuggingFace transformers)."""
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


class ClipMatcher:
    """Match detections against a natural language instruction using CLIP.

    Inference runs in a background thread so it never blocks the tracking loop.
    - First time a track_id is seen: returns 0.0, spawns async inference.
    - Subsequent frames: returns cached score instantly.
    - Cache is cleared only when the instruction changes.
    """

    def __init__(self, device: str = 'cpu'):
        from transformers import CLIPModel, CLIPProcessor

        self._device = device
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._text_features: Optional[torch.Tensor] = None
        self._cache: dict[int, float] = {}
        self._pending: set[int] = set()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip")

    def set_instruction(self, instruction: str) -> None:
        """Encode text instruction (called once per instruction change)."""
        inputs = self._processor(text=[instruction], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            self._text_features = self._model.get_text_features(**inputs)
            self._text_features = self._text_features / self._text_features.norm(dim=-1, keepdim=True)
        with self._lock:
            self._cache.clear()
            self._pending.clear()

    @property
    def active(self) -> bool:
        return self._text_features is not None

    def score(self, crop_bgr: np.ndarray, track_id: Optional[int] = None) -> float:
        """Return cached CLIP score (non-blocking). Returns 0.0 while still computing."""
        if not self.active or crop_bgr.size == 0:
            return 0.0

        with self._lock:
            if track_id is not None and track_id in self._cache:
                return self._cache[track_id]
            if track_id is not None and track_id in self._pending:
                return 0.0
            if track_id is not None:
                self._pending.add(track_id)

        crop_copy = crop_bgr.copy()
        self._executor.submit(self._infer_and_cache, crop_copy, track_id)
        return 0.0

    def _infer_and_cache(self, crop_bgr: np.ndarray, track_id: Optional[int]) -> None:
        """Run CLIP inference in background thread and store result."""
        try:
            pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                img_features = self._model.get_image_features(**inputs)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                sim = float((img_features @ self._text_features.T).squeeze().item())
                sim = (sim + 1.0) / 2.0
            with self._lock:
                if track_id is not None:
                    self._cache[track_id] = sim
        finally:
            with self._lock:
                self._pending.discard(track_id)
