from PIL import Image
import cv2
import numpy as np
from rembg import remove
import io
import base64
import os
from typing import List, Tuple, Optional
from pathlib import Path

BLUE_BG_COLOR = (0, 102, 204)  # Passport blue (RGB)

class PassportPhotoMaker:
    """Utility to detect faces and produce passport-style photos with a blue background.

    Public methods:
    - process(img_bytes) -> List[str]: returns list of base64 JPEG strings (one per detected face)
    - save_base64_outputs(outputs, prefix, output_dir) -> List[str]: saves base64 images to files and returns filepaths
    """

    def __init__(
        self,
        cascade_path: str = "haarcascade_frontalface_default.xml",
        dpi: int = 300,
        target_inches: Tuple[float, float] = (1.5, 1.9),
        expand_factor: float = 1.6,
        min_face_size_px: int = 30,
        blue_bg_color: Tuple[int, int, int] = BLUE_BG_COLOR,
    ) -> None:
        self.dpi = dpi
        self.target_inches = target_inches
        self.expand_factor = expand_factor
        self.min_face_size_px = min_face_size_px
        self.blue_bg_color = blue_bg_color

        # Choose a resampling filter compatible with Pillow versions
        try:
            # Pillow >= 9.1 exposes Resampling
            self._resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            # older Pillow versions: prefer LANCZOS if available, otherwise fall back gracefully
            if hasattr(Image, "LANCZOS"):
                self._resample = Image.LANCZOS  # type: ignore[attr-defined]
            elif hasattr(Image, "ANTIALIAS"):
                self._resample = Image.ANTIALIAS  # type: ignore[attr-defined]
            else:
                # last-resort fallback
                self._resample = Image.NEAREST  # type: ignore[attr-defined]

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"{cascade_path} not found. Download it and place it in the same folder."
            )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    # ------------------------------
    # Low-level helpers
    # ------------------------------
    def _decode_image_bytes(self, img_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image bytes into an RGB numpy array. Returns None if decoding fails."""
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _detect_faces(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(self.min_face_size_px, self.min_face_size_px)
        )
        # explicit 4-tuple conversion for type checker
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]

    def _target_pixels(self) -> Tuple[int, int, float]:
        tgt_w_px = int(round(self.target_inches[0] * self.dpi))
        tgt_h_px = int(round(self.target_inches[1] * self.dpi))
        tgt_aspect = tgt_w_px / tgt_h_px
        return tgt_w_px, tgt_h_px, tgt_aspect

    def _compute_crop_box(
        self, face: Tuple[int, int, int, int], img_shape: Tuple[int, int, int], tgt_aspect: float
    ) -> Tuple[int, int, int, int]:
        """Given a detected face and image shape, compute a crop box that matches the
        target aspect ratio and expansion factor. Returns (x1,y1,x2,y2) clipped to image bounds.
        """
        x, y, fw, fh = face
        h, w = img_shape[:2]
        cx, cy = x + fw / 2.0, y + fh / 2.0
        new_w, new_h = fw * self.expand_factor, fh * self.expand_factor

        if new_w / new_h > tgt_aspect:
            new_h = new_w / tgt_aspect
        else:
            new_w = new_h * tgt_aspect

        x1 = int(cx - new_w / 2.0)
        y1 = int(cy - new_h / 2.0)
        x2 = int(cx + new_w / 2.0)
        y2 = int(cy + new_h / 2.0)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return x1, y1, x2, y2

    def _crop_and_resize(self, img_rgb: np.ndarray, box: Tuple[int, int, int, int], tgt_size: Tuple[int, int]) -> Image.Image:
        x1, y1, x2, y2 = box
        if y2 <= y1 or x2 <= x1:
            # empty crop - return a blank image of target size
            return Image.new("RGB", tgt_size, self.blue_bg_color)
        crop = img_rgb[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)
        resized = crop_pil.resize(tgt_size, self._resample)
        return resized

    def _apply_blue_background(self, img_pil: Image.Image) -> Image.Image:
        """Remove background from PIL image and composite over a blue background."""
        img_no_bg = remove(img_pil).convert("RGBA")
        bg = Image.new("RGBA", img_no_bg.size, self.blue_bg_color + (255,))
        final_img = Image.alpha_composite(bg, img_no_bg).convert("RGB")
        return final_img

    def _image_to_base64(self, img_pil: Image.Image) -> str:
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", dpi=(self.dpi, self.dpi))
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # ------------------------------
    # High-level processing
    # ------------------------------
    def process(self, img_bytes: bytes) -> List[str]:
        """Detect faces and produce passport images encoded as base64 JPEG strings."""
        img_rgb = self._decode_image_bytes(img_bytes)
        if img_rgb is None:
            return []

        faces = self._detect_faces(img_rgb)
        if not faces:
            return []

        tgt_w_px, tgt_h_px, tgt_aspect = self._target_pixels()
        outputs: List[str] = []

        for face in faces:
            box = self._compute_crop_box(face, img_rgb.shape, tgt_aspect)
            resized = self._crop_and_resize(img_rgb, box, (tgt_w_px, tgt_h_px))
            final_img = self._apply_blue_background(resized)
            outputs.append(self._image_to_base64(final_img))

        return outputs

    def save_base64_outputs(self, outputs: List[str], prefix: str = "output", output_dir: str = ".") -> List[str]:
        """Save base64 JPEG strings to files named {prefix}_face{i}.jpg inside output_dir. Returns list of paths."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        for i, b64 in enumerate(outputs, start=1):
            img_data = base64.b64decode(b64)
            filename = f"{prefix}_face{i}.jpg"
            path = out_dir / filename
            # If file exists, append a numeric suffix to avoid overwriting
            count = 1
            final_path = path
            while final_path.exists():
                final_path = out_dir / f"{prefix}_face{i}_{count}.jpg"
                count += 1
            with open(final_path, "wb") as f:
                f.write(img_data)
            paths.append(str(final_path))
        return paths