from PIL import Image
import cv2
import numpy as np
from rembg import remove
import io
import base64
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

BLUE_BG_COLOR = (0, 102, 204)  # Passport blue (RGB)

logger = logging.getLogger(__name__)


class FaceDetector:
    """Multi-stage face detection with strict validation to eliminate false positives."""

    def __init__(
        self,
        cascade_path: str = "haarcascade_frontalface_default.xml",
        dnn_model_path: Optional[str] = None,
        dnn_config_path: Optional[str] = None,
        min_confidence: float = 0.7,
        min_face_size_px: int = 80,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_face_size_px = min_face_size_px

        # Try to load DNN-based face detector (primary method)
        self.dnn_detector = None
        if dnn_model_path and dnn_config_path and os.path.exists(dnn_model_path):
            try:
                self.dnn_detector = cv2.dnn.readNetFromCaffe(dnn_config_path, dnn_model_path)
                logger.info("DNN face detector loaded successfully")
            except Exception as e:
                logger.warning("Failed to load DNN detector: %s", e)

        # Load Haar Cascade as fallback (with strict validation)
        self.haar_cascade = None
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            if self.haar_cascade.empty():
                self.haar_cascade = None
                logger.warning("Haar cascade file is invalid")
            else:
                logger.info("Haar cascade loaded as fallback detector")

        if not self.dnn_detector and not self.haar_cascade:
            raise FileNotFoundError(
                "No face detector available. Provide either DNN model or Haar cascade."
            )

    def _detect_faces_dnn(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using DNN. Returns list of (x, y, w, h, confidence)."""
        if self.dnn_detector is None:
            return []

        h, w = img_rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_rgb, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Convert to (x, y, w, h) format
                face_w = x2 - x1
                face_h = y2 - y1

                # Filter by minimum size
                if face_w >= self.min_face_size_px and face_h >= self.min_face_size_px:
                    faces.append((x1, y1, face_w, face_h, float(confidence)))

        return faces

    def _detect_faces_haar(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using Haar Cascade with strict parameters. Returns list of (x, y, w, h, confidence)."""
        if self.haar_cascade is None:
            return []

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Use stricter parameters to reduce false positives
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,  # Increased from 5 to reduce false positives
            minSize=(self.min_face_size_px, self.min_face_size_px),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Assign pseudo-confidence (Haar doesn't provide confidence)
        return [(int(x), int(y), int(w), int(h), 0.75) for x, y, w, h in faces]

    def _validate_face_geometry(self, x: int, y: int, w: int, h: int, img_shape: Tuple[int, int, int]) -> bool:
        """Validate that detected region has realistic face proportions."""
        img_h, img_w = img_shape[:2]

        # Face aspect ratio should be roughly 0.7-1.3 (width/height)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            logger.debug("Rejected: invalid aspect ratio %.2f", aspect_ratio)
            return False

        # Face should not be too large (occupying >70% of image)
        area_ratio = (w * h) / (img_w * img_h)
        if area_ratio > 0.7:
            logger.debug("Rejected: face too large (%.1f%% of image)", area_ratio * 100)
            return False

        # Face should not be too small relative to image
        if area_ratio < 0.01:
            logger.debug("Rejected: face too small (%.1f%% of image)", area_ratio * 100)
            return False

        # Face should not be at extreme edges
        margin = 0.05  # 5% margin
        if (x < img_w * margin or y < img_h * margin or
            x + w > img_w * (1 - margin) or y + h > img_h * (1 - margin)):
            logger.debug("Rejected: face too close to edge")
            return False

        return True

    def _validate_face_content(self, face_region: np.ndarray) -> bool:
        """Validate that the region contains face-like features (eyes, nose region)."""
        if face_region.size == 0:
            return False

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY) if len(face_region.shape) == 3 else face_region

        # Check intensity distribution - faces typically have varied intensity
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Too uniform = likely not a face
        if std_intensity < 15:
            logger.debug("Rejected: too uniform (std=%.1f)", std_intensity)
            return False

        # Check for reasonable contrast
        if mean_intensity < 30 or mean_intensity > 225:
            logger.debug("Rejected: extreme brightness (mean=%.1f)", mean_intensity)
            return False

        # Detect eye-like regions using Haar cascade
        if self.haar_cascade:
            h, w = gray.shape[:2]
            # Eyes are typically in upper 60% of face
            eye_region = gray[int(h * 0.2):int(h * 0.6), :]

            # Use simple edge detection to check for facial features
            edges = cv2.Canny(eye_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Faces should have moderate edge density (features but not noise)
            if edge_density < 0.05 or edge_density > 0.4:
                logger.debug("Rejected: unusual edge density %.3f", edge_density)
                return False

        return True

    def detect(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect and validate faces with multi-stage filtering.
        Returns list of (x, y, w, h, confidence) for valid faces only.
        """
        # Stage 1: Primary detection (DNN preferred, Haar fallback)
        faces = self._detect_faces_dnn(img_rgb)
        if not faces and self.haar_cascade:
            logger.debug("DNN found no faces, trying Haar cascade")
            faces = self._detect_faces_haar(img_rgb)

        if not faces:
            logger.info("No faces detected by any detector")
            return []

        logger.debug("Initial detection found %d candidates", len(faces))

        # Stage 2: Geometric validation
        valid_faces = []
        for x, y, w, h, conf in faces:
            if not self._validate_face_geometry(x, y, w, h, img_rgb.shape):
                continue

            # Stage 3: Content validation
            face_region = img_rgb[y:y+h, x:x+w]
            if not self._validate_face_content(face_region):
                continue

            valid_faces.append((x, y, w, h, conf))

        logger.info("Validated %d real faces out of %d candidates", len(valid_faces), len(faces))
        return valid_faces


class PassportPhotoMaker:
    """Utility to detect faces and produce passport-style photos with a blue background.

    Public methods:
    - process(img_bytes) -> List[str]: returns list of base64 JPEG strings (one per detected face)
    - process_from_path(img_path) -> List[str]: process image from file path
    - save_base64_outputs(outputs, prefix, output_dir) -> List[str]: saves base64 images to files and returns filepaths
    """

    def __init__(
        self,
        cascade_path: str = "haarcascade_frontalface_default.xml",
        dnn_model_path: Optional[str] = None,
        dnn_config_path: Optional[str] = None,
        dpi: int = 300,
        target_inches: Tuple[float, float] = (1.5, 1.9),
        expand_factor: float = 1.6,
        min_face_size_px: int = 80,
        min_confidence: float = 0.7,
        blue_bg_color: Tuple[int, int, int] = BLUE_BG_COLOR,
    ) -> None:
        self.dpi = dpi
        self.target_inches = target_inches
        self.expand_factor = expand_factor
        self.blue_bg_color = blue_bg_color

        # Choose a resampling filter compatible with Pillow versions
        try:
            self._resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            if hasattr(Image, "LANCZOS"):
                self._resample = Image.LANCZOS  # type: ignore[attr-defined]
            elif hasattr(Image, "ANTIALIAS"):
                self._resample = Image.ANTIALIAS  # type: ignore[attr-defined]
            else:
                self._resample = Image.NEAREST  # type: ignore[attr-defined]

        # Initialize robust face detector
        self.face_detector = FaceDetector(
            cascade_path=cascade_path,
            dnn_model_path=dnn_model_path,
            dnn_config_path=dnn_config_path,
            min_confidence=min_confidence,
            min_face_size_px=min_face_size_px,
        )

    # ------------------------------
    # Low-level helpers
    # ------------------------------
    def _decode_image_bytes(self, img_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image bytes into an RGB numpy array. Returns None if decoding fails."""
        try:
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.warning("Failed to decode image bytes")
                return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            logger.error("Error decoding image: %s", e)
            return None

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
        try:
            img_no_bg = remove(img_pil).convert("RGBA")
            bg = Image.new("RGBA", img_no_bg.size, self.blue_bg_color + (255,))
            final_img = Image.alpha_composite(bg, img_no_bg).convert("RGB")
            return final_img
        except Exception as e:
            logger.error("Background removal failed: %s", e)
            # Return original image with blue background as fallback
            bg = Image.new("RGB", img_pil.size, self.blue_bg_color)
            return img_pil

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

        # Use the robust face detector
        faces = self.face_detector.detect(img_rgb)
        if not faces:
            logger.info("No valid faces detected in image")
            return []

        tgt_w_px, tgt_h_px, tgt_aspect = self._target_pixels()
        outputs: List[str] = []

        for i, face in enumerate(faces, start=1):
            # face is (x, y, w, h, confidence)
            x, y, w, h, conf = face
            logger.info("Processing face %d/%d (confidence: %.2f)", i, len(faces), conf)

            box = self._compute_crop_box((x, y, w, h), img_rgb.shape, tgt_aspect)
            resized = self._crop_and_resize(img_rgb, box, (tgt_w_px, tgt_h_px))
            final_img = self._apply_blue_background(resized)
            outputs.append(self._image_to_base64(final_img))

        logger.info("Successfully processed %d faces", len(outputs))
        return outputs

    def process_from_path(self, img_path: str) -> List[str]:
        """Process an image from a file path."""
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            return self.process(img_bytes)
        except Exception as e:
            logger.error("Failed to process image from path %s: %s", img_path, e)
            return []

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