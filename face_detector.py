import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional


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
