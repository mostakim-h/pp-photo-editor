"""
folder_manager.py
-----------------
Responsible for creating and validating the required directory structure.

Expected structure:
    <base_dir>/
        Edited Photos/
            Drop to Print/
            Printed/
        Processed Raw/
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class FolderManager:
    """Creates and validates the passport photo directory tree."""

    SUBFOLDERS = (
        "Edited Photos",
        "Edited Photos/Drop to Print",
        "Edited Photos/Printed",
        "Processed Raw",
    )

    def __init__(self, base_dir: str | Path) -> None:
        self.base = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_structure(self) -> Dict[str, Path]:
        """
        Create any missing folders and return a dict with named paths.

        Returns:
            {
                "base":          Path to the input/base directory,
                "edited":        Path to "Edited Photos",
                "drop_to_print": Path to "Edited Photos/Drop to Print",
                "printed":       Path to "Edited Photos/Printed",
                "processed_raw": Path to "Processed Raw",
            }
        """
        self.base.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {"base": self.base}

        for rel in self.SUBFOLDERS:
            full = self.base / rel
            full.mkdir(parents=True, exist_ok=True)
            key = Path(rel).name.lower().replace(" ", "_")
            paths[key] = full
            logger.debug("Ensured directory: %s", full)

        logger.info("Directory structure verified under: %s", self.base)
        return paths

    def validate(self) -> bool:
        """Return True only if all required sub-folders already exist."""
        for rel in self.SUBFOLDERS:
            if not (self.base / rel).is_dir():
                return False
        return True

    def get_paths(self) -> Dict[str, Path]:
        """Return named paths dict (same keys as ensure_structure)."""
        return {
            "base":          self.base,
            "edited":        self.base / "Edited Photos",
            "drop_to_print": self.base / "Edited Photos" / "Drop to Print",
            "printed":       self.base / "Edited Photos" / "Printed",
            "processed_raw": self.base / "Processed Raw",
        }

