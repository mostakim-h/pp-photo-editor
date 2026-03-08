"""
settings.py
-----------
Centralised settings manager that loads/saves a JSON configuration
file.  Safe to use from any module — changes are persisted to disk.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default file lives next to this module
_DEFAULT_PATH = Path(__file__).parent / "settings.json"

# Default values (merged on top of the saved file on load)
_DEFAULTS: dict = {
    "input_dir": "",
    "generator": {
        "poll_interval": 10,
    },
    "layout": {
        "poll_interval": 10,
    },
    "printer": {
        "name": "",
        "paper_size": "4x6",
        "copies": 1,
        "auto_print": False,
    },
    "ui": {
        "appearance_mode": "dark",
        "color_theme": "blue",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates and returns base)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Settings:
    """
    Key/value settings store backed by a JSON file.

    Usage::

        cfg = Settings()
        cfg.load()
        cfg.set("input_dir", "/photos")
        cfg.set("printer.copies", 2)       # dot-notation for nested keys
        cfg.save()

        val = cfg.get("printer.auto_print")  # False
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._data: dict = {}
        self.load()

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load settings from disk, merging with defaults for missing keys."""
        import copy

        base = copy.deepcopy(_DEFAULTS)
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                _deep_merge(base, saved)
                logger.debug("Settings loaded from %s", self._path)
            except Exception as exc:
                logger.warning("Could not load settings (%s) — using defaults.", exc)
        self._data = base

    def save(self) -> None:
        """Persist current settings to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=4)
            logger.debug("Settings saved to %s", self._path)
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)

    # ------------------------------------------------------------------
    # Get / Set  (dot-notation for nested keys, e.g. "printer.copies")
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value by *key*.  Use dot notation for nested access::

            cfg.get("printer.copies")  →  1
        """
        parts = key.split(".")
        node = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        """
        Set *value* at *key*.  Nested keys are created as needed::

            cfg.set("printer.auto_print", True)
        """
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def all(self) -> dict:
        """Return a copy of the entire settings dict."""
        import copy

        return copy.deepcopy(self._data)

