"""
main.py
-------
Entry point for the Passport Photo Processing System GUI.

Usage:
    python app/main.py
    python -m app.main        (from the project root)
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so all existing modules
# (passport_photo_maker, batch_photo_processor, etc.) are importable
# regardless of where Python is invoked from.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging setup — must happen before any app imports so all loggers inherit
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# App imports (after sys.path is patched)
# ---------------------------------------------------------------------------
import customtkinter as ctk

from app.config.settings import Settings
from app.gui.main_window import MainWindow


def main() -> None:
    # Load persisted settings
    settings = Settings()

    # Apply saved appearance before creating any widgets
    mode = settings.get("ui.appearance_mode", "dark")
    ctk.set_appearance_mode(mode)
    ctk.set_default_color_theme("blue")

    # Create and run the main window
    app = MainWindow(settings)
    app.mainloop()


if __name__ == "__main__":
    main()

