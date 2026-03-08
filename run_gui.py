"""
run_gui.py
----------
Convenience launcher at the project root.

Usage:
    python run_gui.py
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path (same as app/main.py does)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.main import main

if __name__ == "__main__":
    main()

