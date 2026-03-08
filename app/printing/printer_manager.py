"""
printer_manager.py
------------------
Handles printer detection and print-job submission on Windows.

Falls back gracefully on non-Windows platforms (detection returns an
empty list; print_file raises NotImplementedError).
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import win32print; if not available we degrade gracefully.
# ---------------------------------------------------------------------------
try:
    import win32print  # type: ignore

    _WIN32PRINT_AVAILABLE = True
except ImportError:
    _WIN32PRINT_AVAILABLE = False
    logger.warning(
        "pywin32 not installed — printer detection and printing are unavailable. "
        "Run: pip install pywin32"
    )


class PrinterManager:
    """
    Thin abstraction over the Windows printing subsystem.

    Typical usage::

        pm = PrinterManager()
        printers = pm.list_printers()          # ["Microsoft Print to PDF", ...]
        default  = pm.get_default_printer()    # "HP LaserJet" (or None)
        pm.print_file("layout.jpg", printer_name="HP LaserJet", copies=2)
    """

    # ------------------------------------------------------------------
    # Printer discovery
    # ------------------------------------------------------------------

    def list_printers(self) -> List[str]:
        """Return the names of all locally installed printers."""
        if not _WIN32PRINT_AVAILABLE:
            logger.warning("win32print unavailable — returning empty printer list.")
            return []

        try:
            # PRINTER_ENUM_LOCAL | PRINTER_ENUM_CONNECTIONS = 6
            printers = win32print.EnumPrinters(
                win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS,
                None,
                1,
            )
            names = [p[2] for p in printers]
            logger.debug("Detected printers: %s", names)
            return names
        except Exception as exc:
            logger.error("Failed to enumerate printers: %s", exc)
            return []

    def get_default_printer(self) -> Optional[str]:
        """Return the name of the system default printer, or None."""
        if not _WIN32PRINT_AVAILABLE:
            return None
        try:
            return win32print.GetDefaultPrinter()
        except Exception as exc:
            logger.error("Could not get default printer: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_file(
        self,
        file_path: str | Path,
        printer_name: Optional[str] = None,
        copies: int = 1,
    ) -> None:
        """
        Send *file_path* to *printer_name* (or the system default).

        Strategy:
          1. On Windows with win32print available → use ShellExecute "print"
          2. Fallback → raise NotImplementedError
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if sys.platform != "win32":
            raise NotImplementedError("Direct printing is only supported on Windows.")

        target_printer = printer_name or self.get_default_printer()
        if not target_printer:
            raise ValueError("No printer specified and no default printer found.")

        logger.info(
            "Printing '%s' to '%s' (%d cop%s) …",
            file_path.name,
            target_printer,
            copies,
            "y" if copies == 1 else "ies",
        )

        for copy_num in range(1, copies + 1):
            logger.debug("  Sending copy %d/%d …", copy_num, copies)
            self._shell_print(file_path, target_printer)

        logger.info("Print job submitted for '%s'.", file_path.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shell_print(self, file_path: Path, printer_name: str) -> None:
        """Use Windows ShellExecute to send the file to the printer."""
        try:
            import win32api  # type: ignore
            import win32con  # type: ignore

            win32api.ShellExecute(
                0,
                "print",
                str(file_path),
                f'/d:"{printer_name}"',
                ".",
                win32con.SW_HIDE,
            )
        except ImportError:
            # Fallback: use the built-in os.startfile print verb (less control)
            logger.warning(
                "win32api not available — falling back to os.startfile for printing."
            )
            os.startfile(str(file_path), "print")
        except Exception as exc:
            logger.error("ShellExecute print failed: %s", exc)
            raise


