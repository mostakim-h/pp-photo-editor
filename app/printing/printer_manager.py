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

    def open_printer_settings(
        self,
        printer_name: Optional[str] = None,
        parent_hwnd: int = 0,
    ) -> None:
        """
        Open the full printer driver Printing Preferences dialog for
        *printer_name* (or the system default).

        Gives the user direct access to paper type, media, print quality,
        colour mode, layout — everything the printer driver exposes.

        Parameters
        ----------
        printer_name:
            Name of the printer to configure.  Defaults to the system default.
        parent_hwnd:
            Native window handle of the calling window.  Pass the Tkinter
            root's winfo_id() for proper dialog parenting (optional).

        Strategy (tried in order):
          1. rundll32 printui.dll,PrintUIEntry /p  — Printing Preferences sheet
             (always opens as a visible, foreground window)
          2. win32print.DocumentProperties with parent_hwnd
          3. Open Devices and Printers folder as last resort
        """
        target = printer_name or self.get_default_printer()
        if not target:
            raise ValueError("No printer specified and no default printer found.")

        logger.info("Opening printer settings for: %s", target)

        # ---- Strategy 1: rundll32 printui.dll /p --------------------------
        # /p  = Printing Preferences (driver properties dialog)
        # This is the most reliable way to get a visible foreground dialog.
        try:
            import subprocess
            result = subprocess.Popen(
                [
                    "rundll32.exe",
                    "printui.dll,PrintUIEntry",
                    "/p",          # open Printing Preferences dialog
                    "/n", target,
                ],
                shell=False,
            )
            logger.info(
                "rundll32 printui.dll /p launched for '%s' (pid=%s)",
                target, result.pid,
            )
            return
        except Exception as exc:
            logger.warning(
                "rundll32 printui.dll /p failed (%s) — trying DocumentProperties", exc
            )

        # ---- Strategy 2: win32print.DocumentProperties --------------------
        # Requires a valid hwnd; falls back to 0 if none provided.
        if _WIN32PRINT_AVAILABLE:
            try:
                import win32print  # type: ignore
                import win32con    # type: ignore
                import ctypes

                h_printer = win32print.OpenPrinter(target)
                try:
                    hwnd = parent_hwnd or 0
                    win32print.DocumentProperties(
                        hwnd,
                        h_printer,
                        target,
                        None,
                        None,
                        win32con.DM_IN_PROMPT | win32con.DM_OUT_BUFFER,
                    )
                    logger.info("DocumentProperties dialog closed for '%s'.", target)
                    return
                finally:
                    win32print.ClosePrinter(h_printer)
            except Exception as exc:
                logger.warning(
                    "DocumentProperties failed (%s) — opening Devices & Printers", exc
                )

        # ---- Strategy 3: Devices and Printers folder ----------------------
        try:
            import subprocess
            subprocess.Popen(["explorer.exe", "shell:PrintersFolder"])
            logger.info("Opened Devices and Printers folder as fallback.")
        except Exception as exc:
            logger.error("All strategies to open printer settings failed: %s", exc)
            raise RuntimeError(
                f"Could not open printer settings for '{target}': {exc}"
            ) from exc



