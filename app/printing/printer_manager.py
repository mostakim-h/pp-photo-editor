"""
printer_manager.py
------------------
Handles printer detection and silent print-job submission on Windows.

print_file() sends a JPEG/PNG directly to the printer via the Windows
GDI API (win32print + win32ui + PIL ImageWin).  This is fully silent —
no dialog, no preview, no associated-application pop-up.  The printer
driver uses whatever settings the user last saved through the
Printer Settings dialog.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import win32print   # type: ignore
    import win32ui      # type: ignore
    import win32con     # type: ignore
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False
    logger.warning(
        "pywin32 not fully installed — printing unavailable. "
        "Run: pip install pywin32"
    )


class PrinterManager:
    """Thin abstraction over the Windows printing subsystem."""

    # ------------------------------------------------------------------
    # Printer discovery
    # ------------------------------------------------------------------

    def list_printers(self) -> List[str]:
        """Return the names of all locally installed printers."""
        if not _WIN32_AVAILABLE:
            return []
        try:
            printers = win32print.EnumPrinters(
                win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS,
                None, 1,
            )
            return [p[2] for p in printers]
        except Exception as exc:
            logger.error("Failed to enumerate printers: %s", exc)
            return []

    def get_default_printer(self) -> Optional[str]:
        """Return the name of the system default printer, or None."""
        if not _WIN32_AVAILABLE:
            return None
        try:
            return win32print.GetDefaultPrinter()
        except Exception as exc:
            logger.error("Could not get default printer: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Silent direct printing
    # ------------------------------------------------------------------

    def print_file(
        self,
        file_path: str | Path,
        printer_name: Optional[str] = None,
        copies: int = 1,
    ) -> None:
        """
        Send *file_path* silently and directly to *printer_name*.

        Uses the Windows GDI API to render the image straight onto the
        printer device context.  No dialog, no preview window, no
        associated application is opened.  The printer driver uses
        whatever paper/quality/orientation settings were last saved.

        Parameters
        ----------
        file_path    : Path to the image file (JPEG, PNG, BMP, TIFF …).
        printer_name : Target printer name.  Defaults to system default.
        copies       : Number of copies to print.
        """
        if sys.platform != "win32":
            raise NotImplementedError("Silent printing is only supported on Windows.")

        if not _WIN32_AVAILABLE:
            raise RuntimeError(
                "pywin32 is not installed. Run: pip install pywin32"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        target = printer_name or self.get_default_printer()
        if not target:
            raise ValueError("No printer specified and no default printer found.")

        logger.info(
            "Silent print: '%s' → '%s'  (×%d)",
            file_path.name, target, copies,
        )

        for copy_num in range(1, copies + 1):
            logger.debug("  Copy %d/%d …", copy_num, copies)
            self._gdi_print(file_path, target)

        logger.info("Print job(s) submitted for '%s'.", file_path.name)

    # ------------------------------------------------------------------
    # GDI rendering — the actual silent print engine
    # ------------------------------------------------------------------

    def _gdi_print(self, file_path: Path, printer_name: str) -> None:
        """
        Render *file_path* onto *printer_name* via the Windows GDI API.

        Steps:
          1. Open the printer and retrieve its DEVMODE (stored driver settings).
          2. Create a printer Device Context (DC) from that DEVMODE so the
             driver's paper/quality/orientation settings are honoured.
          3. Open the image with Pillow and scale it to fill the printable area
             while preserving the aspect ratio.
          4. Use PIL.ImageWin.Dib to blit the image onto the DC.
          5. Close the print job cleanly.
        """
        from PIL import Image, ImageWin  # type: ignore

        # ---- 1. Open printer & get stored DEVMODE -------------------------
        h_printer = win32print.OpenPrinter(printer_name)
        try:
            # Level-2 info gives us the DEVMODE with the user's stored settings
            printer_info = win32print.GetPrinter(h_printer, 2)
            devmode = printer_info.get("pDevMode")
        finally:
            win32print.ClosePrinter(h_printer)

        # ---- 2. Create a printer DC from the stored DEVMODE ---------------
        hdc = win32ui.CreateDC()
        hdc.CreatePrinterDC(printer_name)

        # Apply the stored DEVMODE settings onto the DC
        if devmode:
            try:
                hdc.ResetDC(devmode)
            except Exception as exc:
                logger.debug("ResetDC skipped (%s) — using driver defaults", exc)

        # ---- 3. Query printable area dimensions (in device pixels) --------
        # PHYSICALWIDTH / PHYSICALHEIGHT = total paper pixels
        # PHYSICALOFFSETX / PHYSICALOFFSETY = unprintable margin pixels
        phys_w   = hdc.GetDeviceCaps(win32con.PHYSICALWIDTH)
        phys_h   = hdc.GetDeviceCaps(win32con.PHYSICALHEIGHT)
        offset_x = hdc.GetDeviceCaps(win32con.PHYSICALOFFSETX)
        offset_y = hdc.GetDeviceCaps(win32con.PHYSICALOFFSETY)
        print_w  = hdc.GetDeviceCaps(win32con.HORZRES)   # printable width
        print_h  = hdc.GetDeviceCaps(win32con.VERTRES)   # printable height

        logger.debug(
            "Printer DC — phys=(%d×%d) offset=(%d,%d) printable=(%d×%d)",
            phys_w, phys_h, offset_x, offset_y, print_w, print_h,
        )

        # ---- 4. Open image and scale to fit the printable area ------------
        img = Image.open(file_path).convert("RGB")
        img_w, img_h = img.size

        # Scale to fill printable area, preserving aspect ratio
        scale = min(print_w / img_w, print_h / img_h)
        dest_w = int(img_w * scale)
        dest_h = int(img_h * scale)

        # Centre on the printable area
        dest_x = (print_w - dest_w) // 2
        dest_y = (print_h - dest_h) // 2

        # ---- 5. Send to printer -------------------------------------------
        hdc.StartDoc(file_path.name)
        hdc.StartPage()
        try:
            dib = ImageWin.Dib(img)
            dib.draw(
                hdc.GetHandleOutput(),
                (dest_x, dest_y, dest_x + dest_w, dest_y + dest_h),
            )
        finally:
            hdc.EndPage()
            hdc.EndDoc()
            hdc.DeleteDC()

        logger.debug(
            "GDI print complete: image=%dx%d → dest=%dx%d at (%d,%d)",
            img_w, img_h, dest_w, dest_h, dest_x, dest_y,
        )

    # ------------------------------------------------------------------
    # Open printer driver settings dialog
    # ------------------------------------------------------------------

    def open_printer_settings(
        self,
        printer_name: Optional[str] = None,
        parent_hwnd: int = 0,
    ) -> None:
        """
        Open the printer driver Printing Preferences dialog.

        Strategy (tried in order):
          1. rundll32 printui.dll,PrintUIEntry /p  — always opens foreground
          2. win32print.DocumentProperties with parent_hwnd
          3. Open Devices and Printers folder as last resort
        """
        target = printer_name or self.get_default_printer()
        if not target:
            raise ValueError("No printer specified and no default printer found.")

        logger.info("Opening printer settings for: %s", target)

        # ---- Strategy 1: rundll32 printui.dll /p --------------------------
        try:
            import subprocess
            proc = subprocess.Popen(
                ["rundll32.exe", "printui.dll,PrintUIEntry", "/p", "/n", target],
                shell=False,
            )
            logger.info("printui.dll /p launched for '%s' (pid=%s)", target, proc.pid)
            return
        except Exception as exc:
            logger.warning("rundll32 failed (%s) — trying DocumentProperties", exc)

        # ---- Strategy 2: win32print.DocumentProperties --------------------
        if _WIN32_AVAILABLE:
            try:
                h_printer = win32print.OpenPrinter(target)
                try:
                    win32print.DocumentProperties(
                        parent_hwnd, h_printer, target, None, None,
                        win32con.DM_IN_PROMPT | win32con.DM_OUT_BUFFER,
                    )
                    logger.info("DocumentProperties dialog closed for '%s'.", target)
                    return
                finally:
                    win32print.ClosePrinter(h_printer)
            except Exception as exc:
                logger.warning("DocumentProperties failed (%s)", exc)

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

