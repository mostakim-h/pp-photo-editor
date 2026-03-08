"""
layout_processor.py
-------------------
Wraps batch_process_passport_photos_layout() for use in a background
thread.

The job runs in **watch mode**: it checks the Drop to Print folder,
generates layouts for any images it finds, sleeps for `poll_interval`
seconds, then repeats — until `cancel()` is called from the UI thread.

Optional auto-print: if a printer name is supplied the generated layout
pages are sent to that printer via printer_manager.
"""

import logging
import queue
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch_process_passport_photos_layout import (  # noqa: E402
    batch_process_passport_photos_layout,
)
from app.core.folder_manager import FolderManager  # noqa: E402

logger = logging.getLogger(__name__)


class LayoutProcessorJob:
    """
    Watches the Drop to Print folder and generates layouts repeatedly
    until `cancel()` is called.

    Parameters
    ----------
    base_dir:
        Root input directory (must already contain Edited Photos/Drop to Print).
    log_queue:
        Thread-safe queue where log strings are pushed.
    poll_interval:
        Seconds to wait between scan cycles (default 10 s).
    auto_print:
        When True, send generated pages to *printer_name*.
    printer_name:
        Name of the Windows printer to use (ignored when auto_print=False).
    copies:
        Number of copies to send to the printer.
    on_cycle:
        Optional callback invoked after each cycle with (pages_generated: int).
    on_stopped:
        Optional callback invoked once when the watch loop exits.
    """

    def __init__(
        self,
        base_dir: str | Path,
        log_queue: queue.Queue,
        poll_interval: int = 10,
        auto_print: bool = False,
        printer_name: Optional[str] = None,
        copies: int = 1,
        on_cycle: Optional[Callable[[int], None]] = None,
        on_stopped: Optional[Callable[[], None]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.log_queue = log_queue
        self.poll_interval = poll_interval
        self.auto_print = auto_print
        self.printer_name = printer_name
        self.copies = copies
        self.on_cycle = on_cycle
        self.on_stopped = on_stopped

        self._cancelled = False
        self._total_pages = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Signal the watch loop to stop after the current cycle."""
        self._cancelled = True

    def run(self) -> None:
        """
        Start the watch loop (call from a worker thread).
        Runs until cancel() is called.
        """
        self._cancelled = False
        self._push(
            f"[INFO] Layout Processor watching Drop to Print …  "
            f"(poll every {self.poll_interval}s — click Stop to halt)"
        )

        try:
            # Ensure folder structure once at startup
            fm = FolderManager(self.base_dir)
            paths = fm.ensure_structure()
            drop_dir = paths["drop_to_print"]
            printed_dir = paths["printed"]

            self._push(f"[INFO] Reading from : {drop_dir}")
            self._push(f"[INFO] Saving to    : {printed_dir}")

            cycle = 0
            while not self._cancelled:
                cycle += 1
                self._push(f"[INFO] Layout — cycle #{cycle} …")

                pages = batch_process_passport_photos_layout(
                    drop_to_print_dir=str(drop_dir),
                    printed_dir=str(printed_dir),
                )
                self._total_pages += pages

                if pages == 0:
                    self._push("[INFO] No images ready in Drop to Print. Waiting …")
                else:
                    self._push(f"[INFO] Cycle #{cycle} — {pages} page(s) generated.")
                    if self.auto_print:
                        self._auto_print(printed_dir)

                if self.on_cycle:
                    try:
                        self.on_cycle(pages)
                    except Exception:
                        pass

                # Interruptible sleep: check cancel flag every 0.5 s
                deadline = time.monotonic() + self.poll_interval
                while not self._cancelled and time.monotonic() < deadline:
                    time.sleep(0.5)

        except Exception as exc:
            logger.exception("Layout watch loop failed")
            self._push(f"[ERROR] Layout Processor failed: {exc}")

        finally:
            self._push(
                f"[INFO] Layout Processor stopped. "
                f"Total pages generated: {self._total_pages}."
            )
            if self.on_stopped:
                try:
                    self.on_stopped()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_print(self, printed_dir: Path) -> None:
        """Send freshly created layout pages to the printer."""
        try:
            import importlib
            pm_module = importlib.import_module("app.printing.printer_manager")
            PrinterManager = pm_module.PrinterManager

            pm = PrinterManager()
            layout_files = sorted(printed_dir.glob("page_*.jpg"))
            if not layout_files:
                self._push("[WARN] Auto-print: no layout files found.")
                return

            for page_path in layout_files:
                self._push(
                    f"[INFO] Printing {page_path.name} → "
                    f"'{self.printer_name}' (×{self.copies}) …"
                )
                pm.print_file(
                    file_path=str(page_path),
                    printer_name=self.printer_name,
                    copies=self.copies,
                )
            self._push("[INFO] Auto-print complete.")

        except Exception as exc:
            self._push(f"[ERROR] Auto-print failed: {exc}")

    def _push(self, message: str) -> None:
        self.log_queue.put(message)
        logger.info(message)

