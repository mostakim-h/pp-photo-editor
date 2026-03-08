"""
layout_processor.py
-------------------
Wraps the A4PassportLayout pipeline for use in a background thread.

Watch behaviour
~~~~~~~~~~~~~~~
The job polls the Drop to Print folder every `poll_interval` seconds.
It will ONLY generate a layout page when there are AT LEAST 7 images
present — one full page worth.  If fewer than 7 images are found the
job logs the current count and keeps waiting.

Each batch of 7 images is processed and moved to the Completed folder;
then the watcher immediately checks again so back-to-back batches of 7
are handled without waiting for the next poll.

The loop runs until `cancel()` is called from the UI thread.
"""

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a4_passport_photo_layout import A4PassportLayout  # noqa: E402
from app.core.folder_manager import FolderManager       # noqa: E402

logger = logging.getLogger(__name__)

# Minimum images required before a layout page is generated
MIN_IMAGES_PER_PAGE = 7
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class LayoutProcessorJob:
    """
    Watches Drop to Print and generates A4 layouts in batches of 7.

    Parameters
    ----------
    base_dir      : Root input directory.
    log_queue     : Thread-safe queue for log strings.
    poll_interval : Seconds between scans (default 10 s).
    auto_print    : Send generated pages to *printer_name* when True.
    printer_name  : Windows printer name (ignored if auto_print=False).
    copies        : Number of print copies.
    on_cycle      : Callback(pages_generated) after each processing cycle.
    on_stopped    : Callback() once when the loop exits.
    """

    def __init__(
        self,
        base_dir: str | Path,
        log_queue: queue.Queue,
        poll_interval: int = 10,
        auto_print: bool = False,
        printer_name: Optional[str] = None,
        copies: int = 1,
        force_last_page: Optional[threading.Event] = None,
        on_cycle: Optional[Callable[[int], None]] = None,
        on_stopped: Optional[Callable[[], None]] = None,
    ) -> None:
        self.base_dir        = Path(base_dir)
        self.log_queue       = log_queue
        self.poll_interval   = poll_interval
        self.auto_print      = auto_print
        self.printer_name    = printer_name
        self.copies          = copies
        # Shared event: when set(), process whatever images exist (even < 7)
        self.force_last_page = force_last_page or threading.Event()
        self.on_cycle        = on_cycle
        self.on_stopped      = on_stopped

        self._cancelled  = False
        self._total_pages = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Signal the watch loop to stop after the current cycle."""
        self._cancelled = True

    def run(self) -> None:
        """Start the watch loop (call from a worker thread)."""
        self._cancelled = False
        self._push(
            f"[INFO] Layout Processor watching Drop to Print …  "
            f"(need {MIN_IMAGES_PER_PAGE} images per page — "
            f"poll every {self.poll_interval}s — click Stop to halt)"
        )

        try:
            fm = FolderManager(self.base_dir)
            paths = fm.ensure_structure()
            drop_dir    = paths["drop_to_print"]
            printed_dir = paths["printed"]
            completed_dir = drop_dir / "Completed"

            self._push(f"[INFO] Watching     : {drop_dir}")
            self._push(f"[INFO] Layouts saved: {printed_dir}")

            while not self._cancelled:
                images = self._list_images(drop_dir)
                count  = len(images)

                # Check if the UI requested a forced last-page print
                force = self.force_last_page.is_set()
                if force:
                    self.force_last_page.clear()

                if count == 0:
                    self._push("[INFO] Drop to Print is empty. Waiting …")

                elif count < MIN_IMAGES_PER_PAGE and not force:
                    self._push(
                        f"[INFO] {count}/{MIN_IMAGES_PER_PAGE} images ready — "
                        f"waiting for {MIN_IMAGES_PER_PAGE - count} more …"
                    )

                else:
                    # forced=True → process whatever is there (even < 7)
                    # forced=False → only process complete batches of 7
                    if force and count < MIN_IMAGES_PER_PAGE:
                        self._push(
                            f"[INFO] Force-print requested — processing {count} image(s) "
                            f"as last page (less than {MIN_IMAGES_PER_PAGE}) …"
                        )

                    pages_this_round = 0
                    # With force, first drain any partial batch, then continue with full ones
                    first_batch_min = 1 if force else MIN_IMAGES_PER_PAGE

                    while len(images) >= first_batch_min and not self._cancelled:
                        batch = images[:MIN_IMAGES_PER_PAGE]

                        self._push(
                            f"[INFO] Processing batch of {len(batch)} image(s) → layout page …"
                        )
                        pages = self._generate_page(batch, printed_dir, completed_dir)
                        self._total_pages += pages
                        pages_this_round  += pages

                        if pages:
                            self._push(
                                f"[INFO] Page generated. "
                                f"Total pages so far: {self._total_pages}"
                            )
                            if self.auto_print:
                                self._auto_print(printed_dir)

                        # After first (possibly partial) batch, enforce the 7-minimum again
                        first_batch_min = MIN_IMAGES_PER_PAGE

                        # Refresh image list after files move to Completed
                        images = self._list_images(drop_dir)

                    if pages_this_round and self.on_cycle:
                        try:
                            self.on_cycle(pages_this_round)
                        except Exception:
                            pass

                    leftover = len(images)
                    if leftover:
                        self._push(
                            f"[INFO] {leftover} image(s) remain in Drop to Print — "
                            f"waiting for {MIN_IMAGES_PER_PAGE - leftover} more …"
                        )

                # Interruptible sleep — also wakes immediately if force_last_page is set
                deadline = time.monotonic() + self.poll_interval
                while not self._cancelled and time.monotonic() < deadline:
                    if self.force_last_page.is_set():
                        break   # wake up early to handle force request
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

    def _list_images(self, folder: Path) -> List[Path]:
        """Return sorted list of image files directly inside *folder*."""
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    def _generate_page(
        self,
        batch: List[Path],
        printed_dir: Path,
        completed_dir: Path,
    ) -> int:
        """
        Run A4PassportLayout.create_pages() for *batch* and return pages created.
        Images are moved to *completed_dir* by create_pages() itself.
        """
        printed_dir.mkdir(parents=True, exist_ok=True)
        completed_dir.mkdir(parents=True, exist_ok=True)

        layout = A4PassportLayout()
        pages = layout.create_pages(
            image_paths=[str(p) for p in batch],
            printed_dir=str(printed_dir),
            completed_dir=str(completed_dir),
        )
        return pages

    def _auto_print(self, printed_dir: Path) -> None:
        """Send freshly created layout pages to the printer."""
        try:
            import importlib
            pm_module    = importlib.import_module("app.printing.printer_manager")
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

