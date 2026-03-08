"""
generator.py
------------
Wraps the existing BatchProcessor + PassportPhotoMaker pipeline so
the GUI can call it in a background thread via a simple interface.

The job runs in **watch mode by default**: it processes whatever images
are present in the input folder, sleeps for `poll_interval` seconds,
then repeats — until `cancel()` is called from the UI thread.

Progress / log messages are pushed to a thread-safe queue so the GUI
can poll and display them without blocking the UI thread.
"""

import logging
import queue
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so legacy modules are importable
# regardless of how the GUI is launched.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from batch_photo_processor import BatchProcessor  # noqa: E402
from passport_photo_maker import PassportPhotoMaker  # noqa: E402
from app.core.folder_manager import FolderManager  # noqa: E402

logger = logging.getLogger(__name__)


class QueueLogHandler(logging.Handler):
    """Redirect log records into a queue as plain strings."""

    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.log_queue.put(msg)


class GeneratorJob:
    """
    Watches the input folder and runs the passport-photo generation
    pipeline repeatedly until `cancel()` is called.

    Parameters
    ----------
    base_dir:
        Root input directory containing raw camera photos.
    log_queue:
        Thread-safe queue where log strings are pushed.
    poll_interval:
        Seconds to wait between scan cycles (default 10 s).
    on_cycle:
        Optional callback invoked after each cycle with
        (total_files_scanned, total_outputs_produced).
    on_stopped:
        Optional callback invoked once when the watch loop exits.
    """

    def __init__(
        self,
        base_dir: str | Path,
        log_queue: queue.Queue,
        poll_interval: int = 10,
        on_cycle: Optional[Callable[[int, int], None]] = None,
        on_stopped: Optional[Callable[[], None]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.log_queue = log_queue
        self.poll_interval = poll_interval
        self.on_cycle = on_cycle
        self.on_stopped = on_stopped

        self._cancelled = False
        self._total_files = 0
        self._total_outputs = 0

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
        self._push(f"[INFO] Generator watching: {self.base_dir}  "
                   f"(poll every {self.poll_interval}s — click Stop to halt)")

        try:
            # Ensure folder structure once at startup
            fm = FolderManager(self.base_dir)
            fm.ensure_structure()

            # Resolve the Haar cascade to an absolute path so it works
            # regardless of the process working directory.
            cascade_path = _ROOT / "haarcascade_frontalface_default.xml"
            if not cascade_path.exists():
                self._push(
                    f"[ERROR] Haar cascade not found at: {cascade_path}\n"
                    "        Place 'haarcascade_frontalface_default.xml' in the "
                    "project root directory."
                )
                return

            maker = PassportPhotoMaker(cascade_path=str(cascade_path))
            processor = BatchProcessor(str(self.base_dir), maker)

            # Patch process_file for per-file logging
            original_process_file = processor.process_file

            def _tracked_process_file(path: Path) -> int:
                if self._cancelled:
                    return 0
                self._push(f"[INFO] Processing: {path.name}")
                result = original_process_file(path)
                self._push(f"[INFO] → {result} output(s) saved for {path.name}")
                return result

            processor.process_file = _tracked_process_file  # type: ignore[method-assign]

            cycle = 0
            while not self._cancelled:
                cycle += 1
                self._push(f"[INFO] Generator — cycle #{cycle} …")

                files, outputs = processor.process_once()
                self._total_files += files
                self._total_outputs += outputs

                if files == 0:
                    self._push("[INFO] No new images found. Waiting …")
                else:
                    self._push(
                        f"[INFO] Cycle #{cycle} complete — "
                        f"{files} scanned, {outputs} saved."
                    )

                if self.on_cycle:
                    try:
                        self.on_cycle(files, outputs)
                    except Exception:
                        pass

                # Interruptible sleep: check cancel flag every 0.5 s
                deadline = time.monotonic() + self.poll_interval
                while not self._cancelled and time.monotonic() < deadline:
                    time.sleep(0.5)

        except Exception as exc:
            logger.exception("Generator watch loop failed")
            self._push(f"[ERROR] Generator failed: {exc}")

        finally:
            self._push(
                f"[INFO] Generator stopped. "
                f"Total — {self._total_files} scanned, "
                f"{self._total_outputs} saved."
            )
            if self.on_stopped:
                try:
                    self.on_stopped()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push(self, message: str) -> None:
        self.log_queue.put(message)
        logger.info(message)

