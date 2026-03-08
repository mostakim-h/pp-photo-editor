import shutil
import time
from pathlib import Path
from typing import List, Tuple
import logging

from passport_photo_maker import PassportPhotoMaker


class BatchProcessor:
    """Watches an input folder, processes camera photos once, saves edited outputs, and moves processed raws aside."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
            self,
            input_dir: str,
            maker: PassportPhotoMaker,
            poll_interval: int = 60,
    ) -> None:
        self.base = Path(input_dir)
        self.src = self.base
        self.edited = self.base / "Edited Photos"
        self.drop_to_print = self.edited / "Drop to Print"
        self.printed = self.edited / "Printed"
        self.processed_raw = self.base / "Processed Raw"
        self.poll_interval = poll_interval
        self.maker = maker

        # Ensure directories exist
        for d in (self.base, self.edited, self.drop_to_print, self.printed, self.processed_raw):
            d.mkdir(parents=True, exist_ok=True)

    def _list_images(self) -> List[Path]:
        return [p for p in self.src.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS]

    def process_file(self, path: Path) -> int:
        """Process a single image file. Returns number of output images produced."""
        logging.info("Processing %s", path)
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            logging.error("Failed to read %s: %s", path, e)
            return 0

        try:
            outputs = self.maker.process(img_bytes)
        except Exception as e:
            logging.exception("Processing failed for %s", path)
            return 0

        if not outputs:
            logging.info("No faces detected in %s", path)
            return 0

        prefix = path.stem
        saved_paths = self.maker.save_base64_outputs(outputs, prefix=prefix, output_dir=str(self.edited))
        logging.info("Saved %d outputs for %s -> %s", len(saved_paths), path, saved_paths)

        # Move the original file to processed_raw after successful processing
        try:
            dest = self.processed_raw / path.name
            if dest.exists():
                dest = self.processed_raw / f"{path.stem}_{int(time.time())}{path.suffix}"
            shutil.move(str(path), str(dest))
            logging.info("Moved original %s to %s", path, dest)
        except Exception as e:
            logging.error("Failed to move %s to processed folder: %s", path, e)

        return len(saved_paths)

    def process_once(self) -> Tuple[int, int]:
        """Process all images currently in the source folder. Returns (files_scanned, total_outputs)."""
        files = self._list_images()
        logging.info("Found %d candidate files", len(files))
        total_outputs = 0
        for p in files:
            try:
                outs = self.process_file(p)
                total_outputs += outs
            except Exception:
                logging.exception("Unexpected error while processing %s", p)

        return len(files), total_outputs

    def run_forever(self) -> None:
        logging.info("Starting batch processor, polling every %s seconds", self.poll_interval)
        try:
            while True:
                start = time.time()
                files, outputs = self.process_once()
                logging.info("Cycle complete: scanned %d files, produced %d outputs", files, outputs)
                elapsed = time.time() - start
                to_sleep = max(0.0, self.poll_interval - elapsed)
                time.sleep(to_sleep)
        except KeyboardInterrupt:
            logging.info("Interrupted by user, exiting.")
