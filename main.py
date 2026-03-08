import argparse
import logging

from a4_passport_photo_layout import A4PassportLayout
from batch_photo_processor import BatchProcessor
from passport_photo_maker import PassportPhotoMaker

# ------------------------------
# CLI / Entrypoint
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch passport photo maker")
    parser.add_argument("--input", default=".", help="Input folder containing camera photos and Edited Photos structure")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run one processing cycle and exit (useful for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        maker = PassportPhotoMaker()
    except FileNotFoundError as e:
        logging.error("Face cascade not found: %s", e)
        raise

    processor = BatchProcessor(args.input, maker, poll_interval=args.interval)

    if args.once:
        files, outputs = processor.process_once()
        logging.info("Done (once): scanned %d files, produced %d outputs", files, outputs)

    else:
        processor.run_forever()
