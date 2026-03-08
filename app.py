import argparse
import logging
from pathlib import Path
import time

from batch_photo_processor import BatchProcessor
from batch_process_passport_photos_layout import batch_process_passport_photos_layout
from passport_photo_maker import PassportPhotoMaker


def ensure_structure(base_dir: Path) -> dict:
    """Ensure the required folder structure exists and return useful paths."""
    base_dir.mkdir(parents=True, exist_ok=True)
    edited = base_dir / "Edited Photos"
    drop_to_print = edited / "Drop to Print"
    printed = edited / "Printed"
    processed_raw = base_dir / "Processed Raw"

    for d in (edited, drop_to_print, printed, processed_raw):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "edited": edited,
        "drop_to_print": drop_to_print,
        "printed": printed,
        "processed_raw": processed_raw,
    }


def cmd_generate(args) -> None:
    paths = ensure_structure(Path(args.input))
    maker = PassportPhotoMaker()
    processor = BatchProcessor(str(paths["base"]), maker, poll_interval=args.interval)

    if args.watch:
        processor.run_forever()
    else:
        files, outputs = processor.process_once()
        logging.info("Generate complete: scanned %d files, produced %d outputs", files, outputs)


def cmd_print(args) -> None:
    paths = ensure_structure(Path(args.input))

    def run_once() -> int:
        pages = batch_process_passport_photos_layout(
            drop_to_print_dir=str(paths["drop_to_print"]),
            printed_dir=str(paths["printed"]),
        )
        logging.info("Print complete: generated %d pages from drop-to-print images", pages)
        return pages

    if args.watch:
        logging.info("Watching Drop to Print every %s seconds", args.interval)
        try:
            while True:
                start = time.time()
                run_once()
                elapsed = time.time() - start
                time.sleep(max(0.0, args.interval - elapsed))
        except KeyboardInterrupt:
            logging.info("Print watch interrupted by user")
    else:
        run_once()


def cmd_run_all(args) -> None:
    # Always run generator once, then layout
    gen_args = argparse.Namespace(**vars(args))
    gen_args.watch = False
    cmd_generate(gen_args)
    cmd_print(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Passport photo processing system")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)")

    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Run passport photo generator")
    gen.add_argument("--input", required=True, help="Input folder containing camera photos")
    gen.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (when --watch)")
    gen.add_argument("--watch", action="store_true", help="Watch input folder continuously")
    gen.set_defaults(func=cmd_generate)

    # print
    pr = sub.add_parser("print", help="Run layout on Drop to Print -> Printed")
    pr.add_argument("--input", required=True, help="Input folder containing Edited Photos/Drop to Print")
    pr.add_argument("--interval", type=int, default=60, help="Poll interval in seconds when watching")
    pr.add_argument("--watch", action="store_true", help="Watch Drop to Print continuously")
    pr.set_defaults(func=cmd_print)

    # run-all
    both = sub.add_parser("run-all", help="Run generator then layout")
    both.add_argument("--input", required=True, help="Input folder containing camera photos")
    both.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (when --watch)")
    both.add_argument("--watch", action="store_true", help="Watch input folder continuously during generation")
    both.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")

    args.func(args)


if __name__ == "__main__":
    main()

