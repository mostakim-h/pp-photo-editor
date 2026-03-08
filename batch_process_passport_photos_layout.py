from pathlib import Path

from a4_passport_photo_layout import A4PassportLayout

last_img_count = 0
is_last_page = False

def batch_process_passport_photos_layout(drop_to_print_dir: str, printed_dir: str):

    global last_img_count, is_last_page

    drop_path = Path(drop_to_print_dir)
    printed_path = Path(printed_dir)

    drop_path.mkdir(parents=True, exist_ok=True)
    printed_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        str(p) for p in drop_path.glob("*.jpg")
    )

    if not images:
        last_img_count = 0
        is_last_page = False
        return 0

    if last_img_count == len(images):
        is_last_page = True

    last_img_count = len(images)

    if is_last_page or len(images) > 6:

        if len(images) > 7:
            images = images[:7]

        layout = A4PassportLayout()
        total_pages = layout.create_pages(
            image_paths=images,
            printed_dir=str(printed_path),
            completed_dir=str(drop_path / "Completed")
        )

        # Reset counters after a page is produced
        last_img_count = 0
        is_last_page = False

        return total_pages

    return 0
