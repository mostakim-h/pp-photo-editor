from pathlib import Path

from a4_passport_photo_layout import A4PassportLayout

last_img_count = 0
is_last_page = False

def batch_process_passport_photos_layout(output_dir: str):

    global last_img_count, is_last_page

    output_dir = Path(output_dir)
    images = sorted(
        str(p) for p in output_dir.glob("*.jpg")
    )

    if not images:
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
            output_dir=str(output_dir)
        )

        return total_pages

    return 0
