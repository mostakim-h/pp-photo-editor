import uuid
from pathlib import Path

from PIL import Image
from typing import List


class A4PassportLayout:
    def __init__(
        self,
        dpi: int = 300,
        photo_size_inch=(1.835, 1.435),
        columns: int = 4,
        margin_px: int = 20,
        gap_px: int = 60,
        bg_color=(255, 255, 255),
    ):
        self.dpi = dpi
        self.photo_w = int(photo_size_inch[0] * dpi)
        self.photo_h = int(photo_size_inch[1] * dpi)
        self.columns = columns
        self.margin_px = margin_px
        self.gap_px = gap_px
        self.bg_color = bg_color

        self.page_w = int(8.27 * dpi)   # A4 width
        self.page_h = int(11.69 * dpi)  # A4 height

    def prepare_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img = img.rotate(-90, expand=True)
        img = img.resize((self.photo_w, self.photo_h), Image.Resampling.LANCZOS)

        pad = int(0.1 * self.dpi)
        padded = Image.new(
            "RGB",
            (self.photo_w + 2 * pad, self.photo_h + 2 * pad),
            self.bg_color,
        )
        padded.paste(img, (pad, pad))

        border = 1
        bordered = Image.new(
            "RGB",
            (padded.width + 2 * border, padded.height + 2 * border),
            (0, 0, 0),
        )
        bordered.paste(padded, (border, border))

        return bordered

    def create_pages(self, image_paths: List[str], output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- PREPARE ALL IMAGES ONCE ----
        images = [self.prepare_image(p) for p in image_paths]

        MAX_PER_PAGE = 7
        page_index = 1

        for start in range(0, len(images), MAX_PER_PAGE):
            chunk = images[start:start + MAX_PER_PAGE]
            page = Image.new("RGB", (self.page_w, self.page_h), self.bg_color)

            x0 = self.margin_px
            y0 = self.margin_px
            col_step = self.photo_w + self.gap_px
            row_step = self.photo_h + self.gap_px

            row = 0
            i = 0

            while i < len(chunk):
                y = y0 + row * row_step

                # ---- TWO IMAGES ----
                if i + 1 < len(chunk):
                    img_left = chunk[i]
                    img_right = chunk[i + 1]

                    for r in range(2):
                        y_row = y + r * row_step
                        page.paste(img_left, (x0 + 0 * col_step, y_row))
                        page.paste(img_left, (x0 + 1 * col_step, y_row))
                        page.paste(img_right, (x0 + 2 * col_step, y_row))
                        page.paste(img_right, (x0 + 3 * col_step, y_row))

                    i += 2
                    row += 2

                # ---- LAST SINGLE IMAGE (7th) ----
                else:
                    img = chunk[i]
                    y_row = y

                    for c in range(4):
                        page.paste(img, (x0 + c * col_step, y_row))

                    i += 1
                    row += 1

            output_path = output_dir / "printed" / f"page_{uuid.uuid4().hex}.jpg"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            page.save(output_path, dpi=(self.dpi, self.dpi))
            page_index += 1

        return page_index - 1