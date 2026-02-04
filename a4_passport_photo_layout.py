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

    def create_page(self, image_paths: List[str], output_path: str):
        page = Image.new("RGB", (self.page_w, self.page_h), self.bg_color)

        x = self.margin_px
        y = self.margin_px
        col = 0

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.rotate(-90, expand=True)
            img = img.resize((self.photo_w, self.photo_h), Image.Resampling.LANCZOS)

            pad = int(0.1 * self.dpi)
            padded_img = Image.new("RGB", (self.photo_w + 2 * pad, self.photo_h + 2 * pad), self.bg_color)
            padded_img.paste(img, (pad, pad))

            # now add a border around the padded image
            border = 1  # 2 pixels border
            img = Image.new("RGB", (padded_img.width + 2 * border, padded_img.height + 2 * border), (0, 0, 0))
            img.paste(padded_img, (border, border))
            # repeat twice (matches demo)
            for _ in range(4):

                # add padding around the image 0.15 inch (0.15 * dpi pixels)
                page.paste(img, (x, y))

                col += 1
                if col >= self.columns:
                    col = 0
                    x = self.margin_px
                    y += self.photo_h + self.gap_px
                else:
                    x += self.photo_w + self.gap_px

        page.save(output_path, dpi=(self.dpi, self.dpi))
        return output_path
