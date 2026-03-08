Passport Photo Maker

This small utility detects faces in images, produces passport-style images composited over a blue background, and manages files in batch: processed raw images are moved to a processed folder and edited outputs are written to the Edited Photos folder.

What's new / overview
- Code is refactored into two classes: `PassportPhotoMaker` (face detection + image creation) and `BatchProcessor` (watches a folder, processes images, saves outputs, moves processed files aside).
- The script can run continuously (polling every N seconds) or once for testing.

Prerequisites
1. Python 3.8+ (recommended)
2. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Required files:
- `haarcascade_frontalface_default.xml` (OpenCV Haar cascade) must be present in the project folder or you can provide a path to it in code. Download from OpenCV if needed.
- `rembg` is used for background removal. On some platforms it may require extra native dependencies; if rembg is unavailable you can change the background replacement logic in `main.py`.

Folder structure (auto-created inside the input directory):
- `Edited Photos/`
  - `Drop to Print/` (move selected edited photos here manually)
  - `Printed/` (final A4 pages)
- `Processed Raw/` (originals after they've been edited)

Usage (examples)

- Run generator once (process raws -> Edited Photos):

```bash
python app.py generate --input "C:\\path\\to\\camera_photos"
```

- Watch continuously (process every 60s by default):

```bash
python app.py generate --input "C:\\path\\to\\camera_photos" --watch
```

- Build layouts from Drop to Print into Printed (once):

```bash
python app.py print --input "C:\\path\\to\\camera_photos"
```

- Watch Drop to Print continuously:

```bash
python app.py print --input "C:\\path\\to\\camera_photos" --watch --interval 30
```

- Do both sequentially (generate once, then print):

```bash
python app.py run-all --input "C:\\path\\to\\camera_photos"
```

What the script does
1. Scans the input folder for image files (.jpg, .jpeg, .png, .bmp, .tiff).
2. For each file, detects faces and produces one edited passport-style image per face.
3. Saves outputs to `Edited Photos` with filenames `{original_name}_face1.jpg`, `{original_name}_face2.jpg`, etc.; name collisions get a numeric suffix.
4. Moves the processed raw file to `Processed Raw` to avoid reprocessing.
5. When you manually move edited photos into `Edited Photos/Drop to Print`, the batch layout step builds A4 pages and writes them to `Edited Photos/Printed`. Used drop-to-print images are moved to `Edited Photos/Drop to Print/Completed`.

API (programmatic)
You can use the classes directly from Python:

```python
from main import PassportPhotoMaker, BatchProcessor

maker = PassportPhotoMaker()  # configure arguments if you want
processor = BatchProcessor(
    r"<input_dir>",  # contains camera photos + Edited Photos structure
    maker,
)

# Run once:
processor.process_once()

# Or run forever (polling):
processor.run_forever()
```

Notes & troubleshooting
- Ensure `haarcascade_frontalface_default.xml` is available where `main.py` can see it. If you see a FileNotFoundError, download the cascade and place it next to `main.py`.
- `rembg` may need extra OS-level libraries on some systems (FFmpeg, Rust toolchain for building on older setups, etc.). If background removal fails or is slow, consider using a simpler mask-based fallback.
- If an image is being written to the source folder by another process (e.g. automation), ensure that file writes are atomic or add a small delay before processing to avoid partial reads.
