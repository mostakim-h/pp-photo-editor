Passport Photo Maker

This small utility detects faces in images, produces passport-style images composited over a blue background, and manages files in batch: processed raw images are moved to a trash folder and edited outputs are written to an output folder.

What's new / overview
- Code is refactored into two classes: `PassportPhotoMaker` (face detection + image creation) and `BatchProcessor` (watches a folder, processes images, saves outputs, moves processed files to trash).
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

Default folders (these are the defaults used by the CLI):
- Source (raw) images: <images_path>
- Edited outputs: <output_path>
- Trash (processed raws): <trash_path>

Usage (examples)

- Run one processing cycle (useful for testing):

```bash
python main.py --once
```

- Run continuously (poll every 60 seconds by default):

```bash
python main.py
```

- Override defaults and interval:

```bash
python main.py --src "C:\\path\\to\\raw_images" --out "C:\\path\\to\\edited_images" --trash "C:\\path\\to\\trash" --interval 30
```

What the script does
1. Scans the source folder for image files (.jpg, .jpeg, .png, .bmp, .tiff).
2. For each file, detects faces and produces one edited passport-style image per face.
3. Saves outputs to the output folder with filenames in the pattern `{original_name}_face1.jpg`, `{original_name}_face2.jpg`, etc. If a name collision occurs, a numeric suffix is appended.
4. Moves the processed raw file to the trash folder. If a file with the same name exists in trash, a timestamp is appended.

API (programmatic)
You can use the classes directly from Python:

```python
from main import PassportPhotoMaker, BatchProcessor

maker = PassportPhotoMaker()  # configure arguments if you want
# Example with explicit paths (replace with your actual folders):
processor = BatchProcessor(
    r"<images_path>",
    r"<output_path>",
    r"<trash_path>",
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
