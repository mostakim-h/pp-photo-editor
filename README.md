# Passport Photo Processing System

A desktop GUI application (and CLI) for processing passport photos — detecting faces, replacing backgrounds, and generating print-ready A4 layouts.

---

## ✨ Features

- **GUI desktop app** built with [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- **Passport Photo Generator** — detects faces, applies blue background, saves edited photos
- **Batch Layout Processor** — tiles photos into A4 print layouts (7 photos / page)
- **Auto-print** support via Windows printer API (`pywin32`)
- **Settings persistence** — folder paths, printer config, and UI theme saved to `app/config/settings.json`
- Both jobs run **in background threads** — the UI never freezes
- **Log console** with colour-coded `[INFO]` / `[WARN]` / `[ERROR]` output

---

## 🚀 Quickstart — GUI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the GUI
python run_gui.py
```

---

## 📁 Project Structure

```
pp-photo-editor/
│
├── run_gui.py                  ← GUI launcher (project root)
│
├── app/                        ← GUI application package
│   ├── main.py                 ← Entry point
│   ├── gui/
│   │   ├── main_window.py      ← Main application window
│   │   ├── config_window.py    ← Settings / Configuration dialog
│   │   └── widgets.py          ← Reusable widget components
│   ├── core/
│   │   ├── folder_manager.py   ← Directory creation & validation
│   │   ├── generator.py        ← Photo generator pipeline (threaded)
│   │   └── layout_processor.py ← Layout processor pipeline (threaded)
│   ├── printing/
│   │   └── printer_manager.py  ← Windows printer detection & print jobs
│   └── config/
│       ├── settings.py         ← Load/save JSON settings
│       └── settings.json       ← Persisted user settings
│
├── passport_photo_maker.py     ← Core: face detection + blue-BG compositing
├── batch_photo_processor.py    ← Core: batch folder processor
├── batch_process_passport_photos_layout.py  ← Layout generation
├── a4_passport_photo_layout.py ← A4 tiling engine
├── face_detector.py            ← OpenCV / DNN face detector
└── app.py                      ← Legacy CLI entry point
```

---

## 📂 Working Directory Structure

The app auto-creates this layout inside your chosen **Input Folder**:

```
Input Folder/
├── Edited Photos/
│   ├── Drop to Print/      ← Move selected edited photos here
│   └── Printed/            ← Final A4 layout pages saved here
└── Processed Raw/          ← Original camera files moved here after processing
```

---

## 🖥 GUI Walkthrough

1. **Select Input Folder** — browse to your camera dump directory
2. **Run Generator** — processes all raw photos → saves edited versions to `Edited Photos/`
3. **Manual step** — move your chosen edited photos into `Edited Photos/Drop to Print/`
4. **Run Layout** — generates A4 print pages into `Edited Photos/Printed/`
5. **Auto-print** (optional) — configure in ⚙ Configuration → tick *Auto-print after layout generation*

---

## ⚙ Configuration

Open **⚙ Configuration** from the main window to set:

| Setting | Description |
|---|---|
| Printer | Detected from installed Windows printers |
| Paper Size | `4x6` / `A4` / `Custom` |
| Copies | Number of print copies |
| Auto-print | Send layouts directly to printer after generation |
| Theme | `dark` / `light` / `system` |

---

## 🔧 CLI Usage (Legacy)

The original CLI is still available via `app.py`:

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
