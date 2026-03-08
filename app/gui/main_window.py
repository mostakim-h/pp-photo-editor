"""
main_window.py
--------------
The primary application window for the Passport Photo Processing System.

Layout (top → bottom):
  1. App title bar
  2. Input folder selector
  3. Job control panel  (checkboxes + Start/Stop buttons)
  4. Status / log console
  5. Bottom action bar  (Configuration | Open Folder | Exit)

Both jobs run in daemon threads and watch their folders continuously
until the user clicks "Stop".  The UI never freezes because all work
happens in background threads; a queue.Queue is drained every 100 ms
to display log messages safely from the main thread.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import tkinter as tk

from app.config.settings import Settings
from app.core.folder_manager import FolderManager
from app.core.generator import GeneratorJob
from app.core.layout_processor import LayoutProcessorJob
from app.gui.widgets import (
    ActionButton,
    Divider,
    FolderSelector,
    LogConsole,
    SectionLabel,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_MS = 100   # ms between log-queue drains

# Button colour constants
_CLR_START  = ("#1f6aa5", "#1f6aa5")   # blue  — job is idle
_CLR_STOP   = ("#b22222", "#c0392b")   # red   — job is running / click to stop
_CLR_WAIT   = ("gray40",  "gray35")    # grey  — stopping in progress


class MainWindow(ctk.CTk):
    """Root application window."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()

        self._settings = settings
        self._log_queue: queue.Queue = queue.Queue()

        # Active job objects (needed to call .cancel())
        self._gen_job: Optional[GeneratorJob] = None
        self._layout_job: Optional[LayoutProcessorJob] = None

        # Active worker threads
        self._gen_thread: Optional[threading.Thread] = None
        self._layout_thread: Optional[threading.Thread] = None

        # State flags
        self._gen_running = False
        self._layout_running = False

        self._setup_window()
        self._build_ui()
        self._restore_state()
        self._poll_log_queue()

    # ==================================================================
    # Window setup
    # ==================================================================

    def _setup_window(self) -> None:
        self.title("Passport Photo Processing System")
        self.geometry("760x720")
        self.minsize(680, 600)

        mode = self._settings.get("ui.appearance_mode", "dark")
        ctk.set_appearance_mode(mode)
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        # ----------------------------------------------------------------
        # 1. Title bar
        # ----------------------------------------------------------------
        title_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray85", "gray20"))
        title_frame.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            title_frame,
            text="📷  Passport Photo Processing System",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(side="left", padx=20, pady=14)

        # ----------------------------------------------------------------
        # 2. Input folder selector
        # ----------------------------------------------------------------
        folder_frame = ctk.CTkFrame(self, fg_color="transparent")
        folder_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(14, 4))
        folder_frame.grid_columnconfigure(0, weight=1)

        self._folder_selector = FolderSelector(
            folder_frame,
            label="Input Folder:",
            on_change=self._on_folder_changed,
        )
        self._folder_selector.grid(row=0, column=0, sticky="ew")

        self._folder_status_label = ctk.CTkLabel(
            folder_frame, text="", font=ctk.CTkFont(size=11), anchor="w"
        )
        self._folder_status_label.grid(row=1, column=0, sticky="w", padx=2, pady=(2, 0))

        Divider(self).grid(row=2, column=0, sticky="ew", padx=20, pady=4)

        # ----------------------------------------------------------------
        # 3. Job control panel
        # ----------------------------------------------------------------
        jobs_frame = ctk.CTkFrame(self, corner_radius=10)
        jobs_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=6)
        jobs_frame.grid_columnconfigure(1, weight=1)

        SectionLabel(jobs_frame, text="⚙  Jobs").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 6)
        )

        # --- Checkboxes ---------------------------------------------------
        self._gen_var    = tk.BooleanVar(value=True)
        self._layout_var = tk.BooleanVar(value=True)

        ctk.CTkCheckBox(
            jobs_frame,
            text="Passport Photo Generator",
            variable=self._gen_var,
            onvalue=True, offvalue=False,
            font=ctk.CTkFont(size=13),
        ).grid(row=1, column=0, sticky="w", padx=16, pady=4)

        ctk.CTkCheckBox(
            jobs_frame,
            text="Layout Processor",
            variable=self._layout_var,
            onvalue=True, offvalue=False,
            font=ctk.CTkFont(size=13),
        ).grid(row=2, column=0, sticky="w", padx=16, pady=4)

        # --- Status labels ------------------------------------------------
        self._gen_status = ctk.CTkLabel(
            jobs_frame, text="● Idle", font=ctk.CTkFont(size=11), anchor="w",
            text_color="gray60"
        )
        self._gen_status.grid(row=1, column=1, sticky="w", padx=8)

        self._layout_status = ctk.CTkLabel(
            jobs_frame, text="● Idle", font=ctk.CTkFont(size=11), anchor="w",
            text_color="gray60"
        )
        self._layout_status.grid(row=2, column=1, sticky="w", padx=8)

        # --- Buttons (right side) -----------------------------------------
        btn_col = ctk.CTkFrame(jobs_frame, fg_color="transparent")
        btn_col.grid(row=1, column=2, rowspan=3, padx=16, pady=8, sticky="e")

        # Generator start/stop toggle
        self._btn_gen = ctk.CTkButton(
            btn_col,
            text="▶  Start Generator",
            width=200, height=38, corner_radius=8,
            font=ctk.CTkFont(size=13),
            fg_color=_CLR_START,
            command=self._toggle_generator,
        )
        self._btn_gen.pack(pady=5)

        # Layout start/stop toggle
        self._btn_layout = ctk.CTkButton(
            btn_col,
            text="▶  Start Layout",
            width=200, height=38, corner_radius=8,
            font=ctk.CTkFont(size=13),
            fg_color=_CLR_START,
            command=self._toggle_layout,
        )
        self._btn_layout.pack(pady=5)

        # Start/Stop all selected jobs
        self._btn_both = ctk.CTkButton(
            btn_col,
            text="▶  Start Selected Jobs",
            width=200, height=38, corner_radius=8,
            font=ctk.CTkFont(size=13),
            fg_color=("gray35", "gray25"),
            command=self._toggle_selected,
        )
        self._btn_both.pack(pady=5)

        Divider(self).grid(row=4, column=0, sticky="ew", padx=20, pady=4)

        # ----------------------------------------------------------------
        # 4. Log console
        # ----------------------------------------------------------------
        log_frame = ctk.CTkFrame(self, fg_color="transparent")
        log_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=(0, 6))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(5, weight=1)

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew")

        SectionLabel(log_header, text="📋  Status / Logs").pack(side="left")

        ctk.CTkButton(
            log_header,
            text="Clear",
            width=70, height=26, corner_radius=6,
            fg_color="gray40", hover_color="gray50",
            font=ctk.CTkFont(size=11),
            command=lambda: self._console.clear(),
        ).pack(side="right")

        self._console = LogConsole(log_frame)
        self._console.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        # ----------------------------------------------------------------
        # 5. Bottom action bar
        # ----------------------------------------------------------------
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=6, column=0, sticky="ew", padx=20, pady=(4, 14))

        ctk.CTkButton(
            bottom, text="⚙  Configuration",
            width=160, height=36, corner_radius=8,
            command=self._open_config,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            bottom, text="📂  Open Input Folder",
            width=180, height=36, corner_radius=8,
            fg_color=("gray35", "gray25"), hover_color=("gray45", "gray35"),
            command=self._open_input_folder,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            bottom, text="Exit",
            width=100, height=36, corner_radius=8,
            fg_color="#c0392b", hover_color="#e74c3c",
            command=self._on_exit,
        ).pack(side="right")

    # ==================================================================
    # State restore
    # ==================================================================

    def _restore_state(self) -> None:
        saved_dir = self._settings.get("input_dir", "")
        if saved_dir:
            if self._is_managed_subfolder(saved_dir):
                # Saved path is a managed sub-folder — clear it so the user
                # picks the correct parent directory.
                self._settings.set("input_dir", "")
                self._settings.save()
                self._log(
                    "[WARN] Saved input folder pointed to a managed sub-folder "
                    f"('{Path(saved_dir).name}'). Please re-select the correct input directory."
                )
            else:
                self._folder_selector.set(saved_dir)
                self._update_folder_status(saved_dir)
        self._log("[INFO] Application ready. Select an input folder to begin.")

    # ==================================================================
    # Folder handling
    # ==================================================================

    # Names that are managed sub-folders — user must NOT select these as input
    _MANAGED_SUBFOLDER_NAMES = {
        "edited photos", "drop to print", "printed", "processed raw", "completed"
    }

    def _is_managed_subfolder(self, path: str) -> bool:
        """Return True if *path* is one of the auto-created sub-folders."""
        return Path(path).name.lower() in self._MANAGED_SUBFOLDER_NAMES

    def _on_folder_changed(self, path: str) -> None:
        if not path:
            return

        if self._is_managed_subfolder(path):
            self._folder_status_label.configure(
                text=f"⚠ '{Path(path).name}' is a managed sub-folder. "
                     "Please select the parent input directory.",
                text_color=("#e74c3c", "#ff6b6b"),
            )
            self._log(
                f"[WARN] '{Path(path).name}' is a managed sub-folder — "
                "select the parent input directory instead."
            )
            return

        self._settings.set("input_dir", path)
        self._settings.save()
        FolderManager(path).ensure_structure()
        self._update_folder_status(path)
        self._log(f"[INFO] Input folder set: {path}")
        self._log("[INFO] Folder structure verified.")

    def _update_folder_status(self, path: str) -> None:
        if not path:
            self._folder_status_label.configure(text="")
            return
        p = Path(path)
        if not p.exists():
            self._folder_status_label.configure(
                text="⚠ Folder does not exist.", text_color=("#e74c3c", "#ff6b6b")
            )
            return
        if FolderManager(p).validate():
            self._folder_status_label.configure(
                text="✔ Folder structure OK", text_color=("#27ae60", "#2ecc71")
            )
        else:
            self._folder_status_label.configure(
                text="⚠ Sub-folders will be created on first start.",
                text_color=("#e67e22", "#f39c12"),
            )

    # ==================================================================
    # Toggle helpers — shared logic for start / stop
    # ==================================================================

    def _toggle_generator(self) -> None:
        if self._gen_running:
            self._stop_generator()
        else:
            self._start_generator()

    def _toggle_layout(self) -> None:
        if self._layout_running:
            self._stop_layout()
        else:
            self._start_layout()

    def _toggle_selected(self) -> None:
        """If any selected job is running stop it; otherwise start all selected."""
        gen_sel    = self._gen_var.get()
        layout_sel = self._layout_var.get()

        if not gen_sel and not layout_sel:
            self._log("[WARN] No jobs selected. Tick at least one checkbox.")
            return

        # Determine if we are in "start" or "stop" mode based on current state
        any_running = (gen_sel and self._gen_running) or (layout_sel and self._layout_running)

        if any_running:
            if gen_sel and self._gen_running:
                self._stop_generator()
            if layout_sel and self._layout_running:
                self._stop_layout()
        else:
            if gen_sel:
                self._start_generator()
            if layout_sel:
                self._start_layout()

    # ==================================================================
    # Job starters
    # ==================================================================

    def _start_generator(self) -> None:
        input_dir = self._folder_selector.get()
        if not self._validate_input_dir(input_dir):
            return
        if self._gen_running:
            return

        poll = int(self._settings.get("generator.poll_interval", 10))
        self._gen_running = True
        self._set_gen_status("● Watching …", "#f39c12")
        self._btn_gen.configure(text="⏹  Stop Generator", fg_color=_CLR_STOP)
        self._sync_both_button()

        def _on_stopped() -> None:
            self._gen_running = False
            self.after(0, self._on_gen_stopped)

        self._gen_job = GeneratorJob(
            base_dir=input_dir,
            log_queue=self._log_queue,
            poll_interval=poll,
            on_stopped=_on_stopped,
        )
        self._gen_thread = threading.Thread(target=self._gen_job.run, daemon=True)
        self._gen_thread.start()
        self._log("[INFO] Generator started — watching for new photos.")

    def _start_layout(self) -> None:
        input_dir = self._folder_selector.get()
        if not self._validate_input_dir(input_dir):
            return
        if self._layout_running:
            return

        poll        = int(self._settings.get("layout.poll_interval", 10))
        auto_print  = self._settings.get("printer.auto_print", False)
        printer     = self._settings.get("printer.name", "") or None
        copies      = int(self._settings.get("printer.copies", 1))

        self._layout_running = True
        self._set_layout_status("● Watching …", "#f39c12")
        self._btn_layout.configure(text="⏹  Stop Layout", fg_color=_CLR_STOP)
        self._sync_both_button()

        def _on_stopped() -> None:
            self._layout_running = False
            self.after(0, self._on_layout_stopped)

        self._layout_job = LayoutProcessorJob(
            base_dir=input_dir,
            log_queue=self._log_queue,
            poll_interval=poll,
            auto_print=auto_print,
            printer_name=printer,
            copies=copies,
            on_stopped=_on_stopped,
        )
        self._layout_thread = threading.Thread(
            target=self._layout_job.run, daemon=True
        )
        self._layout_thread.start()
        self._log("[INFO] Layout Processor started — watching Drop to Print.")

    # ==================================================================
    # Job stoppers
    # ==================================================================

    def _stop_generator(self) -> None:
        if self._gen_job and self._gen_running:
            self._gen_job.cancel()
            self._btn_gen.configure(text="⏳ Stopping …", fg_color=_CLR_WAIT, state="disabled")
            self._set_gen_status("● Stopping …", "gray60")
            self._log("[INFO] Generator stop requested — finishing current cycle …")

    def _stop_layout(self) -> None:
        if self._layout_job and self._layout_running:
            self._layout_job.cancel()
            self._btn_layout.configure(text="⏳ Stopping …", fg_color=_CLR_WAIT, state="disabled")
            self._set_layout_status("● Stopping …", "gray60")
            self._log("[INFO] Layout Processor stop requested — finishing current cycle …")

    # ==================================================================
    # on_stopped callbacks  (always called via self.after → UI thread)
    # ==================================================================

    def _on_gen_stopped(self) -> None:
        self._gen_running = False
        self._gen_job = None
        self._btn_gen.configure(
            text="▶  Start Generator", fg_color=_CLR_START, state="normal"
        )
        self._set_gen_status("● Idle", "gray60")
        self._sync_both_button()
        self._log("[INFO] Generator stopped.")

    def _on_layout_stopped(self) -> None:
        self._layout_running = False
        self._layout_job = None
        self._btn_layout.configure(
            text="▶  Start Layout", fg_color=_CLR_START, state="normal"
        )
        self._set_layout_status("● Idle", "gray60")
        self._sync_both_button()
        self._log("[INFO] Layout Processor stopped.")

    # ==================================================================
    # "Run Selected Jobs" button sync
    # ==================================================================

    def _sync_both_button(self) -> None:
        """Keep the 'Start/Stop Selected Jobs' button label in sync."""
        gen_sel    = self._gen_var.get()
        layout_sel = self._layout_var.get()
        any_running = (gen_sel and self._gen_running) or (layout_sel and self._layout_running)

        if any_running:
            self._btn_both.configure(
                text="⏹  Stop Selected Jobs", fg_color=_CLR_STOP
            )
        else:
            self._btn_both.configure(
                text="▶  Start Selected Jobs", fg_color=("gray35", "gray25")
            )

    # ==================================================================
    # Status label helpers
    # ==================================================================

    def _set_gen_status(self, text: str, color: str) -> None:
        self._gen_status.configure(text=text, text_color=color)

    def _set_layout_status(self, text: str, color: str) -> None:
        self._layout_status.configure(text=text, text_color=color)

    # ==================================================================
    # Log queue polling
    # ==================================================================

    def _poll_log_queue(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._console.append(msg)
        except queue.Empty:
            pass
        finally:
            self.after(_POLL_INTERVAL_MS, self._poll_log_queue)

    def _log(self, message: str) -> None:
        self._console.append(message)

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_input_dir(self, path: str) -> bool:
        if not path:
            self._log("[ERROR] Please select an input folder first.")
            return False
        if self._is_managed_subfolder(path):
            self._log(
                f"[ERROR] '{Path(path).name}' is a managed sub-folder. "
                "Select the parent input directory."
            )
            return False
        if not Path(path).exists():
            self._log(f"[ERROR] Input folder does not exist: {path}")
            return False
        return True

    # ==================================================================
    # Bottom bar actions
    # ==================================================================

    def _open_config(self) -> None:
        from app.gui.config_window import ConfigWindow
        ConfigWindow(self, self._settings)

    def _open_input_folder(self) -> None:
        import subprocess
        path = self._folder_selector.get()
        if not path or not Path(path).exists():
            self._log("[WARN] No valid input folder to open.")
            return
        subprocess.Popen(["explorer", path])

    def _on_exit(self) -> None:
        # Stop any running jobs gracefully before quitting
        if self._gen_job:
            self._gen_job.cancel()
        if self._layout_job:
            self._layout_job.cancel()
        self._settings.save()
        self.destroy()

