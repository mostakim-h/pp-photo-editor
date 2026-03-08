"""
config_window.py
----------------
Settings / Configuration pop-up window.

Covers:
  • Printer selection (detected from the system)
  • Paper size
  • Auto-print toggle
  • Number of copies
  • UI appearance mode
"""

import tkinter as tk
import logging

import customtkinter as ctk

from app.config.settings import Settings
from app.printing.printer_manager import PrinterManager

logger = logging.getLogger(__name__)

PAPER_SIZES = ["4x6", "A4", "Custom"]
APPEARANCE_MODES = ["dark", "light", "system"]


class ConfigWindow(ctk.CTkToplevel):
    """
    Modal configuration window.

    Parameters
    ----------
    master:
        Parent widget (the main window).
    settings:
        Shared Settings instance; changes are written back on Save.
    """

    def __init__(self, master, settings: Settings) -> None:
        super().__init__(master)
        self._settings = settings
        self._printer_manager = PrinterManager()

        self.title("Configuration")
        self.geometry("520x600")
        self.resizable(False, False)
        self.grab_set()   # make modal
        self.focus_set()

        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Outer scroll-frame so the window stays compact
        outer = ctk.CTkScrollableFrame(self, corner_radius=0, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        outer.grid_columnconfigure(0, weight=1)

        pad = {"padx": 20, "pady": 6}

        # ---- Printer -------------------------------------------------------
        self._section(outer, "🖨  Printer Settings", row=0)

        self._printer_var = tk.StringVar()
        self._printer_label = ctk.CTkLabel(outer, text="Printer:", anchor="w")
        self._printer_label.grid(row=1, column=0, sticky="w", **pad)

        self._printer_combo = ctk.CTkComboBox(
            outer,
            variable=self._printer_var,
            values=self._get_printer_list(),
            width=460,
            height=34,
            corner_radius=8,
        )
        self._printer_combo.grid(row=2, column=0, sticky="ew", **pad)

        # Refresh + Open Printer Settings buttons side by side
        printer_btn_row = ctk.CTkFrame(outer, fg_color="transparent")
        printer_btn_row.grid(row=3, column=0, sticky="w", **pad)

        ctk.CTkButton(
            printer_btn_row,
            text="↻  Refresh",
            width=120,
            height=32,
            corner_radius=8,
            fg_color=("gray35", "gray25"),
            hover_color=("gray45", "gray35"),
            command=self._refresh_printers,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            printer_btn_row,
            text="🖨  Printer Settings",
            width=180,
            height=32,
            corner_radius=8,
            command=self._open_printer_settings,
        ).pack(side="left")

        # ---- Paper size ----------------------------------------------------
        self._paper_var = tk.StringVar()
        ctk.CTkLabel(outer, text="Paper Size:", anchor="w").grid(
            row=4, column=0, sticky="w", **pad
        )
        self._paper_combo = ctk.CTkComboBox(
            outer,
            variable=self._paper_var,
            values=PAPER_SIZES,
            width=200,
            height=34,
            corner_radius=8,
        )
        self._paper_combo.grid(row=5, column=0, sticky="w", **pad)

        # ---- Copies --------------------------------------------------------
        ctk.CTkLabel(outer, text="Copies:", anchor="w").grid(
            row=6, column=0, sticky="w", **pad
        )
        self._copies_var = tk.StringVar(value="1")
        self._copies_entry = ctk.CTkEntry(
            outer,
            textvariable=self._copies_var,
            width=80,
            height=34,
            corner_radius=8,
        )
        self._copies_entry.grid(row=7, column=0, sticky="w", **pad)

        # ---- Auto-print toggle ---------------------------------------------
        self._auto_print_var = tk.BooleanVar()
        ctk.CTkCheckBox(
            outer,
            text="Auto-print after layout generation",
            variable=self._auto_print_var,
            onvalue=True,
            offvalue=False,
        ).grid(row=8, column=0, sticky="w", **pad)

        self._divider(outer, row=9)

        # ---- Watch intervals -----------------------------------------------
        self._section(outer, "⏱  Watch Intervals", row=10)

        intervals_frame = ctk.CTkFrame(outer, fg_color="transparent")
        intervals_frame.grid(row=11, column=0, sticky="w", **pad)

        ctk.CTkLabel(intervals_frame, text="Generator poll (seconds):", anchor="w", width=220).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        self._gen_poll_var = tk.StringVar(value="10")
        ctk.CTkEntry(intervals_frame, textvariable=self._gen_poll_var, width=70, height=32, corner_radius=8).grid(
            row=0, column=1, sticky="w"
        )

        ctk.CTkLabel(intervals_frame, text="Layout poll (seconds):", anchor="w", width=220).grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=(6, 0)
        )
        self._layout_poll_var = tk.StringVar(value="10")
        ctk.CTkEntry(intervals_frame, textvariable=self._layout_poll_var, width=70, height=32, corner_radius=8).grid(
            row=1, column=1, sticky="w", pady=(6, 0)
        )

        self._divider(outer, row=12)

        # ---- Appearance ----------------------------------------------------
        self._section(outer, "🎨  Appearance", row=13)

        self._appearance_var = tk.StringVar()
        ctk.CTkLabel(outer, text="Theme:", anchor="w").grid(
            row=14, column=0, sticky="w", **pad
        )
        self._appearance_combo = ctk.CTkComboBox(
            outer,
            variable=self._appearance_var,
            values=APPEARANCE_MODES,
            width=200,
            height=34,
            corner_radius=8,
            command=self._on_appearance_change,
        )
        self._appearance_combo.grid(row=15, column=0, sticky="w", **pad)

        self._divider(outer, row=16)

        # ---- Buttons -------------------------------------------------------
        btn_row = ctk.CTkFrame(outer, fg_color="transparent")
        btn_row.grid(row=17, column=0, sticky="e", **pad)

        ctk.CTkButton(
            btn_row,
            text="Cancel",
            width=100,
            height=36,
            corner_radius=8,
            fg_color="gray40",
            hover_color="gray50",
            command=self.destroy,
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_row,
            text="Save",
            width=100,
            height=36,
            corner_radius=8,
            command=self._save,
        ).pack(side="left")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _section(self, parent, text: str, row: int) -> None:
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=20, pady=(14, 4))

    def _divider(self, parent, row: int) -> None:
        ctk.CTkFrame(parent, height=1, fg_color=("gray70", "gray30")).grid(
            row=row, column=0, sticky="ew", padx=20, pady=6
        )

    def _get_printer_list(self) -> list:
        printers = self._printer_manager.list_printers()
        return printers if printers else ["(No printers found)"]

    def _refresh_printers(self) -> None:
        printers = self._get_printer_list()
        self._printer_combo.configure(values=printers)
        if printers:
            self._printer_var.set(printers[0])
        logger.info("Printer list refreshed: %s", printers)

    def _open_printer_settings(self) -> None:
        """Open the printer driver preferences dialog for the selected printer."""
        selected = self._printer_var.get().strip()
        if not selected or selected == "(No printers found)":
            from tkinter import messagebox
            messagebox.showwarning(
                "No Printer Selected",
                "Please select a printer first.",
                parent=self,
            )
            return
        try:
            # Pass this window's native hwnd so the dialog has a proper parent
            hwnd = self.winfo_id()
            self._printer_manager.open_printer_settings(selected, parent_hwnd=hwnd)
            logger.info("Printer settings opened for: %s", selected)
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror(
                "Printer Settings Error",
                f"Could not open printer settings:\n{exc}",
                parent=self,
            )
            logger.error("Failed to open printer settings: %s", exc)

    def _on_appearance_change(self, value: str) -> None:
        ctk.set_appearance_mode(value)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load_values(self) -> None:
        """Populate controls from the Settings object."""
        saved_printer = self._settings.get("printer.name", "")
        printers = self._get_printer_list()
        if saved_printer and saved_printer in printers:
            self._printer_var.set(saved_printer)
        elif printers:
            self._printer_var.set(printers[0])

        paper = self._settings.get("printer.paper_size", "4x6")
        self._paper_var.set(paper if paper in PAPER_SIZES else "4x6")

        copies = self._settings.get("printer.copies", 1)
        self._copies_var.set(str(copies))

        auto_print = self._settings.get("printer.auto_print", False)
        self._auto_print_var.set(bool(auto_print))

        self._gen_poll_var.set(str(self._settings.get("generator.poll_interval", 10)))
        self._layout_poll_var.set(str(self._settings.get("layout.poll_interval", 10)))

        mode = self._settings.get("ui.appearance_mode", "dark")
        self._appearance_var.set(mode if mode in APPEARANCE_MODES else "dark")

    def _save(self) -> None:
        """Write current values back to Settings and persist to disk."""
        try:
            copies = int(self._copies_var.get())
        except ValueError:
            copies = 1

        try:
            gen_poll = max(1, int(self._gen_poll_var.get()))
        except ValueError:
            gen_poll = 10

        try:
            layout_poll = max(1, int(self._layout_poll_var.get()))
        except ValueError:
            layout_poll = 10

        self._settings.set("printer.name", self._printer_var.get())
        self._settings.set("printer.paper_size", self._paper_var.get())
        self._settings.set("printer.copies", copies)
        self._settings.set("printer.auto_print", self._auto_print_var.get())
        self._settings.set("generator.poll_interval", gen_poll)
        self._settings.set("layout.poll_interval", layout_poll)
        self._settings.set("ui.appearance_mode", self._appearance_var.get())
        self._settings.save()

        logger.info("Settings saved.")
        self.destroy()

