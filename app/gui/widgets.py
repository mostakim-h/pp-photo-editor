"""
widgets.py
----------
Reusable CustomTkinter widget helpers shared across the application.
"""

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk


# ---------------------------------------------------------------------------
# Section header label
# ---------------------------------------------------------------------------

class SectionLabel(ctk.CTkLabel):
    """A bold, slightly larger label used for section headers."""

    def __init__(self, master, text: str, **kwargs) -> None:
        kwargs.setdefault("font", ctk.CTkFont(size=13, weight="bold"))
        kwargs.setdefault("anchor", "w")
        super().__init__(master, text=text, **kwargs)


# ---------------------------------------------------------------------------
# Horizontal divider
# ---------------------------------------------------------------------------

class Divider(ctk.CTkFrame):
    """A thin 1-pixel horizontal divider line."""

    def __init__(self, master, **kwargs) -> None:
        kwargs.setdefault("height", 1)
        kwargs.setdefault("fg_color", ("gray70", "gray30"))
        super().__init__(master, **kwargs)


# ---------------------------------------------------------------------------
# Log console
# ---------------------------------------------------------------------------

class LogConsole(ctk.CTkFrame):
    """
    Scrollable, read-only text console for displaying log messages.

    Usage::

        console = LogConsole(parent)
        console.append("[INFO] Processing image …")
        console.clear()
    """

    # Colour tags applied to lines that start with the given prefix
    _TAG_COLORS = {
        "[INFO]":  ("#a8d8a8", "#2e7d32"),   # (dark-mode, light-mode)
        "[WARN]":  ("#ffe082", "#f57f17"),
        "[ERROR]": ("#ef9a9a", "#c62828"),
        "[DEBUG]": ("#90caf9", "#1565c0"),
    }

    def __init__(self, master, **kwargs) -> None:
        kwargs.setdefault("corner_radius", 8)
        super().__init__(master, **kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._text = tk.Text(
            self,
            state="disabled",
            wrap="word",
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            relief="flat",
            padx=8,
            pady=8,
        )
        self._text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(self, command=self._text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=scrollbar.set)

        # Configure colour tags
        for prefix, (dark_col, _light_col) in self._TAG_COLORS.items():
            self._text.tag_configure(prefix, foreground=dark_col)

    def append(self, message: str) -> None:
        """Append *message* on a new line, colour-coded by prefix."""
        self._text.configure(state="normal")

        tag = None
        for prefix in self._TAG_COLORS:
            if prefix in message:
                tag = prefix
                break

        self._text.insert("end", message + "\n", tag or "")
        self._text.configure(state="disabled")
        self._text.see("end")   # auto-scroll to bottom

    def clear(self) -> None:
        """Remove all text from the console."""
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


# ---------------------------------------------------------------------------
# Icon button (text-only variant with a consistent style)
# ---------------------------------------------------------------------------

class ActionButton(ctk.CTkButton):
    """Standard action button with consistent sizing and styling."""

    def __init__(
        self,
        master,
        text: str,
        command: Callable,
        width: int = 180,
        **kwargs,
    ) -> None:
        kwargs.setdefault("corner_radius", 8)
        kwargs.setdefault("height", 36)
        kwargs.setdefault("font", ctk.CTkFont(size=13))
        super().__init__(master, text=text, command=command, width=width, **kwargs)


# ---------------------------------------------------------------------------
# Folder path entry row
# ---------------------------------------------------------------------------

class FolderSelector(ctk.CTkFrame):
    """
    A combined Entry + Browse button for selecting a folder path.

    Usage::

        selector = FolderSelector(parent, label="Input Folder:", on_change=my_cb)
        selector.get()        # returns the selected path string
        selector.set("/path") # programmatically set the path
    """

    def __init__(
        self,
        master,
        label: str = "Folder:",
        on_change: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self._on_change = on_change

        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self, text=label, font=ctk.CTkFont(size=13, weight="bold"), width=120, anchor="w"
        ).grid(row=0, column=0, padx=(0, 8), sticky="w")

        self._var = tk.StringVar()
        self._entry = ctk.CTkEntry(
            self,
            textvariable=self._var,
            placeholder_text="No folder selected …",
            height=34,
            corner_radius=8,
        )
        self._entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            self,
            text="Browse …",
            width=100,
            height=34,
            corner_radius=8,
            command=self._browse,
        ).grid(row=0, column=2)

    def _browse(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="Select Input Folder")
        if path:
            self.set(path)

    def get(self) -> str:
        return self._var.get().strip()

    def set(self, path: str) -> None:
        self._var.set(path)
        if self._on_change:
            self._on_change(path)

