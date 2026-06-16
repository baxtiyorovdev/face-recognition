#!/usr/bin/env python3
# =============================================================
#  main.py — Точка входа. Запуск: python main.py
# =============================================================

import logging
import sys
import tkinter as tk

import config
from gui import AccessControlGUI

# ── Логирование ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("system.log", encoding="utf-8"),
    ],
)


def main():
    root = tk.Tk()
    root.geometry(f"{config.GUI_WIDTH}x{config.GUI_HEIGHT}")

    app = AccessControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
