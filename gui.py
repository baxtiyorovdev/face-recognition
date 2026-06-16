# =============================================================
#  gui.py — Tkinter GUI: видео, статусы, лог, настройки
# =============================================================

import csv
import logging
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

import config
from door_controller import DoorController
from face_engine import FaceEngine, FaceResult
from motion_detector import MotionDetector
import telegram_bot
import config as _cfg

log = logging.getLogger("GUI")

# ── Цветовая палитра ──────────────────────────────────────────
CLR = {
    "bg":        "#080C12",
    "panel":     "#0D1118",
    "card":      "#111722",
    "border":    "#1A2535",
    "text":      "#C8D8E8",
    "muted":     "#4A6080",
    "green":     "#22C55E",
    "green_dim": "#0A2E1A",
    "red":       "#EF4444",
    "red_dim":   "#2E0A0A",
    "yellow":    "#FBBF24",
    "blue":      "#3B82F6",
    "blue_dim":  "#0A1830",
    "white":     "#FFFFFF",
}


# ── Вспомогательные функции отрисовки ────────────────────────

def _hex_to_bgr(h: str) -> tuple:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def _get_pil_font(size: int = 14):
    """Загружает шрифт с поддержкой кириллицы. Fallback — дефолтный PIL."""
    FONT_CANDIDATES = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        # macOS
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _put_text_pil(pil_img: Image.Image, text: str, xy: tuple,
                  color: str, font) -> None:
    """Рисует Unicode-текст (включая кириллицу) на PIL-изображении."""
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, fill=color, font=font)


def draw_overlay(frame, results: list[FaceResult]) -> object:
    """
    Рисует боксы, прогресс-бары и подписи поверх кадра (OpenCV BGR).
    Использует PIL для корректного отображения кириллицы.
    Возвращает аннотированный BGR numpy-массив.
    """
    import numpy as np

    out = frame.copy()

    # ── Загружаем шрифты один раз через PIL ──────────────────
    font_main  = _get_pil_font(15)   # имя + уверенность
    font_small = _get_pil_font(13)   # статус живости / прогресс
    font_big   = _get_pil_font(22)   # «ДОСТУП РАЗРЕШЁН»

    for r in results:
        top, right, bottom, left = r.bbox

        # ── Цвет рамки ──────────────────────────────────────
        if not r.is_live:
            hex_c = CLR["red"]
        elif r.name == "Unknown":
            hex_c = CLR["yellow"]
        elif r.door_ready:
            hex_c = CLR["green"]
        else:
            pct   = min(r.held_seconds / config.HOLD_SECONDS, 1.0)
            g     = int(80 + 175 * pct)
            b     = int(246 * (1 - pct))
            hex_c = f"#{b:02x}{g:02x}{b:02x}"

        bgr = _hex_to_bgr(hex_c)

        # ── Рамка лица ──────────────────────────────────────
        cv2.rectangle(out, (left, top), (right, bottom), bgr, 2)

        # ── Угловые маркеры ─────────────────────────────────
        corner, thick = 16, 3
        for x1, y1, dx, dy in [
            (left,  top,    +1, +1),
            (right, top,    -1, +1),
            (left,  bottom, +1, -1),
            (right, bottom, -1, -1),
        ]:
            cv2.line(out, (x1, y1), (x1 + dx * corner, y1), bgr, thick)
            cv2.line(out, (x1, y1), (x1, y1 + dy * corner), bgr, thick)

        # ── Подпись: конвертируем кадр в PIL для текста ──────
        label    = r.name if r.name != "Unknown" else "Неизвестный"
        conf_str = f"  {r.confidence:.0%}" if r.confidence > 0 else ""
        live_str = "✓ Живой" if r.is_live else "✗ Фото/Рисунок"
        live_col = CLR["green"] if r.is_live else CLR["red"]

        # Фоновый прямоугольник под текст
        tx, ty = left, max(top - 42, 0)
        tw = max(right - left, 170)
        cv2.rectangle(out, (tx, ty), (tx + tw, top - 2), (0, 0, 0), -1)
        cv2.rectangle(out, (tx, ty), (tx + tw, top - 2), bgr, 1)

        # PIL → рисуем текст → обратно в numpy
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        _put_text_pil(pil, label + conf_str, (tx + 6, ty + 2),  hex_c,  font_main)
        _put_text_pil(pil, live_str,          (tx + 6, ty + 20), live_col, font_small)
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # ── Прогресс-бар ─────────────────────────────────────
        if r.is_live and r.name != "Unknown":
            pct  = min(r.held_seconds / config.HOLD_SECONDS, 1.0)
            bx1, by1 = left,  bottom + 6
            bx2, by2 = right, bottom + 16
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
            fill_x = bx1 + int((bx2 - bx1) * pct)
            cv2.rectangle(out, (bx1, by1), (fill_x, by2), bgr, -1)

            pct_txt = f"{r.held_seconds:.1f}/{config.HOLD_SECONDS}с"
            pil2 = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            _put_text_pil(pil2, pct_txt, (bx1, by2 + 2),
                          CLR["muted"], font_small)
            out = cv2.cvtColor(np.array(pil2), cv2.COLOR_RGB2BGR)

        # ── ДОСТУП РАЗРЕШЁН overlay ───────────────────────────
        if r.door_ready:
            overlay = out.copy()
            cv2.rectangle(overlay, (left, top), (right, bottom),
                          _hex_to_bgr(CLR["green"]), -1)
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

            pil3 = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            cx = (left + right) // 2
            cy = (top  + bottom) // 2
            _put_text_pil(pil3, "ДОСТУП",   (cx - 44, cy - 26), CLR["white"], font_big)
            _put_text_pil(pil3, "РАЗРЕШЁН", (cx - 52, cy + 2),  CLR["white"], font_big)
            out = cv2.cvtColor(np.array(pil3), cv2.COLOR_RGB2BGR)

    return out


def _draw_standby(frame) -> object:
    """
    Рисует индикатор режима ожидания (нет движения — распознавание выключено).
    Возвращает кадр с лёгкой подписью в углу.
    """
    import numpy as np

    out = frame.copy()
    font = _get_pil_font(16)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    _put_text_pil(pil, "● ОЖИДАНИЕ — нет движения", (16, 14),
                  CLR["muted"], font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── Класс GUI ─────────────────────────────────────────────────

class AccessControlGUI:

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.engine = FaceEngine()
        self.motion = MotionDetector()
        self.door   = DoorController(on_state_change=self._on_door_state)

        self._cap        = None
        self._running    = False
        self._cam_thread = None
        self._q_frame    = queue.Queue(maxsize=2)   # BGR кадры
        self._q_result   = queue.Queue(maxsize=2)   # (annotated_bgr, results)
        self._door_just_opened: set = set()         # имена, кому уже открыли
        self._last_raw_frame = None                      # последний сырой кадр
        self._tg_visitor_notified: set = set()           # кому уже отправили «подошёл»
        self._frame_count = 0
        self._fps_ts      = time.time()
        self._fps_val     = 0.0
        self._device_online = False

        self._build_ui()
        self._init_log_file()
        self._check_device()
        self._start_camera()
        self.root.after(33, self._gui_update_loop)   # ~30 fps GUI

    # ── Проверка устройства ───────────────────────────────────

    def _check_device(self):
        def _check():
            online = self.door.check_online()
            self._device_online = online
            status = "ONLINE ●" if online else "OFFLINE ●"
            color  = CLR["green"] if online else CLR["red"]
            self.root.after(0, lambda: (
                self._lbl_device.config(text=status, fg=color)
            ))
        threading.Thread(target=_check, daemon=True).start()

    # ── Построение UI ─────────────────────────────────────────

    def _build_ui(self):
        self.root.title(config.GUI_TITLE)
        self.root.configure(bg=CLR["bg"])
        self.root.resizable(False, False)

        # Шрифты
        self._fn  = tkfont.Font(family="Courier", size=9)
        self._fn_b= tkfont.Font(family="Courier", size=9,  weight="bold")
        self._fn_l= tkfont.Font(family="Courier", size=8)
        self._fn_h= tkfont.Font(family="Courier", size=11, weight="bold")
        self._fn_t= tkfont.Font(family="Courier", size=16, weight="bold")

        self._build_header()
        body = tk.Frame(self.root, bg=CLR["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._build_camera(body)
        self._build_right_panel(body)
        self._build_statusbar()

    # ── Хедер ────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=CLR["panel"],
                       highlightbackground=CLR["border"], highlightthickness=1)
        hdr.pack(fill=tk.X, padx=0, pady=0)

        tk.Label(hdr, text="DAHUA VTO 2111D-P-S3",
                 font=self._fn_l, fg=CLR["muted"], bg=CLR["panel"]
                 ).pack(side=tk.LEFT, padx=(16, 4), pady=10)
        tk.Label(hdr, text="·  СИСТЕМА КОНТРОЛЯ ДОСТУПА ПО ЛИЦУ",
                 font=self._fn_b, fg=CLR["text"], bg=CLR["panel"]
                 ).pack(side=tk.LEFT, pady=10)

        right = tk.Frame(hdr, bg=CLR["panel"])
        right.pack(side=tk.RIGHT, padx=16)
        tk.Label(right, text="IP: " + config.DAHUA_IP,
                 font=self._fn_l, fg=CLR["muted"], bg=CLR["panel"]
                 ).pack(side=tk.LEFT, padx=8)
        self._lbl_device = tk.Label(right, text="Проверка...",
                                    font=self._fn_b, fg=CLR["yellow"],
                                    bg=CLR["panel"])
        self._lbl_device.pack(side=tk.LEFT)

    # ── Камера ────────────────────────────────────────────────

    def _build_camera(self, parent):
        cam_frame = tk.Frame(parent, bg=CLR["bg"])
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)

        # Метка-холст для видео
        self._lbl_cam = tk.Label(cam_frame, bg="#000000",
                                  width=config.CAMERA_W, height=config.CAMERA_H,
                                  text="Подключение к камере...",
                                  font=self._fn, fg=CLR["muted"])
        self._lbl_cam.pack(fill=tk.BOTH, expand=True)

        # FPS строка
        fps_row = tk.Frame(cam_frame, bg=CLR["bg"])
        fps_row.pack(fill=tk.X, pady=(4, 0))
        self._lbl_fps = tk.Label(fps_row, text="FPS: --",
                                  font=self._fn_l, fg=CLR["muted"], bg=CLR["bg"])
        self._lbl_fps.pack(side=tk.LEFT)
        tk.Label(fps_row, text=f"Задержка: {config.HOLD_SECONDS}с  |  "
                               f"Порог: {config.FACE_CONFIDENCE_THRES:.0%}  |  "
                               f"База: {self.engine.known_count} лиц",
                 font=self._fn_l, fg=CLR["muted"], bg=CLR["bg"]
                 ).pack(side=tk.RIGHT)

    # ── Правая панель ─────────────────────────────────────────

    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg=CLR["bg"], width=310)
        panel.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        panel.pack_propagate(False)

        self._build_door_card(panel)
        self._build_event_card(panel)
        self._build_log_card(panel)
        self._build_controls(panel)

    def _card(self, parent, title: str) -> tk.Frame:
        """Возвращает контентный фрейм карточки."""
        outer = tk.Frame(parent, bg=CLR["card"],
                         highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill=tk.X, pady=(0, 8))
        tk.Label(outer, text=title, font=self._fn_l,
                 fg=CLR["muted"], bg=CLR["card"]
                 ).pack(anchor=tk.W, padx=10, pady=(8, 4))
        inner = tk.Frame(outer, bg=CLR["card"])
        inner.pack(fill=tk.X, padx=10, pady=(0, 10))
        return inner

    def _build_door_card(self, parent):
        c = tk.Frame(parent, bg=CLR["red_dim"],
                     highlightbackground=CLR["red"], highlightthickness=1)
        c.pack(fill=tk.X, pady=(0, 8))
        self._door_card = c

        self._lbl_door_icon = tk.Label(c, text="🔒", font=("", 28),
                                        bg=CLR["red_dim"])
        self._lbl_door_icon.pack(pady=(12, 0))
        self._lbl_door_text = tk.Label(c, text="ЗАКРЫТО",
                                        font=self._fn_t, fg=CLR["red"],
                                        bg=CLR["red_dim"])
        self._lbl_door_text.pack(pady=(4, 14))

    def _build_event_card(self, parent):
        f = self._card(parent, "ТЕКУЩЕЕ СОБЫТИЕ")
        rows = [
            ("Имя",         "—", "name"),
            ("Уверенность", "—", "conf"),
            ("Живость",     "—", "live"),
            ("Удержание",   "—", "hold"),
            ("Статус",      "—", "status"),
        ]
        self._ev = {}
        for label, default, key in rows:
            row = tk.Frame(f, bg=CLR["card"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=self._fn, fg=CLR["muted"],
                     bg=CLR["card"], width=13, anchor=tk.W).pack(side=tk.LEFT)
            lbl = tk.Label(row, text=default, font=self._fn_b,
                           fg=CLR["text"], bg=CLR["card"], anchor=tk.W)
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._ev[key] = lbl

        # прогресс-бар 3 сек
        tk.Label(f, text="Прогресс (3 сек):", font=self._fn_l,
                 fg=CLR["muted"], bg=CLR["card"]).pack(anchor=tk.W, pady=(6, 2))
        self._progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Green.Horizontal.TProgressbar",
                        background=CLR["green"], troughcolor=CLR["border"],
                        thickness=8)
        self._progress = ttk.Progressbar(f, variable=self._progress_var,
                                          maximum=100,
                                          style="Green.Horizontal.TProgressbar")
        self._progress.pack(fill=tk.X, pady=(0, 4))

    def _build_log_card(self, parent):
        outer = tk.Frame(parent, bg=CLR["card"],
                         highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        tk.Label(outer, text="ЛОГ ДОСТУПА", font=self._fn_l,
                 fg=CLR["muted"], bg=CLR["card"]
                 ).pack(anchor=tk.W, padx=10, pady=(8, 2))

        frame = tk.Frame(outer, bg=CLR["card"])
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 8))

        sb = tk.Scrollbar(frame, bg=CLR["border"])
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_box = tk.Listbox(
            frame,
            font=self._fn_l,
            bg=CLR["card"],
            fg=CLR["muted"],
            selectbackground=CLR["border"],
            activestyle="none",
            highlightthickness=0,
            bd=0,
            yscrollcommand=sb.set,
        )
        self._log_box.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._log_box.yview)

    def _build_controls(self, parent):
        f = self._card(parent, "УПРАВЛЕНИЕ")
        btn_cfg = [
            ("🔓  Открыть вручную", self._manual_open,  CLR["green"]),
            ("↺  Обновить базу лиц", self._reload_faces, CLR["blue"]),
        ]
        for txt, cmd, col in btn_cfg:
            btn = tk.Button(f, text=txt, command=cmd,
                            font=self._fn_b, fg=col, bg=CLR["card"],
                            activebackground=CLR["border"],
                            activeforeground=col,
                            relief=tk.FLAT, bd=0, pady=6, cursor="hand2",
                            highlightbackground=col, highlightthickness=1)
            btn.pack(fill=tk.X, pady=3)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=CLR["panel"],
                       highlightbackground=CLR["border"], highlightthickness=1)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self._lbl_status = tk.Label(bar, text="Запуск...",
                                     font=self._fn_l, fg=CLR["muted"],
                                     bg=CLR["panel"])
        self._lbl_status.pack(side=tk.LEFT, padx=12, pady=4)
        ts = datetime.now().strftime("%d.%m.%Y")
        tk.Label(bar, text=ts, font=self._fn_l,
                 fg=CLR["muted"], bg=CLR["panel"]
                 ).pack(side=tk.RIGHT, padx=12)

    # ── CSV лог ───────────────────────────────────────────────

    def _init_log_file(self):
        p = Path(config.LOG_FILE)
        if not p.exists():
            with open(p, "w", newline="", encoding="utf-8") as f:
                import csv
                csv.writer(f).writerow(
                    ["timestamp", "name", "confidence", "is_live", "action"]
                )

    def _write_log(self, name, conf, is_live, action):
        import csv
        with open(config.LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                name, f"{conf:.2%}", is_live, action,
            ])

    # ── Запуск камеры ─────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════
    #  ВИДЕОЗАХВАТ — двухпоточная схема с минимальной задержкой
    #
    #  Поток 1 (_grab_loop): непрерывно читает кадры из RTSP
    #    и кладёт ТОЛЬКО ПОСЛЕДНИЙ в self._fresh_frame (lock-free).
    #    Никакой обработки — только cap.read() в цикле.
    #
    #  Поток 2 (_process_loop): берёт свежий кадр из _fresh_frame,
    #    запускает face_engine, кладёт результат в GUI-очередь.
    #
    #  Итог: GUI всегда получает кадр возраста < 1 сетевого RTT,
    #    независимо от скорости face_engine.
    # ═══════════════════════════════════════════════════════════

    def _build_cap(self) -> cv2.VideoCapture:
        """
        Открывает RTSP с параметрами FFmpeg для минимальной задержки.
        Возможные бэкенды (в порядке приоритета):
          1. FFmpeg + fflags nobuffer + analyzeduration=0 (всегда доступен)
          2. GStreamer (если собран с OpenCV) — ещё ниже задержка
        """
        url = config.RTSP_URL

        # Попытка 1: GStreamer (если доступен) — самая малая задержка
        gst_pipe = (
            f"rtspsrc location={url} latency=0 buffer-mode=auto ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink drop=1 max-buffers=1 sync=false"
        )
        if cv2.CAP_GSTREAMER in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]:
            cap_gst = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
            if cap_gst.isOpened():
                log.info("VideoCapture: GStreamer (latency≈0)")
                return cap_gst
            cap_gst.release()

        # Попытка 2: FFmpeg с агрессивными настройками низкой задержки
        # CAP_PROP_OPEN_TIMEOUT_MSEC и CAP_PROP_READ_TIMEOUT_MSEC
        # задаются через переменную окружения OPENCV_FFMPEG_CAPTURE_OPTIONS
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"          # TCP надёжнее UDP для домофонов
            "fflags;nobuffer|"             # отключаем буферизацию FFmpeg
            "flags;low_delay|"             # low_delay декодирование H.264
            "analyzeduration;0|"           # не анализировать поток (быстрый старт)
            "probesize;32|"                # минимальный probesize (32 байт)
            "avioflags;direct|"            # прямой I/O без буфера libavio
            "max_delay;0|"                 # нулевая максимальная задержка пакетов
            "reorder_queue_size;0"         # нет переупорядочивания пакетов
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # OpenCV internal buffer = 1 кадр
        if cap.isOpened():
            log.info("VideoCapture: FFmpeg low-latency (nobuffer+low_delay)")
            return cap

        # Попытка 3: автовыбор бэкенда
        cap_auto = cv2.VideoCapture(url)
        cap_auto.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.info("VideoCapture: auto backend")
        return cap_auto

    def _start_camera(self):
        self._running       = True
        self._fresh_frame   = None          # самый свежий кадр (пишет grab-поток)
        self._fresh_lock    = threading.Lock()
        self._cap           = None

        self._grab_thread    = threading.Thread(target=self._grab_loop,    daemon=True, name="GrabLoop")
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True, name="ProcessLoop")
        self._grab_thread.start()
        self._process_thread.start()

    # ── Поток 1: только захват, без обработки ────────────────

    def _grab_loop(self):
        """
        Непрерывно читает кадры из RTSP и сохраняет ПОСЛЕДНИЙ.
        Никакой обработки — только I/O. Это минимизирует задержку.
        """
        log.info("GrabLoop: открытие %s", config.RTSP_URL)
        self._cap = self._build_cap()

        if not self._cap.isOpened():
            log.error("GrabLoop: не удалось открыть поток")
            self.root.after(0, lambda: self._lbl_status.config(
                text="❌ Нет подключения к камере. Проверьте RTSP-URL.",
                fg=CLR["red"]))
            return

        fps_n = 0
        fps_t = time.time()
        reconnect_delay = 1.0

        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                log.warning("GrabLoop: потеря кадра — переподключение через %.1fs", reconnect_delay)
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 10.0)
                self._cap.release()
                self._cap = self._build_cap()
                continue

            reconnect_delay = 1.0   # сброс при успехе

            # Кладём только самый свежий кадр (старый дропаем)
            with self._fresh_lock:
                self._fresh_frame = frame

            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.5:
                self._fps_val = fps_n / (now - fps_t)
                fps_n = 0
                fps_t = now

        if self._cap:
            self._cap.release()

    # ── Поток 2: обработка лиц ───────────────────────────────

    def _process_loop(self):
        """
        Берёт свежий кадр → face_engine → GUI-очередь.
        Работает столько, сколько успевает — без блокировки grab-потока.
        """
        frame_n = 0
        last_processed = None   # не обрабатываем один и тот же кадр дважды

        while self._running:
            # Ждём новый кадр
            with self._fresh_lock:
                frame = self._fresh_frame

            if frame is None or frame is last_processed:
                time.sleep(0.005)   # 5 мс пауза чтобы не жечь CPU вхолостую
                continue

            last_processed = frame
            frame_n += 1

            if frame_n % config.PROCESS_EVERY != 0:
                continue

            self._last_raw_frame = frame

            # ── Гейт по движению ─────────────────────────────────
            # Нет движения и таймер истёк → не запускаем FaceEngine,
            # показываем «живое» видео без распознавания (экономия CPU).
            if config.MOTION_DETECTION:
                active = self.motion.update(frame)
            else:
                active = True

            if active:
                results   = self.engine.process(frame)
                annotated = draw_overlay(frame, results)
            else:
                results   = []
                annotated = _draw_standby(frame)

            try:
                self._q_result.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q_result.put_nowait((annotated, results))
            except queue.Full:
                pass

    # ── GUI-цикл обновления (главный поток) ───────────────────

    def _gui_update_loop(self):
        try:
            annotated, results = self._q_result.get_nowait()
            self._show_frame(annotated)
            self._process_results(results)
            if config.MOTION_DETECTION and not self.motion.active:
                recog = "Распознавание: ОЖИДАНИЕ"
            elif config.MOTION_DETECTION:
                recog = f"Распознавание: ON ({self.motion.seconds_left:.0f}с)"
            else:
                recog = "Распознавание: ON"
            self._lbl_fps.config(
                text=f"FPS: {self._fps_val:.1f}  |  "
                     f"База: {self.engine.known_count} лиц  |  "
                     f"{recog}")
        except queue.Empty:
            pass
        self.root.after(33, self._gui_update_loop)

    # ── Отображение кадра ─────────────────────────────────────

    def _show_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (config.CAMERA_W, config.CAMERA_H))
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self._lbl_cam.imgtk = imgtk
        self._lbl_cam.configure(image=imgtk, text="")

    # ── Обработка результатов ─────────────────────────────────

    def _process_results(self, results: list[FaceResult]):
        if not results:
            self._update_event(None)
            return

        # Берём лицо с наибольшей уверенностью
        best = max(results, key=lambda r: r.confidence)
        self._update_event(best)

        # TG: уведомление о посетителе (один раз пока лицо в кадре)
        if _cfg.TG_NOTIFY_VISITOR and best.is_live:
            visitor_key = best.name
            if visitor_key not in self._tg_visitor_notified:
                self._tg_visitor_notified.add(visitor_key)
                frame = self._last_raw_frame
                if frame is not None:
                    if best.name == "Unknown" and _cfg.TG_NOTIFY_UNKNOWN:
                        telegram_bot.notify_visitor(frame, "Unknown", best.confidence)
                    elif best.name != "Unknown":
                        telegram_bot.notify_visitor(frame, best.name, best.confidence)

        # Открываем дверь
        if best.door_ready and best.name not in self._door_just_opened:
            self._door_just_opened.add(best.name)
            log.info("🚪 Открываю дверь для [%s]", best.name)
            threading.Thread(
                target=self._do_open_door,
                args=(best.name, best.confidence, best.is_live, self._last_raw_frame),
                daemon=True,
            ).start()

        # Очищаем «уже открыто» для ушедших лиц
        active = {r.name for r in results}
        for gone in list(self._door_just_opened):
            if gone not in active:
                self._door_just_opened.discard(gone)
                self.engine.reset_hold(gone)
        # Сброс TG-уведомлений для ушедших лиц
        for gone in list(self._tg_visitor_notified):
            if gone not in active:
                self._tg_visitor_notified.discard(gone)

    def _do_open_door(self, name, conf, is_live, frame=None):
        ok     = self.door.open()
        action = "ДОСТУП РАЗРЕШЁН" if ok else "ОШИБКА РЕЛЕ"
        if ok and _cfg.TG_NOTIFY_DOOR_OPEN:
            telegram_bot.notify_door_opened(frame, name, conf)
        self._write_log(name, conf, is_live, action)
        col    = CLR["green"] if ok else CLR["red"]
        self.root.after(0, lambda: (
            self._add_log(name, action, col),
            self._lbl_status.config(
                text=f"{'✅' if ok else '❌'} {name} — {action}",
                fg=col),
        ))
        # Сбрасываем «уже открыто» после закрытия двери
        time.sleep(config.DOOR_OPEN_SECONDS + 1)
        self._door_just_opened.discard(name)
        self.engine.reset_hold(name)

    # ── Обновление карточки события ───────────────────────────

    def _update_event(self, r: FaceResult | None):
        if r is None:
            self._ev["name"].config(text="—", fg=CLR["muted"])
            self._ev["conf"].config(text="—", fg=CLR["muted"])
            self._ev["live"].config(text="—", fg=CLR["muted"])
            self._ev["hold"].config(text="—", fg=CLR["muted"])
            self._ev["status"].config(text="Нет лица", fg=CLR["muted"])
            self._progress_var.set(0)
            return

        self._ev["name"].config(
            text=r.name if r.name != "Unknown" else "Неизвестный",
            fg=CLR["text"] if r.name != "Unknown" else CLR["yellow"])
        self._ev["conf"].config(
            text=f"{r.confidence:.0%}",
            fg=CLR["green"] if r.confidence >= config.FACE_CONFIDENCE_THRES else CLR["red"])
        self._ev["live"].config(
            text="✓ Живой" if r.is_live else "✗ Фото/Рисунок",
            fg=CLR["green"] if r.is_live else CLR["red"])
        self._ev["hold"].config(
            text=f"{r.held_seconds:.1f} / {config.HOLD_SECONDS} сек",
            fg=CLR["text"])

        if not r.is_live:
            status, sc = "Антиспуфинг: ОТКЛОНЕНО", CLR["red"]
        elif r.name == "Unknown":
            status, sc = "Неизвестный — доступ запрещён", CLR["yellow"]
        elif r.door_ready:
            status, sc = "✅ Открываю дверь...", CLR["green"]
        else:
            status, sc = f"Удержание {r.held_seconds:.1f}с...", CLR["blue"]

        self._ev["status"].config(text=status, fg=sc)
        pct = min(r.held_seconds / config.HOLD_SECONDS * 100, 100) if r.is_live and r.name != "Unknown" else 0
        self._progress_var.set(pct)

    # ── Состояние двери ───────────────────────────────────────

    def _on_door_state(self, is_open: bool):
        if is_open:
            cfg = dict(bg=CLR["green_dim"],
                       highlightbackground=CLR["green"], highlightthickness=1)
            self._lbl_door_icon.config(text="🔓", bg=CLR["green_dim"])
            self._lbl_door_text.config(text="ОТКРЫТО", fg=CLR["green"],
                                        bg=CLR["green_dim"])
        else:
            cfg = dict(bg=CLR["red_dim"],
                       highlightbackground=CLR["red"], highlightthickness=1)
            self._lbl_door_icon.config(text="🔒", bg=CLR["red_dim"])
            self._lbl_door_text.config(text="ЗАКРЫТО", fg=CLR["red"],
                                        bg=CLR["red_dim"])
        self.root.after(0, lambda: self._door_card.config(**cfg))

    # ── Лог ───────────────────────────────────────────────────

    def _add_log(self, name: str, action: str, color: str = CLR["muted"]):
        ts  = datetime.now().strftime("%H:%M:%S")
        txt = f"{ts}  {name:<20} {action}"
        self._log_box.insert(0, txt)
        self._log_box.itemconfig(0, fg=color)
        if self._log_box.size() > 100:
            self._log_box.delete(100, tk.END)

    # ── Ручное управление ─────────────────────────────────────

    def _manual_open(self):
        threading.Thread(target=self.door.open, daemon=True).start()
        self._add_log("РУЧНОЕ", "ДВЕРЬ ОТКРЫТА ВРУЧНУЮ", CLR["yellow"])
        self._write_log("manual", 1.0, True, "РУЧНОЕ ОТКРЫТИЕ")
        if _cfg.TG_NOTIFY_DOOR_OPEN:
            telegram_bot.notify_door_opened(self._last_raw_frame, "manual", 1.0, manual=True)

    def _reload_faces(self):
        self.engine.reload()
        msg = f"Загружено {self.engine.known_count} лиц"
        self._add_log("СИСТЕМА", msg, CLR["blue"])
        self._lbl_status.config(text=f"↺ {msg}", fg=CLR["blue"])

    # ── Завершение ────────────────────────────────────────────

    def on_close(self):
        self._running = False
        # Ждём завершения потоков (таймаут 2с)
        if hasattr(self, "_grab_thread") and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2)
        if hasattr(self, "_process_thread") and self._process_thread.is_alive():
            self._process_thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self.root.destroy()
