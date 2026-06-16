#!/usr/bin/env python3
# =============================================================
#  web_app.py — Веб-интерфейс (Flask)
# =============================================================
"""
Браузерная панель управления домофоном. Запуск:

    python web_app.py

Затем открыть в браузере:  http://<ip-компьютера>:8000

Возможности:
  • Живое видео с камеры (MJPEG) с распознаванием лиц
  • Журнал «кто пришёл и когда» с фото
      – повторные появления одного человека с разницей < 5 мин → одно фото
  • Открытие двери кнопкой
  • Добавление / удаление лиц в базе (загрузка фото)
  • Гейт по движению: распознавание включается только при движении
    (5 минут активности), как в config.MOTION_*

Один поток захвата RTSP + один поток распознавания, общий с веб-сервером.
"""

import io
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, request,
                   render_template, send_from_directory, abort)

import config
from door_controller import DoorController
from face_engine import FaceEngine
from motion_detector import MotionDetector
from visitor_log import VisitorLog
import telegram_bot

try:
    from gui import draw_overlay          # переиспользуем отрисовку с кириллицей
except Exception:                          # на случай отсутствия tkinter
    draw_overlay = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("system.log", encoding="utf-8")],
)
log = logging.getLogger("WebApp")


# =============================================================
#  CameraEngine — захват RTSP + распознавание (фоновые потоки)
# =============================================================

class CameraEngine:
    """
    Поток 1 (grab):     читает RTSP, хранит последний кадр.
    Поток 2 (process):  гейт по движению → FaceEngine → аннотированный JPEG,
                        журнал посетителей, авто-открытие двери.
    """

    def __init__(self):
        self.engine = FaceEngine()
        self.motion = MotionDetector()
        self.door   = DoorController()
        self.visitors = VisitorLog()

        self._running = True
        self._fresh_frame = None
        self._fresh_lock  = threading.Lock()
        self._jpeg = None                  # последний аннотированный JPEG
        self._jpeg_lock = threading.Lock()
        self._raw_frame = None             # последний сырой кадр

        self._fps = 0.0
        self._device_online = False
        self._door_just_opened: set = set()

        self._cap = None
        threading.Thread(target=self._grab_loop,    daemon=True, name="WebGrab").start()
        threading.Thread(target=self._process_loop, daemon=True, name="WebProc").start()
        threading.Thread(target=self._device_loop,  daemon=True, name="WebDev").start()

    # ── Захват ───────────────────────────────────────────────────

    def _build_cap(self):
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
            "analyzeduration;0|probesize;32|max_delay;0|reorder_queue_size;0"
        )
        cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            return cap
        return cv2.VideoCapture(config.RTSP_URL)

    def _grab_loop(self):
        log.info("Подключение к камере %s", config.RTSP_URL)
        self._cap = self._build_cap()
        reconnect = 1.0
        fps_n, fps_t = 0, time.time()
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._build_cap()
                time.sleep(reconnect)
                continue
            ret, frame = self._cap.read()
            if not ret:
                log.warning("Потеря кадра — переподключение через %.1fs", reconnect)
                time.sleep(reconnect)
                reconnect = min(reconnect * 1.5, 10.0)
                self._cap.release()
                self._cap = self._build_cap()
                self.motion.reset()
                continue
            reconnect = 1.0
            with self._fresh_lock:
                self._fresh_frame = frame
            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.5:
                self._fps = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now

    # ── Обработка ────────────────────────────────────────────────

    def _process_loop(self):
        last = None
        while self._running:
            with self._fresh_lock:
                frame = self._fresh_frame
            if frame is None or frame is last:
                time.sleep(0.005)
                continue
            last = frame
            self._raw_frame = frame

            if getattr(config, "MOTION_DETECTION", False):
                active = self.motion.update(frame)
            else:
                active = True

            if active:
                results = self.engine.process(frame)
                annotated = draw_overlay(frame, results) if draw_overlay else frame
                self._handle_results(results, frame)
            else:
                results = []
                annotated = self._standby(frame)

            ok, buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, getattr(config, "WEB_JPEG_QUALITY", 75)])
            if ok:
                with self._jpeg_lock:
                    self._jpeg = buf.tobytes()

    def _standby(self, frame):
        out = frame.copy()
        cv2.putText(out, "STANDBY - no motion", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
        return out

    def _handle_results(self, results, frame):
        if not results:
            self._door_just_opened.clear()
            return
        best = max(results, key=lambda r: r.confidence)

        # ── Журнал посетителей (с дедупликацией 5 мин) ───────────
        if best.is_live:
            event = self.visitors.record(best.name, best.confidence,
                                         best.is_live, frame)
            if event and getattr(config, "TG_NOTIFY_VISITOR", False):
                name = best.name
                if name == "Unknown" and getattr(config, "TG_NOTIFY_UNKNOWN", False):
                    telegram_bot.notify_visitor(frame, "Unknown", best.confidence)
                elif name != "Unknown":
                    telegram_bot.notify_visitor(frame, name, best.confidence)

        # ── Авто-открытие двери ──────────────────────────────────
        if best.door_ready and best.name not in self._door_just_opened:
            self._door_just_opened.add(best.name)
            threading.Thread(target=self._open_door,
                             args=(best.name, best.confidence, frame),
                             daemon=True).start()

        active = {r.name for r in results}
        self._door_just_opened &= active

    def _open_door(self, name, conf, frame):
        ok = self.door.open()
        if ok and getattr(config, "TG_NOTIFY_DOOR_OPEN", False):
            telegram_bot.notify_door_opened(frame, name, conf)
        log.info("🚪 Дверь %s для %s", "открыта" if ok else "НЕ открыта", name)
        time.sleep(config.DOOR_OPEN_SECONDS + 1)
        self._door_just_opened.discard(name)
        self.engine.reset_hold(name)

    # ── Проверка устройства ──────────────────────────────────────

    def _device_loop(self):
        while self._running:
            self._device_online = self.door.check_online()
            time.sleep(15)

    # ── Доступ для веб-сервера ───────────────────────────────────

    def get_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def status(self) -> dict:
        motion_on = getattr(config, "MOTION_DETECTION", False)
        return {
            "fps": round(self._fps, 1),
            "device_online": self._device_online,
            "door_open": self.door.is_open,
            "known_count": self.engine.known_count,
            "recognition_active": (not motion_on) or self.motion.active,
            "motion_gate": motion_on,
            "seconds_left": round(self.motion.seconds_left) if motion_on else None,
        }


# =============================================================
#  Flask
# =============================================================

app = Flask(__name__)
cam: CameraEngine | None = None


def _get_cam() -> CameraEngine:
    global cam
    if cam is None:
        cam = CameraEngine()
    return cam


@app.route("/")
def index():
    return render_template("index.html",
                           title=getattr(config, "GUI_TITLE", "Домофон"))


@app.route("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpeg = _get_cam().get_jpeg()
            if jpeg:
                yield boundary + jpeg + b"\r\n"
            time.sleep(1 / 25)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(_get_cam().status())


@app.route("/api/visitors")
def api_visitors():
    limit = int(request.args.get("limit", 100))
    return jsonify(_get_cam().visitors.list(limit))


@app.route("/api/visitors/clear", methods=["POST"])
def api_visitors_clear():
    _get_cam().visitors.clear()
    return jsonify({"ok": True})


@app.route("/visitors/photos/<path:photo>")
def visitor_photo(photo):
    return send_from_directory(_get_cam().visitors.photos, photo)


@app.route("/api/open_door", methods=["POST"])
def api_open_door():
    c = _get_cam()
    ok = c.door.open()
    if ok and getattr(config, "TG_NOTIFY_DOOR_OPEN", False):
        telegram_bot.notify_door_opened(c._raw_frame, "manual", 1.0, manual=True)
    return jsonify({"ok": ok})


# ── Управление базой лиц ─────────────────────────────────────────

@app.route("/api/faces", methods=["GET"])
def api_faces_list():
    p = Path(config.KNOWN_FACES_DIR)
    faces = []
    if p.exists():
        for f in sorted(p.glob("*.[jp][pn][ge]*")):
            faces.append({"file": f.name,
                          "name": f.stem.replace("_", " ").title()})
    return jsonify(faces)


@app.route("/api/faces", methods=["POST"])
def api_faces_add():
    name = (request.form.get("name") or "").strip()
    file = request.files.get("photo")
    if not name or not file:
        return jsonify({"ok": False, "error": "Нужны имя и фото"}), 400

    p = Path(config.KNOWN_FACES_DIR)
    p.mkdir(parents=True, exist_ok=True)
    safe = "_".join(name.split())
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    dest = p / f"{safe}{ext}"

    data = file.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return jsonify({"ok": False, "error": "Не удалось прочитать изображение"}), 400

    # Проверяем, что на фото есть лицо
    c = _get_cam()
    try:
        from face_engine import _get_app
        faces = _get_app().get(arr)
    except Exception:
        faces = None
    if faces is not None and len(faces) == 0:
        return jsonify({"ok": False, "error": "Лицо на фото не найдено"}), 400

    cv2.imwrite(str(dest), arr)
    c.engine.reload()
    log.info("Добавлено лицо: %s → %s", name, dest.name)
    return jsonify({"ok": True, "file": dest.name,
                    "known_count": c.engine.known_count})


@app.route("/api/faces/<path:filename>", methods=["DELETE"])
def api_faces_delete(filename):
    p = Path(config.KNOWN_FACES_DIR) / filename
    # защита от выхода за пределы папки
    if p.parent.resolve() != Path(config.KNOWN_FACES_DIR).resolve():
        abort(403)
    if p.exists():
        p.unlink()
        _get_cam().engine.reload()
        return jsonify({"ok": True, "known_count": _get_cam().engine.known_count})
    return jsonify({"ok": False, "error": "Файл не найден"}), 404


# =============================================================
#  Запуск
# =============================================================

def main():
    _get_cam()   # стартуем камеру до сервера
    host = getattr(config, "WEB_HOST", "0.0.0.0")
    port = getattr(config, "WEB_PORT", 8000)
    log.info("Веб-интерфейс: http://%s:%d  (Ctrl+C для остановки)", host, port)
    # threaded=True — несколько одновременных клиентов (видео + API)
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
