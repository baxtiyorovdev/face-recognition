#!/usr/bin/env python3
# =============================================================
#  headless.py — Multiprocessing-версия для Intel Xeon
# =============================================================
#
#  Архитектура процессов:
#
#   ┌──────────────┐   raw JPEG   ┌───────────────────┐
#   │ GrabProcess  │ ──────────►  │  WorkerProcess(N) │
#   │  (1 проц.)   │  frame_q     │  face_engine.py   │
#   └──────────────┘              └─────────┬─────────┘
#                                           │ result_q
#                                  ┌────────▼────────┐
#                                  │  NotifyProcess  │
#                                  │  Telegram/Door  │
#                                  └─────────────────┘
#
#  Зачем multiprocessing, а не threading:
#    - GIL блокирует NumPy/ONNX в потоках → потери ~40%
#    - Xeon имеет 8-64+ ядер: каждый процесс на своё ядро
#    - ONNX модель загружается отдельно в каждом Worker
#      (не шарится — это норма для onnxruntime)
#    - frame_q с JPEG-сжатием: передача ~30 КБ/кадр вместо 3 МБ
#
#  Настройка количества воркеров (config.py):
#    WORKER_PROCESSES = 2   # 2 для 4-ядерного, 4 для 8-ядерного Xeon
#
# =============================================================

import csv
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

import config

# ── Константы ─────────────────────────────────────────────────
WORKER_PROCESSES = getattr(config, "WORKER_PROCESSES", 2)
FRAME_Q_MAXSIZE  = 2   # не накапливаем старые кадры
RESULT_Q_MAXSIZE = 16


# =============================================================
#  Утилиты
# =============================================================

def _setup_logging(name: str):
    """Настраивает логирование для дочернего процесса."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] {name}: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("system.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger(name)


def _init_csv_log():
    p = Path(config.LOG_FILE)
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp", "name", "confidence", "is_live", "action"]
            )


def _write_csv_log(name, conf, is_live, action):
    with open(config.LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            name, f"{conf:.2%}", is_live, action,
        ])


# =============================================================
#  Процесс 1: GrabProcess — захват кадров с RTSP
# =============================================================

def _build_cap():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|"
        "fflags;discardcorrupt"
    )

    cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    return cap


def grab_process(frame_q: mp.Queue, stop_event: mp.Event):
    """
    Дочерний процесс: читает RTSP, сжимает в JPEG, кладёт в очередь.
    Сжатие JPEG здесь позволяет передавать 30 КБ вместо 3 МБ через IPC.
    """
    log = _setup_logging("Grabber")
    log.info("Подключение к %s", config.RTSP_URL)

    # Детектор движения: пока нет движения — не отправляем кадры воркерам,
    # FaceEngine простаивает. При движении — распознавание на 5 минут.
    motion = None
    if getattr(config, "MOTION_DETECTION", False):
        from motion_detector import MotionDetector
        motion = MotionDetector()
        log.info("Детектор движения включён (активность %d сек после движения)",
                 motion.active_seconds)

    cap = _build_cap()
    if not cap.isOpened():
        log.error("Не удалось открыть RTSP поток. Проверьте config.py")
        stop_event.set()
        return

    log.info("Поток открыт ✓")
    reconnect_delay = 1.0
    frame_n = 0
    was_active = True   # для лога переходов активно/простой

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            log.warning("Потеря кадра, переподключение через %.1fs", reconnect_delay)
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, 15.0)
            cap.release()
            cap = _build_cap()
            if motion is not None:
                motion.reset()
            continue

        reconnect_delay = 1.0
        frame_n += 1

        if frame_n % config.PROCESS_EVERY != 0:
            continue

        # ── Гейт по движению ─────────────────────────────────────
        if motion is not None:
            active = motion.update(frame)
            if active != was_active:
                log.info("Распознавание %s", "ВКЛЮЧЕНО (движение)"
                         if active else "ОТКЛЮЧЕНО (простой)")
                was_active = active
            if not active:
                continue   # простой: воркеры не нагружаем

        # Кодируем в JPEG для эффективной передачи через IPC
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue

        jpeg_bytes = buf.tobytes()

        # Если очередь полна — выбрасываем старый кадр (не ждём)
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except Exception:
                pass

        try:
            frame_q.put_nowait(jpeg_bytes)
        except Exception:
            pass

    cap.release()
    log.info("GrabProcess завершён")


# =============================================================
#  Процесс 2: WorkerProcess — распознавание лиц (N копий)
# =============================================================

def worker_process(worker_id: int, frame_q: mp.Queue,
                   result_q: mp.Queue, stop_event: mp.Event):
    """
    Дочерний процесс: берёт JPEG из frame_q, запускает FaceEngine,
    кладёт список результатов + оригинальный кадр в result_q.

    Каждый Worker загружает свою копию ONNX-модели — это нормально:
    buffalo_sc занимает ~300 МБ RAM, зато нет блокировок.
    На Xeon с 32+ ГБ RAM легко держать 4-8 воркеров.
    """
    log = _setup_logging(f"Worker-{worker_id}")

    # Привязываем к CPU-ядру (Linux only) для снижения cache miss
    try:
        cpu_count = os.cpu_count() or 4
        core = worker_id % cpu_count
        os.sched_setaffinity(0, {core})
        log.info("Привязан к ядру CPU %d", core)
    except AttributeError:
        pass  # Windows — sched_setaffinity не поддерживается

    from face_engine import FaceEngine
    engine = FaceEngine()
    log.info("FaceEngine готов (worker %d)", worker_id)

    fps_n = 0
    fps_t = time.time()
    last_fps_log = time.time()

    while not stop_event.is_set():
        try:
            jpeg_bytes = frame_q.get(timeout=0.5)
        except Exception:
            continue

        # Декодируем JPEG обратно в BGR
        buf = memoryview(jpeg_bytes) if not isinstance(jpeg_bytes, (bytes, bytearray)) else jpeg_bytes
        arr = cv2.imdecode(
            __import__("numpy").frombuffer(buf, dtype="uint8"),
            cv2.IMREAD_COLOR,
        )
        if arr is None:
            continue

        results = engine.process(arr)

        fps_n += 1
        now = time.time()
        if now - last_fps_log >= 10.0:
            fps = fps_n / max(now - fps_t, 0.001)
            log.info("Worker-%d: %.1f FPS | %d лиц в базе | %d лиц в кадре",
                     worker_id, fps, engine.known_count, len(results))
            fps_n = 0; fps_t = now; last_fps_log = now

        if not results:
            # Отправляем пустой результат чтобы NotifyProcess сбросил состояние
            try:
                result_q.put_nowait(([], jpeg_bytes))
            except Exception:
                pass
            continue

        try:
            result_q.put_nowait((
                [(r.name, r.confidence, r.is_live, r.bbox,
                  r.held_seconds, r.door_ready)
                 for r in results],
                jpeg_bytes,
            ))
        except Exception:
            pass

    log.info("Worker-%d завершён", worker_id)


# =============================================================
#  Процесс 3: NotifyProcess — Telegram + дверь
# =============================================================

def notify_process(result_q: mp.Queue, stop_event: mp.Event):
    """
    Дочерний процесс: получает результаты от воркеров,
    отправляет уведомления в Telegram и открывает дверь.

    Здесь один процесс — I/O-bound (HTTP к Telegram, HTTP к домофону),
    поэтому threading внутри достаточно.
    """
    log = _setup_logging("Notifier")
    import threading
    import numpy as np
    import telegram_bot
    from door_controller import DoorController

    door              = DoorController()
    door_just_opened: set = set()
    tg_notified:      set = set()

    _init_csv_log()
    log.info("NotifyProcess запущен")

    def _do_open_door(name, conf, is_live, jpeg_bytes):
        ok     = door.open()
        action = "ДОСТУП РАЗРЕШЁН" if ok else "ОШИБКА РЕЛЕ"
        _write_csv_log(name, conf, is_live, action)

        if ok:
            log.info("✅ %s — %s", name, action)
            if config.TG_NOTIFY_DOOR_OPEN:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, dtype="uint8"),
                    cv2.IMREAD_COLOR,
                )
                telegram_bot.notify_door_opened(frame, name, conf)
        else:
            log.error("❌ %s — %s", name, action)

        time.sleep(config.DOOR_OPEN_SECONDS + 1)
        door_just_opened.discard(name)

    while not stop_event.is_set():
        try:
            results_raw, jpeg_bytes = result_q.get(timeout=0.5)
        except Exception:
            continue

        if not results_raw:
            tg_notified.clear()
            door_just_opened.clear()
            continue

        # Реконструируем FaceResult-подобные объекты из tuple
        class _R:
            __slots__ = ("name","confidence","is_live","bbox","held_seconds","door_ready")
        results = []
        for t in results_raw:
            r = _R()
            r.name, r.confidence, r.is_live, r.bbox, r.held_seconds, r.door_ready = t
            results.append(r)

        best = max(results, key=lambda r: r.confidence)

        # ── Telegram: уведомление о посетителе ───────────────
        if config.TG_NOTIFY_VISITOR and best.is_live:
            key = best.name
            if key not in tg_notified:
                tg_notified.add(key)
                frame = cv2.imdecode(
                    __import__("numpy").frombuffer(jpeg_bytes, dtype="uint8"),
                    cv2.IMREAD_COLOR,
                )
                if best.name == "Unknown" and config.TG_NOTIFY_UNKNOWN:
                    log.info("TG: неизвестный у двери")
                    telegram_bot.notify_visitor(frame, "Unknown", best.confidence)
                elif best.name != "Unknown":
                    log.info("TG: %s у двери (%.0f%%)", best.name, best.confidence * 100)
                    telegram_bot.notify_visitor(frame, best.name, best.confidence)

        # ── Открытие двери ────────────────────────────────────
        if best.door_ready and best.name not in door_just_opened:
            door_just_opened.add(best.name)
            log.info("🚪 Открываю дверь для [%s] (уверенность %.0f%%)",
                     best.name, best.confidence * 100)
            threading.Thread(
                target=_do_open_door,
                args=(best.name, best.confidence, best.is_live, jpeg_bytes),
                daemon=True,
            ).start()

        # Сброс для ушедших
        active = {r.name for r in results}
        door_just_opened &= active
        tg_notified      &= active

    log.info("NotifyProcess завершён")


# =============================================================
#  HeadlessRunner — главный процесс-координатор
# =============================================================

class HeadlessRunner:

    def __init__(self):
        # Очереди между процессами
        self._frame_q  = mp.Queue(maxsize=FRAME_Q_MAXSIZE)
        self._result_q = mp.Queue(maxsize=RESULT_Q_MAXSIZE)
        self._stop     = mp.Event()

        self._processes: list[mp.Process] = []

    def run(self):
        log = logging.getLogger("Main")
        log.info("=" * 60)
        log.info("  HEADLESS (multiprocessing) — воркеров: %d", WORKER_PROCESSES)
        log.info("  RTSP: %s", config.RTSP_URL)
        log.info("  Нажмите Ctrl+C для остановки")
        log.info("=" * 60)

        # Запуск GrabProcess
        p_grab = mp.Process(
            target=grab_process,
            args=(self._frame_q, self._stop),
            name="GrabProcess",
            daemon=True,
        )
        p_grab.start()
        self._processes.append(p_grab)

        # Запуск N WorkerProcess
        for i in range(WORKER_PROCESSES):
            p_w = mp.Process(
                target=worker_process,
                args=(i, self._frame_q, self._result_q, self._stop),
                name=f"Worker-{i}",
                daemon=True,
            )
            p_w.start()
            self._processes.append(p_w)

        # Запуск NotifyProcess
        p_notify = mp.Process(
            target=notify_process,
            args=(self._result_q, self._stop),
            name="NotifyProcess",
            daemon=True,
        )
        p_notify.start()
        self._processes.append(p_notify)

        try:
            while not self._stop.is_set():
                time.sleep(0.5)
                # Проверяем, что все процессы живы
                for p in self._processes:
                    if not p.is_alive() and not self._stop.is_set():
                        log.error("Процесс %s упал! Завершение.", p.name)
                        self._stop.set()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        log = logging.getLogger("Main")
        log.info("Остановка всех процессов...")
        self._stop.set()
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                log.warning("Принудительное завершение %s", p.name)
                p.terminate()
        log.info("Завершено.")


# =============================================================
#  Точка входа
# =============================================================

def main():
    # Обязательно для multiprocessing на Windows/macOS
    mp.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("system.log", encoding="utf-8"),
        ],
    )

    runner = HeadlessRunner()

    def _sig(sig, frame):
        logging.getLogger("Main").info("Сигнал %s → завершение", sig)
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sig)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig)

    runner.run()


if __name__ == "__main__":
    main()
