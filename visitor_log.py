# =============================================================
#  visitor_log.py — Журнал посетителей с фото и дедупликацией
# =============================================================
"""
Хранит события «кто пришёл и когда» + фото.

Дедупликация:
  Если один и тот же человек появляется много раз, но интервал между
  появлениями меньше VISITOR_DEDUP_SECONDS (по умолчанию 5 минут) —
  сохраняется ТОЛЬКО ОДНО фото/событие на всю серию появлений.

  Реализация — «скользящее окно»: каждое появление лица обновляет
  отметку времени `last_seen[name]`. Новое событие создаётся только
  если с прошлого появления прошло >= dedup_seconds (т.е. человек
  уходил минимум на 5 минут и вернулся).

Хранилище:
  visitors/photos/<id>.jpg   — фото кадра
  visitors/events.json       — список событий (новые сверху)
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import config

log = logging.getLogger("VisitorLog")


class VisitorLog:

    def __init__(self, base_dir: str | None = None,
                 dedup_seconds: float | None = None,
                 max_events: int | None = None):
        self.base = Path(base_dir or getattr(config, "VISITORS_DIR", "visitors"))
        self.photos = self.base / "photos"
        self.events_file = self.base / "events.json"
        self.dedup_seconds = (dedup_seconds if dedup_seconds is not None
                              else getattr(config, "VISITOR_DEDUP_SECONDS", 300))
        self.max_events = (max_events if max_events is not None
                           else getattr(config, "VISITOR_MAX_EVENTS", 500))

        self.photos.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}   # name → timestamp последнего появления
        self._events: list[dict] = self._load()

    # ── Загрузка / сохранение ────────────────────────────────────

    def _load(self) -> list[dict]:
        if self.events_file.exists():
            try:
                with open(self.events_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                log.warning("Не удалось прочитать журнал: %s", exc)
        return []

    def _save(self):
        tmp = self.events_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._events, f, ensure_ascii=False, indent=1)
        tmp.replace(self.events_file)

    # ── Запись события ───────────────────────────────────────────

    def record(self, name: str, confidence: float, is_live: bool,
               frame_bgr) -> dict | None:
        """
        Регистрирует появление человека.
        Возвращает dict нового события, либо None если сработала
        дедупликация (повтор в окне < dedup_seconds).
        """
        now = time.time()
        with self._lock:
            last = self._last_seen.get(name, 0.0)
            self._last_seen[name] = now          # скользящее окно
            if last and (now - last) < self.dedup_seconds:
                return None                       # тот же человек, < 5 мин → пропуск

        # ── Сохраняем фото ───────────────────────────────────────
        ts = datetime.now()
        safe_name = "".join(c if c.isalnum() else "_" for c in name) or "Unknown"
        event_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        photo_name = f"{event_id}.jpg"
        try:
            cv2.imwrite(str(self.photos / photo_name), frame_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as exc:
            log.error("Не удалось сохранить фото: %s", exc)
            photo_name = None

        event = {
            "id":         event_id,
            "name":       name,
            "known":      name != "Unknown",
            "confidence": round(float(confidence), 4),
            "is_live":    bool(is_live),
            "timestamp":  ts.isoformat(timespec="seconds"),
            "photo":      photo_name,
        }

        with self._lock:
            self._events.insert(0, event)
            # Обрезаем старые события и их фото
            if len(self._events) > self.max_events:
                for old in self._events[self.max_events:]:
                    self._delete_photo(old.get("photo"))
                self._events = self._events[: self.max_events]
            self._save()

        log.info("Новый посетитель: %s (%.0f%%)", name, confidence * 100)
        return event

    def _delete_photo(self, photo_name):
        if not photo_name:
            return
        try:
            (self.photos / photo_name).unlink(missing_ok=True)
        except Exception:
            pass

    # ── Чтение / управление ──────────────────────────────────────

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return list(self._events[:limit])

    def clear(self):
        with self._lock:
            for ev in self._events:
                self._delete_photo(ev.get("photo"))
            self._events = []
            self._last_seen.clear()
            self._save()

    def photo_path(self, photo_name: str) -> Path:
        return self.photos / photo_name
