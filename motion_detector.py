# =============================================================
#  motion_detector.py — Лёгкий детектор движения (frame diff)
# =============================================================
"""
Гейт для распознавания лиц.

Идея:
  Распознавание лиц (InsightFace) дорогое (~15-30 мс/кадр).
  Детектор движения дешёвый (~1 мс/кадр) — сравнение соседних кадров.

  Пока перед камерой нет движения — FaceEngine НЕ запускается,
  CPU простаивает. Как только обнаружено движение — взводим таймер
  на MOTION_ACTIVE_SECONDS (по умолчанию 5 минут). Пока таймер активен,
  распознавание работает. Каждое новое движение продлевает таймер.

  Когда движение прекратилось и таймер истёк — снова уходим в простой.

Алгоритм:
  1. Кадр → grayscale → уменьшение до ~320px → размытие.
  2. Абсолютная разница с предыдущим кадром.
  3. Порог + дилатация → бинарная маска изменений.
  4. Если площадь изменений >= MOTION_MIN_AREA — есть движение.
"""

import time

import cv2

import config


class MotionDetector:
    """Детектор движения с «удержанием» активного состояния по таймеру."""

    # Ширина, до которой уменьшаем кадр перед анализом (скорость).
    _PROC_WIDTH = 320

    def __init__(self, min_area: int | None = None,
                 threshold: int | None = None,
                 active_seconds: float | None = None):
        self.min_area = (min_area if min_area is not None
                         else getattr(config, "MOTION_MIN_AREA", 800))
        self.threshold = (threshold if threshold is not None
                          else getattr(config, "MOTION_THRESHOLD", 25))
        self.active_seconds = (active_seconds if active_seconds is not None
                               else getattr(config, "MOTION_ACTIVE_SECONDS", 300))

        self._prev_gray = None        # предыдущий обработанный кадр
        self._active_until = 0.0      # timestamp, до которого распознавание включено
        self._last_motion_ts = 0.0    # момент последнего движения

    # ── Внутренняя детекция движения ─────────────────────────────

    def _detect(self, bgr) -> bool:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        if w > self._PROC_WIDTH:
            scale = self._PROC_WIDTH / w
            gray = cv2.resize(gray, (self._PROC_WIDTH, int(h * scale)),
                              interpolation=cv2.INTER_AREA)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        thresh = cv2.threshold(delta, self.threshold, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_pixels = cv2.countNonZero(thresh)
        return motion_pixels >= self.min_area

    # ── Публичный API ────────────────────────────────────────────

    def update(self, bgr) -> bool:
        """
        Обрабатывает кадр и возвращает True, если распознавание лиц
        должно сейчас работать (движение есть или таймер ещё не истёк).
        """
        now = time.time()
        if self._detect(bgr):
            self._last_motion_ts = now
            self._active_until = now + self.active_seconds
        return now < self._active_until

    @property
    def active(self) -> bool:
        """True, пока распознавание должно работать (таймер не истёк)."""
        return time.time() < self._active_until

    @property
    def seconds_left(self) -> float:
        """Сколько секунд ещё будет активно распознавание (0, если простой)."""
        return max(0.0, self._active_until - time.time())

    def reset(self):
        """Сбрасывает состояние (например, при переподключении камеры)."""
        self._prev_gray = None
        self._active_until = 0.0
        self._last_motion_ts = 0.0
