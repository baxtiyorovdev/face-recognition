# =============================================================
#  telegram_bot.py — Уведомления в Telegram
# =============================================================
"""
Отправляет фото + сообщение в Telegram когда:
  1. Обнаружено неизвестное лицо (посетитель пришёл)
  2. Распознанный человек — дверь открыта
  3. Ручное открытие двери

Настройки в config.py:
  TG_BOT_TOKEN = "123456:ABC..."
  TG_CHAT_ID   = "-100123456789"  # или личный chat_id
"""

import io
import logging
import threading
import time
from datetime import datetime

import cv2
import requests

import config

log = logging.getLogger("TelegramBot")

# Антиспам: не слать одно и то же лицо чаще раза в N секунд
_COOLDOWN = getattr(config, "TG_COOLDOWN_SECONDS", 30)
_last_sent: dict[str, float] = {}   # name → timestamp


def _can_send(key: str) -> bool:
    now = time.time()
    last = _last_sent.get(key, 0)
    if now - last >= _COOLDOWN:
        _last_sent[key] = now
        return True
    return False


def _frame_to_jpeg(bgr) -> bytes:
    """Конвертирует BGR numpy-кадр в JPEG байты."""
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _send(token: str, chat_id: str, text: str, jpeg_bytes: bytes | None = None):
    """Отправляет сообщение (с фото или без) в Telegram. Блокирующий вызов."""
    base = f"https://api.telegram.org/bot{token}"
    try:
        if jpeg_bytes:
            r = requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": chat_id, "caption": text, "parse_mode": "HTML"},
                files={"photo": ("face.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
                timeout=10,
            )
        else:
            r = requests.post(
                f"{base}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        if r.status_code != 200:
            log.warning("TG error %d: %s", r.status_code, r.text[:200])
        else:
            log.info("TG ✅ отправлено: %s", text[:60])
    except Exception as exc:
        log.error("TG send error: %s", exc)


def _async_send(token, chat_id, text, jpeg_bytes=None):
    """Отправка в фоновом потоке — не блокирует основной цикл."""
    threading.Thread(
        target=_send,
        args=(token, chat_id, text, jpeg_bytes),
        daemon=True,
    ).start()


# ── Публичные функции ─────────────────────────────────────────

def notify_visitor(frame, name: str, confidence: float):
    """
    Вызывается когда обнаружено лицо на камере.
    name == "Unknown" → неизвестный посетитель.
    name != "Unknown" → распознан (ещё не открыто).
    """
    token = getattr(config, "TG_BOT_TOKEN", "")
    chat  = getattr(config, "TG_CHAT_ID",   "")
    if not token or not chat:
        return

    key = f"visitor_{name}"
    if not _can_send(key):
        return

    ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    if name == "Unknown":
        text = (
            f"🚨 <b>Неизвестный посетитель</b>\n"
            f"🕐 {ts}\n"
            f"📷 Снято с камеры домофона"
        )
    else:
        text = (
            f"👤 <b>{name}</b> у двери\n"
            f"✅ Уверенность: {confidence:.0%}\n"
            f"🕐 {ts}"
        )

    jpeg = _frame_to_jpeg(frame)
    _async_send(token, chat, text, jpeg)


def notify_door_opened(frame, name: str, confidence: float, manual: bool = False):
    """
    Вызывается когда дверь открыта (авто или вручную).
    Отправляет фото + сообщение «дверь открыта».
    """
    token = getattr(config, "TG_BOT_TOKEN", "")
    chat  = getattr(config, "TG_CHAT_ID",   "")
    if not token or not chat:
        return

    key = f"door_{name}"
    if not _can_send(key):
        return

    ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    if manual:
        text = (
            f"🔓 <b>Дверь открыта вручную</b>\n"
            f"🕐 {ts}"
        )
    else:
        text = (
            f"🔓 <b>Дверь открыта</b>\n"
            f"👤 Пользователь: <b>{name}</b>\n"
            f"✅ Уверенность: {confidence:.0%}\n"
            f"🕐 {ts}"
        )

    jpeg = _frame_to_jpeg(frame) if frame is not None else None
    _async_send(token, chat, text, jpeg)


def notify_unknown_attempt(frame):
    """Неизвестный попытался войти — отдельное уведомление с фото."""
    token = getattr(config, "TG_BOT_TOKEN", "")
    chat  = getattr(config, "TG_CHAT_ID",   "")
    if not token or not chat:
        return

    key = "unknown_attempt"
    if not _can_send(key):
        return

    ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    text = (
        f"⚠️ <b>Попытка доступа — НЕИЗВЕСТНЫЙ</b>\n"
        f"🚫 Доступ запрещён\n"
        f"🕐 {ts}"
    )
    jpeg = _frame_to_jpeg(frame) if frame is not None else None
    _async_send(token, chat, text, jpeg)
