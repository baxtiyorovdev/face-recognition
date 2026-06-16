# =============================================================
#  door_controller.py — Управление замком через Dahua HTTP API
# =============================================================

import logging
import threading
import time

import requests
from requests.auth import HTTPDigestAuth

import config

log = logging.getLogger("Door")


class DoorController:
    """
    Управляет электромагнитным замком Dahua VTO 2111D-P-S3
    через встроенный HTTP CGI-интерфейс.

    Endpoint открытия:
      GET /cgi-bin/accessControl.cgi?action=openDoor
          &channel=1&UserID=101&Type=Remote
    Аутентификация: HTTP Digest (стандарт Dahua).
    """

    def __init__(self, on_state_change=None):
        """
        on_state_change(is_open: bool) — колбэк для GUI.
        """
        self._auth    = HTTPDigestAuth(config.DAHUA_USER, config.DAHUA_PASSWORD)
        self._base    = f"http://{config.DAHUA_IP}:{config.DAHUA_PORT}"
        self._session = requests.Session()
        self._session.auth = self._auth
        self._open    = False
        self._cb      = on_state_change

    # ── Публичный API ─────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self, channel: int = None) -> bool:
        """Открыть дверь. Возвращает True при успехе."""
        ch = channel or config.DAHUA_CHANNEL
        url = (
            f"{self._base}/cgi-bin/accessControl.cgi"
            f"?action=openDoor&channel={ch}&UserID=101&Type=Remote"
        )
        try:
            r = self._session.get(url, timeout=5)
            ok = r.status_code == 200 and "OK" in r.text
        except requests.RequestException as exc:
            log.error("Сетевая ошибка: %s", exc)
            ok = False

        if ok:
            log.info("✅ Дверь открыта (канал %d)", ch)
            self._set_state(True)
            # авто-закрытие через N секунд
            threading.Thread(
                target=self._auto_close,
                args=(config.DOOR_OPEN_SECONDS,),
                daemon=True,
            ).start()
        else:
            log.error("❌ Не удалось открыть дверь")
        return ok

    def check_online(self) -> bool:
        """Проверяет доступность устройства."""
        url = f"{self._base}/cgi-bin/magicBox.cgi?action=getSystemInfo"
        try:
            r = self._session.get(url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ── Внутренние методы ─────────────────────────────────────

    def _auto_close(self, delay: int):
        time.sleep(delay)
        self._set_state(False)
        log.info("🔒 Дверь закрыта (авто, %d сек)", delay)

    def _set_state(self, is_open: bool):
        self._open = is_open
        if self._cb:
            self._cb(is_open)
