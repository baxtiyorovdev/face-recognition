# =============================================================
#  config.py — Все настройки системы
# =============================================================

# ── Dahua VTO 2111D-P-S3 ─────────────────────────────────────
DAHUA_IP       = "192.168.100.23"
DAHUA_PORT     = 80
DAHUA_USER     = "admin"
DAHUA_PASSWORD = "admin777"
DAHUA_CHANNEL  = 1

# RTSP-поток (основной / субпоток)
RTSP_MAIN = (
    f"rtsp://{DAHUA_USER}:{DAHUA_PASSWORD}"
    f"@{DAHUA_IP}:554/cam/realmonitor"
    f"?channel={DAHUA_CHANNEL}&subtype=0"
)
RTSP_SUB = (
    f"rtsp://{DAHUA_USER}:{DAHUA_PASSWORD}"
    f"@{DAHUA_IP}:554/cam/realmonitor"
    f"?channel={DAHUA_CHANNEL}&subtype=1"
)
RTSP_URL = RTSP_MAIN

# ── Распознавание ─────────────────────────────────────────────
KNOWN_FACES_DIR       = "known_faces"
FACE_CONFIDENCE_THRES = 0.70
FACE_MODEL            = "hog"

# ── Антиспуфинг ───────────────────────────────────────────────
LIVENESS_LAPLACIAN_THRES = 80.0
LIVENESS_EAR_THRES       = 0.25
LIVENESS_EAR_FRAMES      = 3
LIVENESS_MIN_BLINKS      = 1

# ── Логика доступа ────────────────────────────────────────────
HOLD_SECONDS      = 2   # уменьшено с 3 до 1.5 сек
DOOR_OPEN_SECONDS = 5

# ── Детектор движения (гейт для распознавания) ────────────────
# Пока перед камерой нет движения — FaceEngine не запускается (экономия CPU).
# При обнаружении движения распознавание включается на MOTION_ACTIVE_SECONDS.
# Каждое новое движение продлевает этот таймер.
MOTION_DETECTION       = True    # False → распознавание работает всегда (старое поведение)
MOTION_MIN_AREA        = 800     # порог площади изменений (px на кадре ~320px шириной)
MOTION_THRESHOLD       = 25      # порог разницы яркости пикселей (0-255)
MOTION_ACTIVE_SECONDS  = 300     # 5 минут работы распознавания после движения

# ── Веб-интерфейс (web_app.py) ────────────────────────────────
WEB_HOST              = "0.0.0.0"   # 0.0.0.0 → доступ из локальной сети
WEB_PORT              = 8000
VISITORS_DIR          = "visitors"  # папка с фото и журналом посетителей
VISITOR_DEDUP_SECONDS = 300         # 5 мин: повторные приходы в окне → одно фото
VISITOR_MAX_EVENTS    = 500         # сколько последних событий хранить
WEB_JPEG_QUALITY      = 75          # качество JPEG для видеопотока в браузере

# ── Производительность ────────────────────────────────────────
PROCESS_EVERY    = 1
FRAME_SKIP_GRAB  = 2      # grab() для очистки буфера (было 3)

# ── Логирование ───────────────────────────────────────────────
LOG_FILE = "access_log.csv"

# ── GUI ───────────────────────────────────────────────────────
GUI_TITLE      = "Dahua VTO · Контроль доступа по лицу"
GUI_WIDTH      = 1600
GUI_HEIGHT     = 980
CAMERA_W       = 1280
CAMERA_H       = 720
GUI_FPS        = 30

# ── Telegram Bot ──────────────────────────────────────────────
TG_BOT_TOKEN        = "8178511649:AAE4C_Z7oyyNA9hXOMQ1KZ6Nkk5VcpJvyrU"     # токен от @BotFather
TG_CHAT_ID          = "7605829941"     # ваш chat_id или id группы
TG_COOLDOWN_SECONDS = 5    # антиспам: сек между одинаковыми уведомлениями

TG_NOTIFY_VISITOR   = True   # кто-то подошёл
TG_NOTIFY_UNKNOWN   = True   # неизвестный
TG_NOTIFY_DOOR_OPEN = True   # дверь открыта

# ── Multiprocessing (headless.py) ─────────────────────────────
# Количество воркеров распознавания лиц.
# Рекомендации для Xeon:
#   4-ядерный  → 2
#   8-ядерный  → 4
#   16-ядерный → 6  (оставьте 2 ядра для GrabProcess/NotifyProcess/ОС)
#   32-ядерный → 12
# Каждый воркер загружает buffalo_sc (~300 МБ RAM).
import os as _os
WORKER_PROCESSES = max(6, (_os.cpu_count() or 6) // 6)
