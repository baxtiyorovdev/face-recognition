#!/bin/bash
# ═══════════════════════════════════════════════════
#  start_headless.sh — Фоновый запуск (Linux/macOS)
# ═══════════════════════════════════════════════════
cd "$(dirname "$0")"

nohup python3 headless.py >> system.log 2>&1 &
PID=$!
echo $PID > headless.pid
echo "Запущено в фоне (PID=$PID). Лог: system.log"
echo "Остановить: bash stop_headless.sh"
