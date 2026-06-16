#!/bin/bash
# Остановить фоновый процесс
if [ -f headless.pid ]; then
    PID=$(cat headless.pid)
    kill "$PID" 2>/dev/null && echo "Остановлено (PID=$PID)" || echo "Процесс уже не запущен"
    rm -f headless.pid
else
    pkill -f headless.py && echo "Остановлено" || echo "Процесс не найден"
fi
