#!/bin/sh
set -e

# 创建持久化缓存目录（volume 挂载后执行）
mkdir -p /app/data/.cache
rm -rf /root/.cache
ln -sf /app/data/.cache /root/.cache

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
