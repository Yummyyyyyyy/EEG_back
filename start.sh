#!/bin/bash
# FastAPI 启动脚本

echo "正在启动 EEG Data Processing API..."
echo "========================================"

# 激活虚拟环境（如果有）
# source venv/bin/activate

# 启动服务器
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 或者使用：
# python -m app.main
