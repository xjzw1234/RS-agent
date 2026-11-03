#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if python -c "import gradio, openai, dotenv, PIL" >/dev/null 2>&1; then
    echo "已检测到完整依赖，直接使用当前 Python 环境。"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install -r requirements.txt
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "已生成 .env，请填入 MOONSHOT_API_KEY 后重跑。"
    exit 0
fi

echo "启动 RS-Talker ... 浏览器访问 http://127.0.0.1:7860"
python app.py
