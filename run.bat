@echo off
chcp 65001 >nul
cd /d "%~dp0"

python -c "import gradio, openai, dotenv, PIL" >nul 2>&1
if %errorlevel%==0 (
    echo 已检测到完整依赖，直接使用当前 Python 环境。
) else (
    if not exist ".venv" (
        echo [1/2] 创建虚拟环境...
        python -m venv .venv || goto :fail
    )
    call ".venv\Scripts\activate.bat"
    echo [2/2] 安装依赖...
    python -m pip install -U pip
    pip install -r requirements.txt || goto :fail
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo 已生成 .env，请填入 MOONSHOT_API_KEY，保存后重新运行本脚本。
    notepad ".env"
    pause
    exit /b 0
)

echo 启动 RS-Talker ... 浏览器访问 http://127.0.0.1:7860
python app.py
goto :eof

:fail
echo 安装失败，请检查上面的报错。
pause
