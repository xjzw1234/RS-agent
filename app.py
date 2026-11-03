import os

import gradio as gr
from dotenv import load_dotenv

from llm import encode_image, load_settings, make_client
from prompts import DEFAULT_PRESET, PROMPT_PRESETS, build_system_prompt

load_dotenv()

KEEP_IMAGE_TURNS = 2


def _parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, dict):
        content = [content]
    parts: list[dict] = []
    for item in content or []:
        if isinstance(item, str):
            if os.path.isfile(item):
                parts.append(encode_image(item))
            elif item.strip():
                parts.append({"type": "text", "text": item})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            parts.append(encode_image(item[0]))
        elif isinstance(item, dict):
            text = item.get("text")
            files = item.get("files") or []
            if isinstance(text, str) and text.strip():
                parts.append({"type": "text", "text": text})
            for f in files:
                if isinstance(f, str) and os.path.isfile(f):
                    parts.append(encode_image(f))
            if not files:
                path = item.get("path") or item.get("name")
                if isinstance(path, str) and os.path.isfile(path):
                    parts.append(encode_image(path))
    return parts


def _merge(parts: list[dict]):
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def build_messages(history, message, system_prompt: str) -> list[dict]:
    turns: list[dict] = []
    for turn in history or []:
        if isinstance(turn, dict):
            role, content = turn.get("role", "user"), turn.get("content")
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            role, content = ("user" if turn[0] else "assistant"), turn[1]
        else:
            continue
        parts = _parts(content)
        if parts:
            turns.append({"role": role, "content": _merge(parts)})

    # 每轮回传整景 base64 会迅速耗尽上下文与额度，只留最近几轮的图像。
    seen = 0
    for msg in reversed(turns):
        if msg["role"] != "user" or not isinstance(msg["content"], list):
            continue
        seen += 1
        if seen > KEEP_IMAGE_TURNS:
            kept = [p for p in msg["content"] if p["type"] == "text"]
            msg["content"] = kept or "（早先的影像已省略，如需追问请重新上传）"

    current = _parts(message)
    if current:
        turns.append({"role": "user", "content": _merge(current)})

    return [{"role": "system", "content": system_prompt}] + turns


def chat(message, history, preset):
    settings = load_settings()
    if not settings.ready:
        yield (
            f"⚠️ 还没有配置 API Key。\n\n"
            f"当前预设：**{settings.label}** · `{settings.model}`\n\n"
            f"把 `.env.example` 复制为 `.env`，填入 `MOONSHOT_API_KEY=...`，然后重启应用。"
        )
        return

    messages = build_messages(history, message, build_system_prompt(preset))

    # k2.6 / k2.7 这类推理模型只接受 temperature=1，传别的值会直接 400，
    # 所以默认不传，交给服务端决定；确实需要时才在 .env 里设 LLM_TEMPERATURE。
    kwargs = {}
    if os.getenv("LLM_TEMPERATURE"):
        kwargs["temperature"] = float(os.environ["LLM_TEMPERATURE"])

    try:
        client = make_client(settings)
        stream = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        answer = ""
        thinking = False
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 推理模型先吐一大段 reasoning_content，token 花在这上面时 content 一直为空，
            # 界面会看起来像卡死，所以给个占位提示。
            if not answer and getattr(delta, "reasoning_content", None):
                if not thinking:
                    thinking = True
                    yield "🧠 思考中…"
                continue
            piece = delta.content
            if piece:
                answer += piece
                yield answer
        if not answer:
            yield "（模型没有返回内容。可能是图像未被识别，或该模型不支持图片输入。）"
    except Exception as exc:
        yield (
            f"❌ 调用失败：`{type(exc).__name__}: {exc}`\n\n"
            f"排查顺序：1) `{settings.model}` 是否支持图片输入；"
            f"2) `{settings.base_url}` 是否为 OpenAI 兼容端点；3) 账号额度是否充足。"
        )


def build_ui():
    settings = load_settings()
    if settings.ready:
        status = f"当前模型：**{settings.label}** · `{settings.model}`"
    else:
        status = "⚠️ 未检测到 API Key，请先配置 `.env`"

    return gr.ChatInterface(
        fn=chat,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_types=["image"],
            file_count="multiple",
            placeholder="上传一张影像，然后提问：图里有什么？",
            show_label=False,
        ),
        chatbot=gr.Chatbot(height=560, show_label=False),
        additional_inputs=[
            gr.Dropdown(
                choices=list(PROMPT_PRESETS),
                value=DEFAULT_PRESET,
                label="Prompt 模式",
            ),
        ],
        additional_inputs_accordion=gr.Accordion("Prompt 模板", open=False),
        title="🛰️ RS-Talker · 遥感影像对话 Agent（L1）",
        description=f"{status}\n\n上传影像 → 直接问“图里有什么”。只做图生文，不做 GIS 计算。",
    )


if __name__ == "__main__":
    build_ui().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
