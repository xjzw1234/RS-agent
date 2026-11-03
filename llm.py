"""OpenAI 兼容的视觉模型客户端。L1 阶段只做图生文，不碰 GIS。"""

import base64
import io
import mimetypes
import os
from dataclasses import dataclass

from openai import OpenAI

PRESETS = {
    "kimi": {
        "label": "Kimi（Moonshot）",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.6",
        "key_env": "MOONSHOT_API_KEY",
    },
    "kimi-intl": {
        "label": "Kimi 国际站",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2.6",
        "key_env": "MOONSHOT_API_KEY",
    },
    "qwen": {
        "label": "Qwen-VL（阿里百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "siliconflow": {
        "label": "DeepSeek-VL2（硅基流动托管）",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "deepseek-ai/deepseek-vl2",
        "key_env": "SILICONFLOW_API_KEY",
    },
    "openai": {
        "label": "OpenAI GPT-4o",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "label": "本地 Ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5vl:7b",
        "key_env": "OLLAMA_API_KEY",
    },
}

DEFAULT_PROVIDER = "kimi"


@dataclass
class Settings:
    provider: str
    label: str
    base_url: str
    model: str
    api_key: str

    @property
    def ready(self) -> bool:
        return self.provider == "ollama" or bool(self.api_key)


def load_settings() -> Settings:
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    preset = PRESETS.get(provider, PRESETS[DEFAULT_PROVIDER])
    return Settings(
        provider=provider,
        label=preset["label"],
        base_url=os.getenv("LLM_BASE_URL") or preset["base_url"],
        model=os.getenv("LLM_MODEL") or preset["model"],
        api_key=os.getenv("LLM_API_KEY") or os.getenv(preset["key_env"], ""),
    )


def make_client(settings: Settings) -> OpenAI:
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key or "not-needed",
        timeout=120.0,
    )


def encode_image(path: str, max_edge: int | None = None) -> dict:
    """图片 → 等比缩放到 max_edge → JPEG base64 的 image_url 片段。

    整景遥感图动辄上万像素，原图 base64 会撞上接口的体积上限；
    视觉模型内部本来就要降采样，先缩到 1536 长边几乎不损失判读信息。
    """
    max_edge = max_edge or int(os.getenv("IMAGE_MAX_EDGE", "1536"))
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        raw = f.read()
    data, mime = _shrink(raw, mime, max_edge)
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _shrink(raw: bytes, mime: str, max_edge: int) -> tuple[bytes, str]:
    try:
        from PIL import Image
    except ImportError:
        return raw, mime
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return raw, mime

    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    out = buf.getvalue()
    return (out, "image/jpeg") if out else (raw, mime)
