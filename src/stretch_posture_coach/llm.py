"""Minimal Ollama client (HTTP API) for short spoken coaching messages."""

from __future__ import annotations

import json
import re

import requests

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2:1.5b",
        temperature: float = 0.6,
        max_tokens: int = 70,
        timeout_s: float = 20,
        keep_alive: str = "30m",
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive

    @classmethod
    def from_config(cls, c: dict) -> OllamaClient:
        return cls(
            c.get("host", "http://localhost:11434"),
            c.get("model", "qwen2:1.5b"),
            float(c.get("temperature", 0.6)),
            int(c.get("max_tokens", 70)),
            float(c.get("timeout_s", 20)),
            str(c.get("keep_alive", "30m")),
        )

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"Ollama not reachable at {self.host}: {e}") from e
        return [m.get("name", "") for m in r.json().get("models", [])]

    def has_model(self) -> bool:
        names = self.list_models()
        want = self.model if ":" in self.model else f"{self.model}:latest"
        return any(n == want or n == self.model for n in names)

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": max_tokens or self.max_tokens},
        }
        try:
            r = requests.post(f"{self.host}/api/chat", data=json.dumps(payload), timeout=self.timeout_s)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(str(e)) from e
        return r.json().get("message", {}).get("content", "")

    def warmup(self) -> None:
        """Load the model into memory so the first coaching message is fast."""
        self.chat([{"role": "user", "content": "Say OK."}], max_tokens=2)


def clean_spoken(text: str, max_chars: int = 240, allow_cjk: bool = False) -> str:
    """Make LLM output safe to speak: strip markdown/emoji/quotes, keep at most two sentences."""
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"[*_#`>\[\]]", "", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"^\s*(stretch|assistant|robot)\s*:\s*", "", t, flags=re.I)
    t = t.strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    if not allow_cjk and _CJK.search(t):
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", t)
    t = " ".join(sentences[:2]).strip()
    if len(t) > max_chars:
        cut = t[:max_chars]
        t = cut[: cut.rfind(" ")].rstrip(",;:") + "." if " " in cut else cut
    return t
