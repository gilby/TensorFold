"""Chat prompts through DeepSeek-V4.1's own encoder: spaced DSML tool tags, a numeric reasoning effort, <｜System｜>."""

from __future__ import annotations

import copy
from typing import Any

from tensorfold.families.deepseek_v41.vendor.encoding_dsv41 import ASSISTANT_SP_TOKEN as ASSISTANT
from tensorfold.families.deepseek_v41.vendor.encoding_dsv41 import IMAGE_PLACEHOLDER, encode_messages

# The server's effort names onto V4.1's 1-100 budget: DeepSeek names low (50), high (75) and max (100)
EFFORTS = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}
# What the server's effort matching reads as this model's template: the levels DeepSeek's encoder names
EFFORT_NAMES = "DeepSeek-V4.1 encoder: reasoning effort 'low', 'high', 'max' or 1-100"


def effort_budget(effort: Any) -> str | int | None:
    """A reasoning effort as the encoder takes it: low / high / max, or an integer budget from 1 to 100."""

    if effort is None or str(effort).strip().lower() == "none":
        return None
    text = str(effort).strip().lower()
    if text in EFFORTS:
        return EFFORTS[text]
    if text.isdigit() and 1 <= int(text) <= 100:
        return int(text)
    raise ValueError(f"DeepSeek-V4.1 takes reasoning_effort low, medium, high, xhigh, max or 1-100, not {effort!r}")


def _images(message: dict[str, Any]) -> dict[str, Any]:
    """The server's prepared image parts as the encoder's image blocks (it writes their placeholders itself)."""

    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "image":
            parts.append({"type": "image", "data": "prepared"})       # the pixels are the vision frontend's
        elif kind in ("text", "image_url"):
            parts.append(part)
        else:
            raise ValueError(f"DeepSeek-V4.1 takes text and image content parts, not {kind!r}")
    return {**message, "content": parts}


def render(messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, thinking: bool = False,
           reasoning_effort: Any = None) -> str:
    """The prompt text for OpenAI-style messages; tools join the first system message (one is added if needed)."""

    msgs = [_images(copy.deepcopy(m)) for m in messages]
    for m in msgs:
        if m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""
    if tools:
        if not msgs or msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = [t if "function" in t else {"type": "function", "function": t} for t in tools]
    effort = effort_budget(reasoning_effort) if thinking else None
    return encode_messages(msgs, thinking_mode="thinking" if thinking else "chat", reasoning_effort=effort)


class DeepSeekV41Tokenizer:
    """The checkpoint's tokenizer with ``apply_chat_template`` rendered by DeepSeek-V4.1's encoder."""

    chat_template = EFFORT_NAMES        # the checkpoint ships no template; this names the encoder's effort levels

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def message_markers(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Replies start at <｜Assistant｜>, an added token the checkpoint leaves unmarked; no message opener."""

        return (), tuple(int(i) for i in self._inner.encode(ASSISTANT, add_special_tokens=False))

    def apply_chat_template(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                            add_generation_prompt: bool = True, tokenize: bool = True, enable_thinking: bool = False,
                            thinking_mode: str | None = None, reasoning_effort: Any = None, **_: Any) -> Any:
        """The encoder adds the reply prefix after a last user turn itself; without a generation prompt it goes."""

        thinking = thinking_mode == "thinking" if thinking_mode is not None else bool(enable_thinking)
        text = render(messages, tools=tools, thinking=thinking, reasoning_effort=reasoning_effort)
        prefix = ASSISTANT + ("<think>" if thinking else "</think>")
        last = messages[-1].get("role") if messages else None
        if add_generation_prompt and last == "assistant":
            text += prefix
        elif not add_generation_prompt and last != "assistant" and text.endswith(prefix):
            text = text[:-len(prefix)]                   # the history alone, so its reply starts a prompt chunk
        if not tokenize:
            return text
        return list(self._inner.encode(text, add_special_tokens=False))


__all__ = ["DeepSeekV41Tokenizer", "EFFORTS", "IMAGE_PLACEHOLDER", "effort_budget", "render"]
