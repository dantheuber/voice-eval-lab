"""Streaming OpenAI-compatible chat client that records time to first token and collects tool calls."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

SENTENCE_END = re.compile(r"(.+?[.!?])(\s+|$)", re.S)
THINK = re.compile(r"<think>.*?</think>\s*", re.S)


@dataclass
class ChatResult:
    ttft_ms: float | None = None           # first content or tool-call delta
    first_content_ms: float | None = None  # first spoken content
    first_sentence_ms: float | None = None
    total_ms: float = 0.0
    content: str = ""
    reasoning: str = ""
    finish: str | None = None
    tool_calls: list = field(default_factory=list)
    malformed: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property
    def tps(self) -> float | None:
        return self.timings.get("predicted_per_second")

    @property
    def prompt_tps(self) -> float | None:
        return self.timings.get("prompt_per_second")


def chat(url: str, messages: list[dict], tools: list | None = None, temperature: float = 0.2, max_tokens: int = 256,
         extra: dict | None = None, on_sentence=None, on_delta=None, client: httpx.Client | None = None,
         chat_path: str = "/v1/chat/completions") -> ChatResult:
    """Streams one completion. on_sentence(text, t_perf_counter) fires for each finished sentence of content."""
    body: dict = {"messages": messages, "stream": True, "temperature": temperature, "max_tokens": max_tokens,
                  "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    if extra:
        body.update(extra)
    res = ChatResult()
    pending = ""
    calls: dict[int, dict] = {}
    t0 = time.perf_counter()
    c = client or httpx.Client(timeout=300)
    with c.stream("POST", url + chat_path, json=body) as r:
        if r.status_code != 200:
            r.read()
            raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("timings"):
                res.timings = ev["timings"]
            if ev.get("usage"):
                res.usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {})
                now = time.perf_counter()
                if ch.get("finish_reason"):
                    res.finish = ch["finish_reason"]
                if delta.get("reasoning_content"):
                    res.reasoning += delta["reasoning_content"]
                    if res.ttft_ms is None:
                        res.ttft_ms = (now - t0) * 1000
                if delta.get("content"):
                    if res.ttft_ms is None:
                        res.ttft_ms = (now - t0) * 1000
                    if res.first_content_ms is None:
                        res.first_content_ms = (now - t0) * 1000
                    res.content += delta["content"]
                    pending += delta["content"]
                    if on_delta:
                        on_delta(delta["content"], now)
                    while True:
                        m = SENTENCE_END.match(pending)
                        # needs following text to be sure, unless the stream ends; a number like 4.99 is not a sentence end
                        if not m or m.end() >= len(pending) or re.search(r"\d\.$", m.group(1)):
                            break
                        if res.first_sentence_ms is None:
                            res.first_sentence_ms = (now - t0) * 1000
                        if on_sentence:
                            on_sentence(m.group(1).strip(), now)
                        pending = pending[m.end():]
                for tc in delta.get("tool_calls") or []:
                    if res.ttft_ms is None:
                        res.ttft_ms = (now - t0) * 1000
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
    now = time.perf_counter()
    res.total_ms = (now - t0) * 1000
    tail = pending.strip()
    if tail:
        if res.first_sentence_ms is None:
            res.first_sentence_ms = res.total_ms
        if on_sentence:
            on_sentence(tail, now)
    for idx in sorted(calls):
        call = calls[idx]
        try:
            call["args"] = json.loads(call["arguments"] or "{}")
            res.tool_calls.append(call)
        except json.JSONDecodeError as e:
            res.malformed.append({**call, "error": str(e)})
    return res
