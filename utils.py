import pathlib
import streamlit as st
import httpx
import asyncio
import threading
import queue
import json
import time
from dataclasses import dataclass
from typing import Generator, Dict, Any

@dataclass
class StreamResponse:
    content: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_time: float = 0.0
    model: str = ""
    finish_reason: str = ""

class Utils:
    @staticmethod
    def load_css(file_path):
        with open(file_path) as f:
            st.html(f"<style>{f.read()}</style>")

    @staticmethod
    def stream_openai_response(url, prompt, api_key, model, temperature=0.5, max_tokens=1000) -> Generator[Dict[str, Any], None, None]:
        if "anthropic" in url:
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        start_time = time.time()
        def run_stream(q):
            async def fetch():
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            print(f"DEBUG: response: {response}")
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data.strip() == "[DONE]":
                                        break
                                    try:
                                        delta = json.loads(data)
                                        # OpenAI stream response: {"choices":[{"delta":{"content":"..."}}], ...}
                                        print(f"DEBUG: delta: {delta}")
                                        content = delta.get("choices", [{}])[0].get("delta", {}).get("content", None)
                                        if content is None:
                                            content = delta.get("delta", {}).get("text", "")
                                        finish_reason = delta.get("choices", [{}])[0].get("finish_reason", "")
                                        usage = delta.get("usage", {})
                                        q.put({
                                            "type": "content",
                                            "content": content,
                                            "usage": usage,
                                            "finish_reason": finish_reason,
                                            "model": model
                                        })
                                    except Exception:
                                        continue
                    # After streaming is done, send done event
                    end_time = time.time()
                    q.put({"type": "done", "time": end_time - start_time})
                except Exception as e:
                    q.put({"type": "error", "error": str(e)})
            asyncio.run(fetch())
        q = queue.Queue()
        threading.Thread(target=run_stream, args=(q,), daemon=True).start()
        while True:
            item = q.get()
            if item.get("type") == "done":
                yield item
                break
            yield item 