import pathlib
import streamlit as st
import httpx
import asyncio
import threading
import queue
import json

class Utils:
    @staticmethod
    def load_css(file_path):
        with open(file_path) as f:
            st.html(f"<style>{f.read()}</style>")

    @staticmethod
    def stream_openai_response(url, prompt, api_key, model, temperature=0.5, max_tokens=1000):
        print(f"DEBUG: url: {url}, api_key: {api_key}, model: {model}")
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async def fetch():
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    print(f"DEBUG: {response}")
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                delta = json.loads(data)
                                content = delta.get("choices", [{}])[0].get("delta", {}).get("content")
                                if content:
                                    yield content
                            except Exception:
                                continue
        q = queue.Queue()
        sentinel = object()
        def runner():
            async def consume():
                async for chunk in fetch():
                    q.put(chunk)
                q.put(sentinel)
            asyncio.run(consume())
        threading.Thread(target=runner, daemon=True).start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item 