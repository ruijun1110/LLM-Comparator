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
import tiktoken  # Import tiktoken for accurate token counting

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
    def count_tokens(text, model="gpt-3.5-turbo"):
        """Count tokens using the tiktoken library for accurate results"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {str(e)}")
            # Fallback to approximate counting (4 chars per token)
            return max(1, len(text) // 4)

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
                        # For token calculation, we'll use tiktoken for OpenAI API calls
                        token_usage = None
                        if "openai.com" in url:
                            # Calculate prompt tokens accurately using tiktoken
                            prompt_tokens = Utils.count_tokens(prompt, model)
                            token_usage = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": 0,  # Will be updated during streaming
                                "total_tokens": prompt_tokens  # Will be updated during streaming
                            }
                        
                        # Now do the actual streaming call
                        full_response = ""  # Track the complete response for accurate token counting
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            print(f"DEBUG: response: {response}")
                            
                            if response.status_code != 200:
                                # Handle non-200 responses
                                error_text = await response.text()
                                try:
                                    error_json = json.loads(error_text)
                                    error_message = error_json.get("error", {}).get("message", f"API Error: {response.status_code}")
                                except:
                                    error_message = f"API Error: {response.status_code} - {error_text[:100]}"
                                q.put({
                                    "type": "error",
                                    "error": error_message
                                })
                                return
                                
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data.strip() == "[DONE]":
                                        break
                                    try:
                                        delta = json.loads(data)
                                        # OpenAI stream response: {"choices":[{"delta":{"content":"..."}}], ...}
                                        print(f"DEBUG: delta: {delta}")
                                        
                                        # Check for error in the response
                                        if "error" in delta:
                                            error_message = delta.get("error", {}).get("message", "Unknown error occurred")
                                            q.put({
                                                "type": "error",
                                                "error": error_message
                                            })
                                            break
                                        
                                        content = delta.get("choices", [{}])[0].get("delta", {}).get("content", None)
                                        if content is None:
                                            content = delta.get("delta", {}).get("text", "")
                                        
                                        # Keep track of the full response for accurate token counting
                                        if content:
                                            full_response += content
                                        
                                        # Update token usage with tiktoken if we're using OpenAI
                                        if "openai.com" in url and token_usage and content:
                                            # Update completion tokens by counting the whole response so far
                                            completion_tokens = Utils.count_tokens(full_response, model)
                                            token_usage["completion_tokens"] = completion_tokens
                                            token_usage["total_tokens"] = token_usage["prompt_tokens"] + completion_tokens
                                        
                                        finish_reason = delta.get("choices", [{}])[0].get("finish_reason", "")
                                        usage = delta.get("usage", {})
                                        
                                        # If we have our tiktoken-based token usage for OpenAI, use that
                                        if not usage and token_usage and "openai.com" in url:
                                            usage = token_usage
                                            
                                        q.put({
                                            "type": "content",
                                            "content": content,
                                            "usage": usage,
                                            "finish_reason": finish_reason,
                                            "model": model
                                        })
                                    except Exception as e:
                                        print(f"Error processing chunk: {str(e)}")
                                        continue
                                        
                    # After streaming is done, send done event with final token usage
                    end_time = time.time()
                    done_item = {"type": "done", "time": end_time - start_time}
                    
                    # For OpenAI, use our final tiktoken-based count
                    if token_usage and "openai.com" in url:
                        # Do final accurate count on the complete response
                        if full_response:
                            completion_tokens = Utils.count_tokens(full_response, model)
                            token_usage["completion_tokens"] = completion_tokens
                            token_usage["total_tokens"] = token_usage["prompt_tokens"] + completion_tokens
                        
                        done_item["usage"] = token_usage
                    
                    q.put(done_item)
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