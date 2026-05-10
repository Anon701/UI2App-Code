#!/usr/bin/env python3
"""
UI2App LLM client + cost tracking + attempts log.
"""

import json, os, re, time
import contextvars
from datetime import datetime
from openai import OpenAI

from .projects import ROOT, COST_LOG

import contextvars
# Pipeline run sets these so log_cost / log_attempt know which (project, model) the
# call belongs to without threading args through every callsite.
CURRENT_PROJECT = contextvars.ContextVar('CURRENT_PROJECT', default=None)
CURRENT_MODEL = contextvars.ContextVar('CURRENT_MODEL', default=None)
ATTEMPTS_LOG = ROOT / "data" / "runs" / "cost_log" / "attempts.jsonl"


def log_cost(stage, model, prompt_tokens, completion_tokens):
    entry = {
        "ts": datetime.now().isoformat(),
        "project": CURRENT_PROJECT.get(),
        "stage": stage, "model": model,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
    }
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def log_attempt(status, project=None, model=None, run_dir=None, error=None, **extra):
    """Append a row to attempts.jsonl. Call once at pipeline start (status='started')
    and once at end (status in {'ok','failed'}). Persists even when result.json is
    not produced (e.g. plan parse fail, npm timeout, watchdog kill)."""
    entry = {
        "ts": datetime.now().isoformat(),
        "status": status,
        "project": project or CURRENT_PROJECT.get(),
        "model": model or CURRENT_MODEL.get(),
        "run_dir": str(run_dir) if run_dir else None,
        "error": (str(error)[:300] if error else None),
        **extra,
    }
    ATTEMPTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTEMPTS_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry

# ─── LLM Helpers ──────────────────────────────────────────────────
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=300.0)


_MODEL_MAX_TOKENS_CAP = {
    "qwen2.5-vl-3b-instruct": 8192,
    "qwen2.5-vl-7b-instruct": 8192,
    "qwen2.5-vl-32b-instruct": 8192,
    "qwen2.5-vl-72b-instruct": 8192,
}


def chat(client, model, messages, max_tokens=4096, stage="unknown", retries=8):
    cap = _MODEL_MAX_TOKENS_CAP.get(model)
    if cap is not None and max_tokens > cap:
        max_tokens = cap
    for attempt in range(retries):
        try:
            stream = client.chat.completions.create(
                model=model, max_tokens=max_tokens, messages=messages, stream=True,
                stream_options={"include_usage": True},
            )
            text = ""
            finish = "stop"
            prompt_tokens = 0
            completion_tokens = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
            text = text.strip()
            # Fallback: estimate completion tokens from text if API didn't report
            if completion_tokens == 0 and text:
                completion_tokens = len(text) // 4
            log_cost(stage, model, prompt_tokens, completion_tokens)
            print(f"  [Tokens] {stage}: {prompt_tokens} in + {completion_tokens} out")
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            return text, finish
        except Exception as e:
            if attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  [API] Error: {str(e)[:80]}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                raise

