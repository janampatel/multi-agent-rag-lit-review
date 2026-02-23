"""
utils/cache.py — Disk-based LLM response caching.

Why cache LLM responses?
- Ollama calls are slow (2-10s per call even locally).
- Re-running the same query on unchanged data gives identical results.
- Caching avoids redundant inference during development and repeated runs.

Cache key: SHA-256 hash of all inputs that determine the output (model, prompt, temperature).
Storage:   diskcache.Cache — SQLite-backed, safe for concurrent access, auto-evicts by size.
"""

import hashlib
import json
import os
from typing import Any, Callable

import diskcache

_cache: diskcache.Cache | None = None
_CACHE_DIR = os.getenv("CACHE_DIR", ".cache/llm_responses")


def get_cache() -> diskcache.Cache:
    """Lazily initialises and returns the disk cache singleton."""
    global _cache
    if _cache is None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        # size_limit=500MB — prevents runaway disk usage
        _cache = diskcache.Cache(_CACHE_DIR, size_limit=500 * 1024 * 1024)
    return _cache


def make_cache_key(key_data: dict) -> str:
    """
    Produces a stable SHA-256 key from any JSON-serialisable dict.
    Sort keys so insertion order doesn't affect the hash.
    """
    serialised = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def cached_call(fn: Callable, key_data: dict) -> Any:
    """
    Wraps any callable with transparent disk caching.

    Usage:
        result = cached_call(
            fn=lambda: llm_chain.invoke({"topic": query}),
            key_data={"model": model_name, "prompt_template": template, "input": query}
        )

    Args:
        fn:       A zero-argument callable that performs the expensive operation.
        key_data: dict of all inputs that uniquely determine the output.

    Returns:
        Cached result on hit; fn() result (stored in cache) on miss.
    """
    cache = get_cache()
    key = make_cache_key(key_data)

    if key in cache:
        print(f"[Cache HIT]  {list(key_data.keys())}")
        return cache[key]

    print(f"[Cache MISS] {list(key_data.keys())} — calling LLM...")
    result = fn()
    cache[key] = result
    return result


def clear_cache():
    """Clears all cached LLM responses. Useful for testing fresh runs."""
    cache = get_cache()
    count = len(cache)
    cache.clear()
    print(f"Cache cleared — {count} entries removed.")
