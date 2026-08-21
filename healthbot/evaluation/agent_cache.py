"""
Persistent cache for agent evaluation results.

Supports:
- Cache-first evaluation (minimize API calls)
- Cache versioning (invalidation when agent changes)
- Separate execution from evaluation
- Free-tier quota awareness

Cache Key Design:
    hash(query + model + agent_version + tool_schema_version + eval_version)

This ensures cache invalidation when behavior-affecting changes occur.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Cache versioning - increment when agent prompt, tools, or evaluation logic changes
AGENT_EVAL_VERSION = "v1"

# Cache configuration
CACHE_DIR = Path("evaluation_cache")
CACHE_FILE = CACHE_DIR / "agent_results.jsonl"


def get_cache_key(
    query: str,
    model: str,
    agent_version: str = AGENT_EVAL_VERSION,
    patient_level: str = "beginner"
) -> str:
    """
    Generate deterministic cache key for query.

    Cache key changes if any behavior-affecting input changes:
    - query: Different question
    - model: Different LLM
    - agent_version: Agent prompt/tools changed
    - patient_level: Different education level

    Args:
        query: User query
        model: LLM model identifier
        agent_version: Agent version (for invalidation)
        patient_level: Education level

    Returns:
        SHA256 hash as cache key
    """
    cache_input = f"{query}|{model}|{agent_version}|{patient_level}"
    return hashlib.sha256(cache_input.encode()).hexdigest()[:16]


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def save_to_cache(
    query: str,
    model: str,
    result: Dict[str, Any],
    patient_level: str = "beginner",
    status: str = "success",
    error: Optional[str] = None
):
    """
    Save agent result to persistent cache.

    Only successful results should be cached. Failures (rate limits, errors)
    should NOT be cached as successful results.

    Args:
        query: User query
        model: LLM model identifier
        result: Agent execution result
        patient_level: Education level
        status: "success" or "error" (only success gets cached)
        error: Error message if status is "error"
    """
    ensure_cache_dir()

    cache_key = get_cache_key(query, model, patient_level=patient_level)

    cache_entry = {
        "cache_key": cache_key,
        "query": query,
        "model": model,
        "agent_version": AGENT_EVAL_VERSION,
        "patient_level": patient_level,
        "timestamp": datetime.now().isoformat(),
        "status": status,
    }

    if status == "success":
        # Cache successful result
        cache_entry.update({
            "tools_called": result.get("tools_called", []),
            "summary": result.get("summary", ""),
            "disclaimer_shown": result.get("disclaimer_shown", False),
            "tool_call_trace": result.get("tool_call_trace", []),
        })
    else:
        # Record error but don't treat as successful result
        cache_entry["error"] = error

    # Append to JSONL cache file
    with open(CACHE_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(cache_entry) + '\n')


def load_from_cache(
    query: str,
    model: str,
    patient_level: str = "beginner"
) -> Optional[Dict[str, Any]]:
    """
    Load agent result from cache if available.

    Returns None if:
    - Cache file doesn't exist
    - No matching cache entry
    - Cached entry was an error (not a success)

    Args:
        query: User query
        model: LLM model identifier
        patient_level: Education level

    Returns:
        Cached result or None
    """
    if not CACHE_FILE.exists():
        return None

    cache_key = get_cache_key(query, model, patient_level=patient_level)

    # Read JSONL file and find matching entry
    # Use last matching entry (most recent)
    cached_result = None

    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)

            # Match on cache_key and successful status
            if entry.get("cache_key") == cache_key and entry.get("status") == "success":
                cached_result = entry

    return cached_result


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dictionary with cache stats
    """
    if not CACHE_FILE.exists():
        return {
            "total_entries": 0,
            "successful_entries": 0,
            "error_entries": 0,
            "unique_queries": 0,
        }

    entries = []
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    successful = [e for e in entries if e.get("status") == "success"]
    errors = [e for e in entries if e.get("status") != "success"]
    unique_keys = set(e.get("cache_key") for e in entries)

    return {
        "total_entries": len(entries),
        "successful_entries": len(successful),
        "error_entries": len(errors),
        "unique_queries": len(unique_keys),
    }


def clear_cache():
    """Delete cache file."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print(f"Cache cleared: {CACHE_FILE}")
    else:
        print("No cache file to clear")


if __name__ == "__main__":
    # CLI for cache management
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats = get_cache_stats()
        print("Cache Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Successful: {stats['successful_entries']}")
        print(f"  Errors: {stats['error_entries']}")
        print(f"  Unique queries: {stats['unique_queries']}")
    elif len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_cache()
    else:
        print("Usage:")
        print("  python -m healthbot.evaluation.agent_cache stats")
        print("  python -m healthbot.evaluation.agent_cache clear")
