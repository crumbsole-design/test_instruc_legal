import hashlib
import json
import os


def get_step_hash(step) -> str:
    """SHA-256 of model + temperature + num_ctx + prompt. Changes trigger a re-run."""
    key = f"{step.model}|{step.temperature}|{step.num_ctx}|{step.prompt}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _load(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}


def is_cached(file_name: str, step_id: str, step_hash: str, cache_path: str) -> bool:
    return _load(cache_path).get(file_name, {}).get(step_id) == step_hash


def save_cache(file_name: str, step_id: str, step_hash: str, cache_path: str):
    cache = _load(cache_path)
    cache.setdefault(file_name, {})[step_id] = step_hash
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
