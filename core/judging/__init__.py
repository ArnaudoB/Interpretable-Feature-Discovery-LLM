"""LLM judging utilities: content-addressed response cache, request specs,
OpenAI Batch-API orchestration, retry/backoff, and cost accounting.

Submodules are imported directly, e.g. ``from core.judging import cache`` or
``from core.judging.batch import build_jsonl, submit_batch``.
"""
