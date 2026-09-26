"""Request specs + body builders shared by the realtime and batch paths of every provider.

A :class:`RequestSpec` is the single source of truth for one logical LLM call. Its
``cache_key()`` hashes only the request-defining fields (model, endpoint, prompt,
token cap, schema, seed, reasoning effort, cache version) — never ``custom_id`` or
``meta`` — so two essays with identical prompts would dedupe (desired) while
different essays never collide (their text is in ``user``).

Five endpoints are supported:
  * ``/v1/chat/completions`` — free-form generation. Reasoning models
    (gpt-5/o-series) take ``max_completion_tokens`` + optional ``reasoning_effort``
    and omit ``temperature``; legacy chat models take ``max_tokens`` + ``temperature``.
  * ``/v1/responses`` — OpenAI strict-JSON scoring (``json_schema`` response format).
  * ``:generateContent`` — Gemini. Same logical call again: the ``json_schema`` becomes
    a ``responseSchema`` with ``responseMimeType="application/json"``. Gemini honours
    ``propertyOrdering``, so the verdict-before-rationale generation order the schema
    intends is preserved here (the Anthropic tool-use path cannot guarantee it).
  * ``/v1/messages`` — Anthropic. Same logical call as ``/v1/responses``: the
    ``json_schema`` becomes a single tool and ``tool_choice`` forces it, since the
    Messages API has no strict-JSON response format. Endpoint and model are both in
    ``canonical()``, so Anthropic records can never collide with OpenAI ones.
  * ``deepseek:/chat/completions`` — DeepSeek (OpenAI chat dialect at its own base_url),
    with the schema carried by forced tool use as on the Anthropic path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.judging.cache import request_hash

CHAT_ENDPOINT = "/v1/chat/completions"
RESPONSES_ENDPOINT = "/v1/responses"
MESSAGES_ENDPOINT = "/v1/messages"  # Anthropic
GENERATE_ENDPOINT = ":generateContent"  # Gemini
#: DeepSeek speaks the OpenAI chat/completions dialect but at its own base_url, and
#: `canonical()` carries no provider/base_url field -- so it gets its OWN endpoint
#: constant rather than reusing CHAT_ENDPOINT, both to dispatch unambiguously in
#: `realtime_call` and to guarantee cache keys can never collide with an OpenAI spec.
DEEPSEEK_ENDPOINT = "deepseek:/chat/completions"


def gemini_response_schema(json_schema: dict) -> dict:
    """Translate a strict OpenAI json_schema into Gemini's OpenAPI-3 subset.

    Explicit rather than pass-through, because the dialects genuinely differ:
      * ``additionalProperties`` is dropped -- Gemini rejects it;
      * ``propertyOrdering`` is ADDED from each object's declared ``properties`` order,
        which makes Gemini emit ``overall_winner`` before ``qualities`` exactly as the
        schema intends (property order == generation order);
      * ``enum``/``maxLength`` carry over unchanged (both are supported).
    """

    def conv(node: dict) -> dict:
        out: dict = {}
        for k, v in node.items():
            if k == "additionalProperties":
                continue
            if k == "properties":
                out["properties"] = {pk: conv(pv) for pk, pv in v.items()}
                out["propertyOrdering"] = list(v)  # declared order == generation order
            elif k == "items":
                out["items"] = conv(v)
            else:
                out[k] = v
        return out

    return conv(json_schema["schema"])


def _is_legacy_chat(model: str) -> bool:
    return model.startswith(("gpt-4", "gpt-3"))


@dataclass(frozen=True)
class RequestSpec:
    custom_id: str
    model: str
    system: str
    user: str
    max_output_tokens: int
    endpoint: str = CHAT_ENDPOINT
    temperature: float = 0.0
    seed: int | None = None
    json_schema: dict | None = None  # {"name": ..., "schema": {...}} for /v1/responses
    reasoning_effort: str | None = None
    thinking_level: str | None = None  # Gemini only: MINIMAL|LOW|MEDIUM|HIGH
    thinking: str | None = None  # DeepSeek only: "enabled"|"disabled"
    cache_version: str = "v1"
    meta: dict = field(default_factory=dict, compare=False)  # essay_id/judge/condition — not hashed

    def canonical(self) -> dict:
        # `thinking_level` / `thinking` are appended only when set, so specs that do not
        # use them hash independently of these provider-specific fields.
        d = {
            "endpoint": self.endpoint,
            "model": self.model,
            "system": self.system,
            "user": self.user,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "json_schema": self.json_schema,
            "reasoning_effort": self.reasoning_effort,
            "cache_version": self.cache_version,
        }
        if self.thinking_level is not None:
            d["thinking_level"] = self.thinking_level
        if self.thinking is not None:
            d["thinking"] = self.thinking
        return d

    def cache_key(self) -> str:
        return request_hash(self.canonical())

    # -- body builders (identical shape for realtime calls and batch JSONL lines) --
    def body(self) -> dict:
        if self.endpoint == CHAT_ENDPOINT:
            return self._chat_body()
        if self.endpoint == RESPONSES_ENDPOINT:
            return self._responses_body()
        if self.endpoint == MESSAGES_ENDPOINT:
            return self._messages_body()
        if self.endpoint == GENERATE_ENDPOINT:
            return self._gemini_body()
        if self.endpoint == DEEPSEEK_ENDPOINT:
            return self._deepseek_body()
        raise ValueError(f"unsupported endpoint {self.endpoint!r}")

    def _chat_body(self) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
        }
        if _is_legacy_chat(self.model):
            body["max_tokens"] = self.max_output_tokens
            body["temperature"] = self.temperature
            if self.seed is not None:
                body["seed"] = self.seed
        else:
            body["max_completion_tokens"] = self.max_output_tokens
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        return body

    def _responses_body(self) -> dict:
        body: dict = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.json_schema:
            body["text"] = {"format": {"type": "json_schema", **self.json_schema, "strict": True}}
        return body

    def _messages_body(self) -> dict:
        """Anthropic Messages body. Schema-conformant JSON comes from forced tool use.

        The Messages API has no strict-JSON response format, so the ``json_schema`` is
        declared as the sole tool and ``tool_choice`` forces it. ``thinking`` is
        deliberately omitted -- that is the analogue of ``reasoning_effort: none`` on the
        OpenAI side, and Haiku 4.5 predates both adaptive thinking and ``output_config``.
        """
        body: dict = {
            "model": self.model,
            "system": self.system,
            "messages": [{"role": "user", "content": self.user}],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.json_schema:
            body["tools"] = [
                {
                    "name": self.json_schema["name"],
                    "description": "Record the comparison result.",
                    "input_schema": self.json_schema["schema"],
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": self.json_schema["name"]}
        return body

    def _gemini_body(self) -> dict:
        """Gemini wire-format body, used verbatim by both the realtime and batch paths.

        Kept in wire (camelCase) form because the Batch API consumes JSONL of exactly this
        shape; the realtime branch coerces it back through ``types.GenerateContentConfig``.
        Gemini 3.x controls reasoning with ``thinkingLevel``, not ``thinkingBudget``:
        this model returns 400 INVALID_ARGUMENT for ``thinkingBudget: 0``. ``MINIMAL``
        is the off-equivalent -- it returns no thought tokens at all -- and is what gives
        parity with ``reasoning_effort=none`` on the OpenAI judge and Haiku's
        no-extended-thinking default. Thoughts bill as output, so this also bounds cost.
        """
        gen: dict = {
            "maxOutputTokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.json_schema:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = gemini_response_schema(self.json_schema)
        if self.thinking_level is not None:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        return {
            "model": self.model,
            "contents": [{"role": "user", "parts": [{"text": self.user}]}],
            "systemInstruction": {"parts": [{"text": self.system}]},
            "generationConfig": gen,
        }

    def _deepseek_body(self) -> dict:
        """DeepSeek chat/completions body. Schema conformance via forced tool use.

        DeepSeek has no ``/v1/responses`` and no strict ``json_schema`` response format --
        only ``json_object`` -- and the frozen prompt never states the field names (they
        live in ``COMPARATIVE_SCHEMA``), so bare JSON mode could not reproduce
        ``overall_winner``/``qualities``/... Forced tool use carries the schema instead,
        as on the Anthropic path, with ``strict`` for exact compliance.

        ``max_tokens`` rather than ``max_completion_tokens``: ``_is_legacy_chat`` matches
        only gpt-4/gpt-3 ids, so the generic chat builder would send the wrong field here.

        ``thinking`` defaults to ENABLED on DeepSeek (``reasoning_effort`` "high"), and
        reasoning bills as output -- ``"disabled"`` is what gives parity with the other
        judges and bounds the cost.
        """
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.thinking is not None:
            body["thinking"] = {"type": self.thinking}
        if self.json_schema:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": self.json_schema["name"],
                        "description": "Record the comparison result.",
                        "parameters": self.json_schema["schema"],
                        "strict": True,
                    },
                }
            ]
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": self.json_schema["name"]},
            }
        return body


def build_continuation_spec(
    custom_id,
    model,
    system,
    user,
    max_output_tokens,
    *,
    reasoning_effort=None,
    seed=None,
    cache_version="v1",
    meta=None,
) -> RequestSpec:
    return RequestSpec(
        custom_id=custom_id,
        model=model,
        system=system,
        user=user,
        max_output_tokens=max_output_tokens,
        endpoint=CHAT_ENDPOINT,
        reasoning_effort=reasoning_effort,
        seed=seed,
        cache_version=cache_version,
        meta=meta or {},
    )


def build_scoring_spec(
    custom_id,
    model,
    system,
    user,
    json_schema,
    max_output_tokens,
    *,
    reasoning_effort=None,
    cache_version="v1",
    meta=None,
) -> RequestSpec:
    return RequestSpec(
        custom_id=custom_id,
        model=model,
        system=system,
        user=user,
        max_output_tokens=max_output_tokens,
        endpoint=RESPONSES_ENDPOINT,
        json_schema=json_schema,
        reasoning_effort=reasoning_effort,
        cache_version=cache_version,
        meta=meta or {},
    )


def build_anthropic_scoring_spec(
    custom_id,
    model,
    system,
    user,
    json_schema,
    max_output_tokens,
    *,
    temperature=0.0,
    cache_version="v1",
    meta=None,
) -> RequestSpec:
    """Anthropic twin of :func:`build_scoring_spec` (forced tool use for the schema)."""
    return RequestSpec(
        custom_id=custom_id,
        model=model,
        system=system,
        user=user,
        max_output_tokens=max_output_tokens,
        endpoint=MESSAGES_ENDPOINT,
        temperature=temperature,
        json_schema=json_schema,
        reasoning_effort=None,
        cache_version=cache_version,
        meta=meta or {},
    )


def build_gemini_scoring_spec(
    custom_id,
    model,
    system,
    user,
    json_schema,
    max_output_tokens,
    *,
    temperature=0.0,
    thinking_level="MINIMAL",
    cache_version="v1",
    meta=None,
) -> RequestSpec:
    """Gemini twin of :func:`build_scoring_spec` (responseSchema + propertyOrdering)."""
    return RequestSpec(
        custom_id=custom_id,
        model=model,
        system=system,
        user=user,
        max_output_tokens=max_output_tokens,
        endpoint=GENERATE_ENDPOINT,
        temperature=temperature,
        json_schema=json_schema,
        reasoning_effort=None,
        thinking_level=thinking_level,
        cache_version=cache_version,
        meta=meta or {},
    )


def build_deepseek_scoring_spec(
    custom_id,
    model,
    system,
    user,
    json_schema,
    max_output_tokens,
    *,
    temperature=0.0,
    thinking="disabled",
    cache_version="v1",
    meta=None,
) -> RequestSpec:
    """DeepSeek twin of :func:`build_scoring_spec` (forced tool use, thinking off)."""
    return RequestSpec(
        custom_id=custom_id,
        model=model,
        system=system,
        user=user,
        max_output_tokens=max_output_tokens,
        endpoint=DEEPSEEK_ENDPOINT,
        temperature=temperature,
        json_schema=json_schema,
        reasoning_effort=None,
        thinking=thinking,
        cache_version=cache_version,
        meta=meta or {},
    )
