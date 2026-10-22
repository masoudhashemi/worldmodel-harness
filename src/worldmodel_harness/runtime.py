"""Thin model-client adapter. Stirrup owns the agent loop."""

import asyncio
import json
from time import perf_counter

from pydantic import SecretStr
from stirrup.clients.litellm_client import LiteLLMClient
from stirrup.core.models import AssistantMessage, TextBlock, TokenUsage

from .config import ModelSettings


class BoundedClient:
    """Stop waiting for a model response and hand back an empty message."""

    def __init__(self, client, timeout_seconds: float):
        self.client = client
        self.timeout_seconds = timeout_seconds

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def generate(self, messages, tools):
        try:
            return await asyncio.wait_for(
                self.client.generate(messages, tools),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            return AssistantMessage(
                blocks=[TextBlock(text="")],
                token_usage=TokenUsage(),
            )


def parse_gemma_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    prefix = "<|tool_call>call:"
    text = text.replace("<|tool_call|>call:", prefix).replace("<tool_call>call:", prefix)
    calls = []
    clean_text_parts = []
    last_end = 0
    idx = 0
    while True:
        start = text.find(prefix, idx)
        if start == -1:
            clean_text_parts.append(text[last_end:])
            break
        clean_text_parts.append(text[last_end:start])
        fn_start = start + len(prefix)
        brace_open = text.find("{", fn_start)
        if brace_open == -1:
            clean_text_parts.append(text[start:])
            break
        fn_name = text[fn_start:brace_open].strip()
        depth = 0
        i = brace_open
        end = -1
        in_str = False
        while i < len(text):
            if text[i : i + 5] == "<|\"|>":
                in_str = not in_str
                i += 5
                continue
            if not in_str:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            i += 1
        if end == -1:
            clean_text_parts.append(text[start:])
            break
        args_raw = text[brace_open : end + 1]
        close_tag = "<tool_call|>"
        tag_end = end + 1
        if text[end + 1 : end + 1 + len(close_tag)] == close_tag:
            tag_end += len(close_tag)
        last_end = tag_end
        idx = tag_end

        def tokenize_gemma_obj(s):
            tokens, j, n = [], 0, len(s)
            while j < n:
                if s[j].isspace():
                    j += 1
                    continue
                if s[j] in "{}[],:":
                    tokens.append(s[j])
                    j += 1
                    continue
                if s[j : j + 5] == "<|\"|>":
                    end_q = s.find("<|\"|>", j + 5)
                    if end_q == -1:
                        tokens.append(s[j:])
                        break
                    tokens.append(s[j : end_q + 5])
                    j = end_q + 5
                    continue
                if s[j] == "\"":
                    end_q = s.find("\"", j + 1)
                    while end_q != -1 and s[end_q - 1] == "\\":
                        end_q = s.find("\"", end_q + 1)
                    if end_q == -1:
                        tokens.append(s[j:])
                        break
                    tokens.append(s[j : end_q + 1])
                    j = end_q + 1
                    continue
                start_id = j
                while j < n and not s[j].isspace() and s[j] not in "{}[],:":
                    if s[j : j + 5] == "<|\"|>" or s[j] == "\"":
                        break
                    j += 1
                if start_id == j:
                    tokens.append(s[j])
                    j += 1
                else:
                    tokens.append(s[start_id:j])
            return tokens

        def parse_gemma_val(tokens, pos):
            if pos >= len(tokens):
                return None, pos
            tok = tokens[pos]
            if tok == "{":
                obj = {}
                pos += 1
                while pos < len(tokens) and tokens[pos] != "}":
                    k = tokens[pos]
                    if k.startswith("<|\"|>") and k.endswith("<|\"|>"):
                        k = k[5:-5]
                    elif k.startswith("\"") and k.endswith("\""):
                        try:
                            k = json.loads(k)
                        except Exception:
                            k = k[1:-1]
                    pos += 1
                    if pos >= len(tokens) or tokens[pos] != ":":
                        break
                    pos += 1
                    v, pos = parse_gemma_val(tokens, pos)
                    obj[k] = v
                    if pos < len(tokens) and tokens[pos] == ",":
                        pos += 1
                if pos < len(tokens) and tokens[pos] == "}":
                    pos += 1
                return obj, pos
            elif tok == "[":
                arr = []
                pos += 1
                while pos < len(tokens) and tokens[pos] != "]":
                    v, pos = parse_gemma_val(tokens, pos)
                    arr.append(v)
                    if pos < len(tokens) and tokens[pos] == ",":
                        pos += 1
                if pos < len(tokens) and tokens[pos] == "]":
                    pos += 1
                return arr, pos
            elif tok.startswith("<|\"|>") and tok.endswith("<|\"|>"):
                return tok[5:-5], pos + 1
            elif tok.startswith("\"") and tok.endswith("\""):
                try:
                    return json.loads(tok), pos + 1
                except Exception:
                    return tok[1:-1], pos + 1
            elif tok in ("true", "True"):
                return True, pos + 1
            elif tok in ("false", "False"):
                return False, pos + 1
            elif tok in ("null", "None"):
                return None, pos + 1
            else:
                try:
                    if "." in tok:
                        return float(tok), pos + 1
                    return int(tok), pos + 1
                except ValueError:
                    return tok, pos + 1

        try:
            clean_json = args_raw.replace("<|\"|>", "\"")
            parsed_dict = json.loads(clean_json)
            calls.append((fn_name, parsed_dict))
        except Exception:
            try:
                toks = tokenize_gemma_obj(args_raw)
                parsed_dict, _ = parse_gemma_val(toks, 0)
                if isinstance(parsed_dict, dict):
                    calls.append((fn_name, parsed_dict))
                else:
                    calls.append((fn_name, {"_raw": args_raw}))
            except Exception:
                calls.append((fn_name, {"_raw": args_raw}))

    clean_text = "".join(clean_text_parts).strip()
    return clean_text, calls


class GemmaLiteLLMClient(LiteLLMClient):
    """LiteLLM client that handles Gemma native tool calling formatting on vLLM."""

    async def generate(self, messages, tools):
        import json
        import uuid
        from litellm import acompletion
        from litellm.exceptions import ContextWindowExceededError
        from stirrup.clients.litellm_client import (
            _parse_thinking_blocks,
            to_openai_messages,
            to_openai_tools,
        )
        from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
        from stirrup.core.models import AssistantMessage, ReasoningBlock, TextBlock, TokenUsage, ToolCall

        try:
            r = await acompletion(
                model=self.model_slug,
                messages=to_openai_messages(messages, allow_tool_call_signatures=True),
                tools=to_openai_tools(tools) if tools else None,
                tool_choice="none" if tools else None,
                max_tokens=self._max_tokens,
                reasoning_effort=self._reasoning_effort,
                api_key=self._api_key,
                **self._kwargs,
            )
        except ContextWindowExceededError as e:
            raise ContextOverflowError(str(e)) from e

        choice = r["choices"][0]
        if choice.finish_reason in ("max_tokens", "length"):
            raise OutputTokenLimitError(
                model_slug=self.model_slug,
                max_tokens=self._max_tokens,
                provider_reason=choice.finish_reason,
            )
        msg = choice["message"]
        blocks = list(_parse_thinking_blocks(getattr(msg, "thinking_blocks", None)))
        if not blocks and getattr(msg, "reasoning_content", None) is not None:
            blocks.append(ReasoningBlock(content=msg.reasoning_content))

        raw_content = msg.get("content") or ""
        clean_text, parsed_calls = parse_gemma_calls(raw_content)
        if clean_text:
            blocks.append(TextBlock(text=clean_text))
        for fn_name, fn_args in parsed_calls:
            blocks.append(
                ToolCall.from_provider(
                    provider_id=f"call_{uuid.uuid4().hex[:8]}",
                    name=fn_name,
                    arguments=json.dumps(fn_args),
                )
            )
        for tc in msg.get("tool_calls") or []:
            blocks.append(
                ToolCall.from_provider(
                    provider_id=tc.get("id"),
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
            )
        usage = r.get("usage") or {}
        token_usage = TokenUsage(
            input=usage.get("prompt_tokens") or 0,
            answer=usage.get("completion_tokens") or 0,
            reasoning=getattr(usage.get("completion_tokens_details") or {}, "reasoning_tokens", 0)
            if isinstance(usage.get("completion_tokens_details"), dict)
            else 0,
        )
        return AssistantMessage(blocks=blocks, token_usage=token_usage)


def build_client(settings: ModelSettings, api_key: SecretStr) -> BoundedClient:
    is_gemma = "gemma" in settings.name.lower() or "elementai.com" in settings.api_base
    client_cls = GemmaLiteLLMClient if is_gemma else LiteLLMClient
    client = client_cls(
        model=settings.name,
        api_key=api_key.get_secret_value(),
        max_tokens=settings.max_tokens,
        context_window_tokens=settings.context_window_tokens,
        kwargs={
            "api_base": settings.api_base,
            "temperature": settings.temperature,
            "timeout": settings.timeout_seconds,
            "num_retries": 0,
            **({"response_format": {"type": "json_object"}} if settings.json_mode else {}),
            **(
                {"extra_body": {"reasoning": {"effort": settings.reasoning_effort}}}
                if settings.reasoning_effort is not None
                else {}
            ),
        },
    )
    return BoundedClient(client, settings.timeout_seconds)


class MeteredClient:
    """Record returned usage and failed-call latency without SDK error strings."""

    def __init__(self, client):
        self.client = client
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def generate(self, messages, tools):
        started = perf_counter()
        cost = {"usage_available": False}
        try:
            response = await self.client.generate(messages, tools)
            cost["token_usage"] = response.token_usage.model_dump()
            cost["usage_available"] = True
            return response
        except BaseException as exc:
            cost["error_type"] = type(exc).__name__
            raise
        finally:
            cost["elapsed_seconds"] = perf_counter() - started
            self.calls.append(cost)
