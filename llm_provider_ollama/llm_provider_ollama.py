#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
import threading
import uuid
from decimal import Decimal
from queue import Queue
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import ollama
import pendulum
from llm_provider_base import (
    LlmProviderHandler,
    bearer,
    endpoint_base,
    fetch_upstream_models,
)
from silvaengine_utility.performance_monitor import performance_monitor
from silvaengine_utility.serializer import Serializer

# Names of built-in Ollama web tools; actual methods are on self.client for auth
WEB_TOOL_NAMES = {"web_fetch", "web_search"}

# Maximum characters for search tool results (~2000 tokens)
WEB_RESULT_MAX_CHARS = 2000 * 4


def format_search_results(results, user_search: str) -> str:
    """
    Format search/fetch tool results into structured text for the model.
    Follows the official ollama-python example format.

    Args:
        results: WebSearchResponse or WebFetchResponse from ollama
        user_search: The original query or URL used for the search/fetch
    """
    output = []
    if isinstance(results, ollama.WebSearchResponse):
        output.append(f'Search results for "{user_search}":')
        for result in results.results:
            output.append(f"{result.title}" if result.title else f"{result.content}")
            output.append(f"   URL: {result.url}")
            output.append(f"   Content: {result.content}")
            output.append("")
        return "\n".join(output).rstrip()

    elif isinstance(results, ollama.WebFetchResponse):
        output.append(f'Fetch results for "{user_search}":')
        output.extend(
            [
                f"Title: {results.title}",
                f"URL: {user_search}" if user_search else "",
                f"Content: {results.content}",
            ]
        )
        if results.links:
            output.append(f'Links: {", ".join(results.links)}')
        output.append("")
        return "\n".join(output).rstrip()

    # Fallback for unexpected types
    return str(results)


# ----------------------------
# Ollama Response Streaming with Function Handling and History
# ----------------------------
class OllamaProvider(LlmProviderHandler):
    """
    Manages conversations and function calls in real-time with Ollama.

    BanyanOS provider-plugin contract (LLM Provider Plugin plan §5.1/A.6):
    single-turn (tool_mode="return"); tools execute in agent_engine; raw
    HTTP on the injected pooled client (D7 — ``ollama.Client`` has no client
    injection point), auth per request from ``ctx.credentials``.
    """

    # ------------------------------------------------------------------
    # Plugin metadata (discovered by llm_engine via the entry point)
    # ------------------------------------------------------------------
    api_protocol = "ollama"
    #: llm_engine built-in protocol this plugin takes over (plan D8).
    replaces_protocol = "OLLAMA"
    display_name = "Ollama"
    contract_version = "1"
    fallback_protocol = None
    capabilities = {
        "streaming": True,
        "tools": True,
        "list_models": True,
    }
    default_pool_name = "httpx_llm_ollama"

    @classmethod
    def list_models(cls, ctx: Any) -> List[Dict[str, str]]:
        """Ollama model list: ``GET {root}/api/tags`` (``…/v1`` endpoints accepted)."""
        root = cls._native_root(endpoint_base(ctx, "http://localhost:11434"))
        return fetch_upstream_models(
            ctx, f"{root}/api/tags", headers=bearer(ctx), items_key="models", id_key="name",
        )

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initialize the Ollama event handler

        Args:
            logger: Logger instance for debug/info messages
            agent: Configuration dict containing model instructions and settings
            setting: Additional settings for handler configuration
        """
        LlmProviderHandler.__init__(self, logger, agent, **setting)

        # Enable timeline logging (default: False)
        self.enable_timeline_log = setting.get("enable_timeline_log", False)

        if "enabled_tools" in self.agent["configuration"]:
            # Add tools if available - matching example.py structure.
            # Neutral tool shape (plan §5.5): OpenAI function entries carry
            # the name under "function"; Ollama-shaped entries at top.

            def _tool_name(tool: Any) -> Optional[str]:
                if not isinstance(tool, dict):
                    return None
                if isinstance(tool.get("function"), dict):
                    return tool["function"].get("name")
                return tool.get("name")

            enabled_tools = []
            if "tools" in self.agent["configuration"]:
                for tool in self.agent["configuration"]["tools"]:
                    if _tool_name(tool) not in self.agent["configuration"].get(
                        "enabled_tools", []
                    ):
                        continue
                    enabled_tools.append(tool)
            self.agent["configuration"]["tools"] = enabled_tools

        # Convert Decimal to float once during initialization (performance
        # optimization). Q8/CREDS: api_key is a credential (ctx), headers
        # were client-only — both excluded from model_setting.
        self.model_setting = {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in agent["configuration"].items()
            if k not in ["api_key", "headers"]
        }

        # Cache frequently accessed configuration values (performance optimization)
        self.output_format_type = (
            self.model_setting.get("text", {}).get("format", {}).get("type", "text")
        )

        # Validate reasoning configuration if present
        if "reasoning" in self.model_setting:
            if not isinstance(self.model_setting["reasoning"], dict):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning configuration should be a dictionary. "
                        "Reasoning features may not work correctly."
                    )
            elif self.model_setting["reasoning"].get("enabled") is None:
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning is not explicitly enabled in configuration. "
                        "Reasoning events will be skipped during streaming."
                    )

        # Pre-build options dict for model invocation (performance optimization)
        option_keys = [
            "temperature",
            "top_p",
            "num_predict",
            "top_k",
            "repeat_penalty",
            "num_ctx",
        ]
        self.model_options = {
            k: self.model_setting[k] for k in option_keys if k in self.model_setting
        }

        # Plugin contract (plan D7 / CREDS): ollama.Client always builds its
        # own httpx client (no injection point — checked ollama/_client.py),
        # so the handler calls /api/chat via raw HTTP on the pooled client
        # injected by llm_engine (plan §5.2.1); auth per request from
        # ctx.credentials as ``Authorization: Bearer``.
        http_client = getattr(self.ctx, "http_client", None) if self.ctx else None
        if http_client is None:
            raise Exception(
                "OllamaProvider requires ctx.http_client (the pooled "
                "client injected by llm_engine's HandlerProviderAdapter, "
                "plan §5.2.1)"
            )
        base_url = (self.model_setting.get("base_url") or "").strip()
        if not base_url:
            raise Exception(
                "OllamaProvider requires configuration.base_url (the "
                "raw-HTTP endpoint for /api/chat, plan A.6 D7)"
            )
        self._endpoint = self._native_root(base_url)

    def _cleanup_input_messages(
        self, input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Cleans up conversation history by handling broken tool interaction sequences.

        Valid tool sequence: assistant (with tool_calls) → tool result(s) → next message.
        Handles two types of broken sequences:
        1. Orphaned tool messages (no preceding assistant with tool_calls):
           Extracts the tool output and merges it into the next assistant message
           as [Tool call context], preserving search results and other tool data.
        2. Assistant with tool_calls but no following tool results:
           Skipped entirely as the sequence is incomplete.

        Single-pass O(n) algorithm using an in_tool_sequence flag to track state.

        Args:
            input_messages: Raw conversation messages

        Returns:
            Cleaned messages with orphaned tool outputs merged into assistant responses
        """
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"[_cleanup_input_messages] Cleaning {len(input_messages)} messages"
            )

        if not input_messages:
            return []

        result = []
        tool_call_role = self.agent["tool_call_role"]
        # Track whether we're inside a valid tool call sequence
        in_tool_sequence = False
        # Collect orphaned tool outputs to merge into a single tool message
        orphaned_tool_outputs = []
        i = 0

        while i < len(input_messages):
            current_msg = input_messages[i]
            current_role = current_msg.get("role")

            if current_role == tool_call_role:
                # Tool messages are only valid when preceded by an assistant
                # message with tool_calls (either directly or via earlier tool
                # messages in the same sequence).
                if in_tool_sequence:
                    result.append(current_msg)
                else:
                    # Extract output from orphaned tool message to merge later
                    tool_output = self._extract_tool_output(current_msg)
                    if tool_output:
                        orphaned_tool_outputs.append(tool_output)
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"[_cleanup_input_messages] Orphaned tool at [{i}], collecting output"
                        )
                i += 1
                continue

            # When a non-tool message arrives, end any active tool sequence
            in_tool_sequence = False

            if current_role == "assistant" and "tool_calls" in current_msg:
                # Look ahead: valid sequence needs at least one tool result
                j = i + 1
                has_tool_results = (
                    j < len(input_messages)
                    and input_messages[j].get("role") == tool_call_role
                )

                if has_tool_results:
                    # Valid start of tool call sequence
                    result.append(current_msg)
                    in_tool_sequence = True
                    i += 1
                else:
                    # Assistant with tool_calls but no tool results follow — skip
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"[_cleanup_input_messages] Skipping assistant with tool_calls but no tool results at [{i}]"
                        )
                    i += 1
                continue

            # Flush orphaned tool outputs as a system message with context.
            # Cannot use "tool" role here since there's no preceding assistant
            # with tool_calls — Ollama rejects orphaned tool messages with 500.
            if orphaned_tool_outputs:
                merged_context = "\n\n".join(orphaned_tool_outputs)
                result.append(
                    {
                        "role": "system",
                        "content": f"[Previous tool call results]\n{merged_context}",
                    }
                )
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"[_cleanup_input_messages] Inserted merged tool context as user message with {len(orphaned_tool_outputs)} outputs before [{i}]"
                    )
                orphaned_tool_outputs.clear()

            # Regular messages (user, assistant without tool_calls)
            result.append(current_msg)
            i += 1

        # Flush any remaining orphaned tool outputs at end of messages
        if orphaned_tool_outputs:
            merged_context = "\n\n".join(orphaned_tool_outputs)
            result.append(
                {
                    "role": "system",
                    "content": f"[Previous tool call results]\n{merged_context}",
                }
            )
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[_cleanup_input_messages] Inserted merged tool context as user message with {len(orphaned_tool_outputs)} outputs at end"
                )
            orphaned_tool_outputs.clear()

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"[_cleanup_input_messages] Retained {len(result)} messages"
            )
        return result

    @staticmethod
    def _extract_tool_output(tool_msg: Dict[str, Any]) -> Optional[str]:
        """
        Extract the output from a tool message.
        Tool message content can be either a JSON string (with tool/output fields)
        or a plain string.
        """
        content = tool_msg.get("content", "")
        if not content:
            return None

        try:
            parsed = (
                Serializer.json_loads(content) if isinstance(content, str) else content
            )
            if isinstance(parsed, dict):
                tool_info = parsed.get("tool", {})
                tool_name = tool_info.get("name", "unknown")
                output = parsed.get("output", "")
                if output:
                    return f"[{tool_name}]: {output}"
            return content
        except Exception:
            return content

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time in milliseconds from the first ask_model call.

        Returns:
            Elapsed time in milliseconds, or 0 if global start time not set
        """
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        """
        Reset the global timeline for a new run.
        Should be called at the start of each new user interaction/run.
        """
        self._global_start_time = None
        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("[TIMELINE] Timeline reset for new run")

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Invokes the Ollama model with provided messages and handles tool calls

        Args:
            kwargs: Contains input messages and streaming configuration

        Returns:
            Model response or streaming response

        Raises:
            Exception: If model invocation fails
        """
        try:
            invoke_start = pendulum.now("UTC")
            # Build messages list with system instruction as proper message dict
            messages = [
                {
                    "role": "system",
                    "content": self.agent["instructions"],
                }
            ] + kwargs.get("input_messages", [])

            # Prepare chat parameters
            chat_params = {
                "model": self.model_setting.get("model"),
                "messages": messages,
            }

            # Use pre-built options from __init__ (performance optimization)
            if self.model_options:
                chat_params["options"] = self.model_options

            # Add tools if available
            # Copy tools list to avoid mutating model_setting on repeated calls
            if "tools" in self.model_setting:
                chat_params["tools"] = list(self.model_setting["tools"])

            # Append built-in server-side web tools (web_fetch, web_search)
            # when enabled — plain typed declarations over raw HTTP (plan A.6;
            # provider-native execution, client-side fallback deferred to P5).
            if "web_tools_enabled" in self.model_setting:
                if "tools" not in chat_params:
                    chat_params["tools"] = []
                if self.model_setting["web_tools_enabled"].get("web_search", False):
                    chat_params["tools"].append({"type": "web_search"})
                if self.model_setting["web_tools_enabled"].get("web_fetch", False):
                    chat_params["tools"].append({"type": "web_fetch"})

            # Add format option using cached value (performance optimization)
            if self.output_format_type == "json_object":
                chat_params["format"] = "json"
            elif self.output_format_type == "json_schema":
                # Ollama supports json schema format
                format_config = self.model_setting.get("text", {}).get("format", {})
                if "schema" in format_config:
                    chat_params["format"] = format_config["schema"]

            # Add reasoning/thinking parameter - must explicitly set False to
            # disable, since Ollama enables thinking by default for supported models
            chat_params["think"] = False  # Default to False for backward compatibility
            reasoning_config = self.model_setting.get("reasoning", {})
            if isinstance(reasoning_config, dict) and reasoning_config.get("enabled"):
                chat_params["think"] = True
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("[invoke_model] Reasoning enabled, think=True")

            # Return streaming or non-streaming response
            if kwargs["stream"]:
                chat_params["stream"] = True

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[invoke_model] Sending messages: {Serializer.json_dumps(chat_params.get('messages', []))}"
                )

            url = self._endpoint_url()
            headers = self._request_headers()

            if chat_params.get("stream"):
                # Streaming: NDJSON lines over the pooled client → plain dicts
                # (the parser below consumes ``chunk.get(...)`` shapes).

                def _stream() -> Iterator[Dict[str, Any]]:
                    with self.ctx.http_client.stream(
                        "POST", url, json=chat_params, headers=headers
                    ) as http_response:
                        if http_response.status_code >= 400:
                            raise self._http_error(
                                http_response.status_code,
                                getattr(http_response, "text", "") or "",
                            )
                        for line in http_response.iter_lines():
                            stripped = (line or "").strip()
                            if not stripped:
                                continue
                            yield json.loads(stripped)

                if self.enable_timeline_log and self.logger.isEnabledFor(
                    logging.INFO
                ):
                    invoke_end = pendulum.now("UTC")
                    invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                    )

                return _stream()

            http_response = self.ctx.http_client.post(
                url, json=chat_params, headers=headers
            )
            if http_response.status_code >= 400:
                raise self._http_error(
                    http_response.status_code,
                    getattr(http_response, "text", "") or "",
                )
            result = http_response.json()

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result

        except Exception as e:
            if getattr(e, "response", None) is not None:
                # Already duck-typed for mapping (plan §5.4) — pass through.
                raise
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    @performance_monitor.monitor_operation(operation_name="Ollama")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Makes a request to the Ollama model with streaming or non-streaming mode

        Args:
            input_messages: List of conversation messages
            queue: Queue for streaming events
            stream_event: Event to signal streaming completion
            model_setting: Optional model configuration overrides

        Returns:
            Response ID for non-streaming requests, None for streaming

        Raises:
            Exception: If request processing fails
        """
        # Track preparation time
        ask_model_start = pendulum.now("UTC")

        # Track recursion depth to identify top-level vs recursive calls
        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        # Establish per-run state here rather than relying on __init__:
        # handler instances are cached and reused across requests, so building
        # these in __init__ and mutating them in place (final_output.update(...))
        # meant concurrent runs shared one set of containers and overwrote each
        # other's results — final_output *is* the response payload. Rebinding,
        # never mutating, is what isolates each run.
        #
        # Only at top level: this method recurses for tool-call follow-ups, and
        # final_output / _short_term_memory accumulate across those levels.
        #
        # Inlined rather than a base-class helper on purpose: llm_provider_base
        # is a separately installed package, so calling a new base method would
        # raise AttributeError wherever that package is older than this one.
        if is_top_level:
            self.accumulated_text = ""
            self.final_output = {}
            self.uploaded_files = []
            self._short_term_memory = []

        # Initialize global start time only on top-level ask_model call
        # Recursive calls will use the same start time for the entire run timeline
        if is_top_level:
            self._global_start_time = ask_model_start

            # Reset reasoning_summary for new conversation turn
            # Recursive calls (function call loops) will continue accumulating
            if "reasoning_summary" in self.final_output:
                del self.final_output["reasoning_summary"]

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                self.logger.info("[TIMELINE] T+0ms: Run started - First ask_model call")
        else:
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started"
                )

        try:
            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                # Rebind rather than mutate: model_setting is built once in
                # __init__ and shared with every other instance of this cached
                # handler, so an in-place update would leak this request's
                # overrides into all later requests.
                self.model_setting = {**self.model_setting, **model_setting}

            # Clean up input messages to remove broken tool sequences (performance optimization)
            cleanup_start = pendulum.now("UTC")
            input_messages = self._cleanup_input_messages(input_messages)
            cleanup_end = pendulum.now("UTC")
            cleanup_time = (cleanup_end - cleanup_start).total_seconds() * 1000

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Track total preparation time before API call
                preparation_end = pendulum.now("UTC")
                preparation_time = (
                    preparation_end - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms, cleanup: {cleanup_time:.2f}ms)"
                )

            timestamp = pendulum.now("UTC").int_timestamp
            # Optimized UUID generation - use .hex instead of str() conversion
            run_id = f"run-ollama-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

            response = self.invoke_model(
                **{
                    "input_messages": input_messages,
                    "stream": stream,
                }
            )

            if stream:
                queue.put({"name": "run_id", "value": run_id})
                self.handle_stream(response, input_messages, stream_event=stream_event)
            else:
                self.handle_response(response, input_messages)

            return run_id
        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")
        finally:
            # Decrement depth when exiting ask_model
            self._ask_model_depth -= 1

            # Reset timeline when returning to depth 0 (top-level call complete)
            if self._ask_model_depth == 0:
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Run complete - Resetting timeline"
                    )
                self._global_start_time = None

    def _update_conversation_history(
        self,
        function_call_data: Dict[str, Any],
        function_output: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Updates the conversation history with function call results.
        Formats and appends function output as a user message.

        Args:
            function_call_data: Metadata about the executed function
            function_output: Result from function execution
            input_messages: Current conversation history to update
        """

        # Return in Ollama's tool message format
        # Append tool result in Ollama format with "tool" role
        # Ensure content is a JSON string
        content = (
            Serializer.json_dumps(function_output)
            if not isinstance(function_output, str)
            else function_output
        )
        input_messages.append(
            {
                "role": self.agent["tool_call_role"],
                "content": content,
                "tool_name": function_call_data["name"],
            }
        )

    def to_native_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Neutral (OpenAI-style) → Ollama native (plan §5.5 / A.6).

        - system-role messages are dropped: ``invoke_model`` prepends the
          system instruction from ``agent['instructions']`` itself;
        - assistant ``tool_calls`` convert to Ollama shape with parsed dict
          arguments (neutral carries a JSON string);
        - ``tool`` messages use ``agent['tool_call_role']`` with an explicit
          ``tool_name`` (per ``_update_conversation_history``);
        - ``provider_state`` is not used by this provider and is stripped.
        """
        native: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "")).lower()

            if role == "system":
                continue

            if role == "assistant":
                outgoing: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                }
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    converted = []
                    for tool_call in tool_calls:
                        function = tool_call.get("function") or {}
                        arguments = function.get("arguments") or "{}"
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments or "{}")
                            except json.JSONDecodeError:
                                arguments = {}
                        converted.append(
                            {
                                "function": {
                                    "name": str(function.get("name") or ""),
                                    "arguments": (
                                        arguments
                                        if isinstance(arguments, dict)
                                        else {}
                                    ),
                                }
                            }
                        )
                    outgoing["tool_calls"] = converted
                native.append(outgoing)
                continue

            if role == "tool":
                result = message.get("content") or ""
                native.append(
                    {
                        "role": self.agent["tool_call_role"],
                        "content": (
                            json.dumps(result, ensure_ascii=False)
                            if not isinstance(result, str)
                            else result
                        ),
                        "tool_name": str(message.get("tool_name") or ""),
                    }
                )
                continue

            native.append({"role": "user", "content": message.get("content") or ""})
        return native

    @staticmethod
    def _native_root(base_url: str) -> str:
        """Normalise a model endpoint to Ollama's native API root.

        The llm_engine ``OLLAMA`` builtin called the OpenAI-compatible
        ``{host}/v1/chat/completions``, so existing model records often store
        ``…/v1`` (plan D8, gap G3). The plugin calls native ``/api/chat``;
        stripping a trailing ``/v1`` (and ``/api``) lets those records work
        unchanged, with no data migration.
        """
        root = base_url.strip().rstrip("/")
        for suffix in ("/v1", "/api"):
            if root.endswith(suffix):
                root = root[: -len(suffix)].rstrip("/")
        return root

    def _endpoint_url(self) -> str:
        """Raw-HTTP endpoint (plan A.6 D7): ``{base_url}/api/chat``."""
        return f"{self._endpoint.rstrip('/')}/api/chat"

    def _request_headers(self) -> Dict[str, str]:
        """Per-request auth (plan A.6/D7/CREDS): Authorization Bearer from
        ``ctx.credentials`` — never from ``model_setting['headers']``."""
        headers: Dict[str, str] = {}
        api_key = (self.credentials.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _http_error(
        self, status: Optional[int], detail: str
    ) -> Exception:
        """Adapt a raw-HTTP failure with a duck-typed ``.response`` so the
        adapter's mapping (plan §5.4) classifies it for retry logic."""
        error = Exception(f"Ollama HTTP {status or 'error'}: {detail}")
        try:
            error.response = SimpleNamespace(
                status_code=int(status) if status else None
            )
        except Exception:  # noqa: BLE001 — duck typing is best effort
            pass
        return error

    @staticmethod
    def _usage_from_counts(
        prompt_eval_count: Any, eval_count: Any, done_reason: Any = None
    ) -> Dict[str, Any]:
        """Ollama usage → plan §5.3 shape (Q6: usage was unreported before)."""
        usage: Dict[str, Any] = {
            "prompt_tokens": int(prompt_eval_count or 0),
            "completion_tokens": int(eval_count or 0),
            "total_tokens": int(prompt_eval_count or 0) + int(eval_count or 0),
        }
        if done_reason:
            usage["done_reason"] = done_reason
        return usage

    def _check_retry_limit(self, retry_count: int) -> None:
        """
        Check if retry limit has been exceeded and raise exception if so.

        Args:
            retry_count: Current retry count

        Raises:
            Exception: If retry_count exceeds MAX_RETRIES
        """
        MAX_RETRIES = 5
        if retry_count > MAX_RETRIES:
            error_msg = (
                f"Maximum retry limit ({MAX_RETRIES}) exceeded for empty responses"
            )
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _has_valid_content(self, text: str) -> bool:
        """
        Check if response text contains valid content.

        Args:
            text: Response text to check

        Returns:
            True if text is not None/empty/whitespace-only, False otherwise
        """
        return bool(text and text.strip())

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> None:
        """
        Processes non-streaming model output.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty response → Retry up to 5 times
        3. Valid response → Set final_output

        Args:
            response: Model response object (Ollama format)
            input_messages: Current conversation history
            retry_count: Current retry count (max 5 retries)
        """
        self._check_retry_limit(retry_count)

        self.logger.info("Processing output: %s", response)

        # Ollama response format: {"message": {"role": "assistant", "content": "..."}, "model": "...", ...}
        message = response.get("message", {})

        # Extract and store reasoning/thinking if enabled
        if self.model_setting.get("reasoning", {}).get("enabled") and (
            "thinking" in message and message["thinking"]
        ):
            thinking_text = message["thinking"]
            try:
                if isinstance(thinking_text, str):
                    self.final_output["reasoning_summary"] = thinking_text
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"[handle_response] Captured reasoning: {thinking_text[:100]}..."
                        )
            except Exception as e:
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error(f"Failed to process reasoning: {e}")
                self.final_output["reasoning_summary"] = "Error processing reasoning"

        # Scenario 1: tool calls — single-turn (plan D2): return the
        # requested tool calls to agent_engine instead of executing
        # in-handler. Ollama tool calls carry no id → synthesize one for
        # OpenAI-function pairing (agent_engine tool_call_id pairing).
        if "tool_calls" in message and message["tool_calls"]:
            neutral_tool_calls = []
            for tc in message["tool_calls"]:
                tc = tc if isinstance(tc, dict) else dict(tc)
                if "function" in tc and not isinstance(tc["function"], dict):
                    tc["function"] = dict(tc["function"])
                function = tc["function"]
                arguments = function.get("arguments") or {}
                arguments = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                neutral_tool_calls.append(
                    {
                        "id": f"call-{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": function.get("name") or "",
                            "arguments": arguments or "{}",
                        },
                    }
                )
            self.final_output.update(
                {
                    "message_id": response.get("id")
                    or f"msg-{pendulum.now('UTC').int_timestamp}-{uuid.uuid4().hex[:8]}",
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", "") or "",
                    "tool_calls": neutral_tool_calls,
                    "finish_reason": "tool_calls",
                    "usage": self._usage_from_counts(
                        response.get("prompt_eval_count"),
                        response.get("eval_count"),
                        response.get("done_reason"),
                    ),
                }
            )
            return

        # Scenario 2: Empty response - retry
        content = message.get("content", "")
        if not self._has_valid_content(content):
            self.logger.warning(
                f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
            )
            next_response = self.invoke_model(
                **{"input_messages": input_messages, "stream": False}
            )
            self.handle_response(
                next_response, input_messages, retry_count=retry_count + 1
            )
            return

        # Scenario 3: Valid response - set final output
        timestamp = pendulum.now("UTC").int_timestamp
        # Optimized UUID generation
        message_id = f"msg-ollama-{self.model_setting.get('model')}-{timestamp}-{uuid.uuid4().hex[:8]}"
        self.final_output.update(
            {
                "message_id": message_id,
                "role": message.get("role", "assistant"),
                "content": content,
                "finish_reason": response.get("done_reason") or "stop",
                "usage": self._usage_from_counts(
                    response.get("prompt_eval_count"),
                    response.get("eval_count"),
                ),
            }
        )

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
        retry_count: int = 0,
    ) -> None:
        """
        Processes streaming model responses chunk by chunk.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty stream → Retry up to 5 times
        3. Valid stream → Accumulate and set final_output

        Args:
            response_stream: Iterator of response chunks (Ollama format)
            input_messages: Current conversation history
            stream_event: Event to signal completion
            retry_count: Current retry count (max 5 retries)

        Handles:
            - Accumulating response text
            - Processing JSON vs text formats
            - Handling tool calls in streaming chunks
            - Sending chunks to websocket
            - Signaling completion
        """
        self._check_retry_limit(retry_count)

        message_id = None
        # Use list for efficient string concatenation (performance optimization)
        accumulated_text_parts = []
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        accumulated_tool_calls = []
        received_any_content = False
        # Terminal usage from the final done chunk (plan Q6)
        stream_usage = None
        # Use cached output format type (performance optimization)
        output_format = self.output_format_type
        index = 0

        # Reasoning/thinking tracking variables
        reasoning_no = 0
        reasoning_index = 0
        accumulated_reasoning_parts = []
        accumulated_partial_reasoning_text = ""
        reasoning_started = False

        for chunk in response_stream:
            # Stop streaming if the client disconnected mid-generation.
            if self.is_stream_cancelled():
                self.logger.info(
                    "Stream cancelled (client disconnected); stopping generation."
                )
                break
            # Get the message from the chunk
            message = chunk.get("message", {})

            # Capture terminal usage (plan Q6: usage was unreported before)
            if chunk.get("done"):
                stream_usage = {
                    "prompt_eval_count": chunk.get("prompt_eval_count"),
                    "eval_count": chunk.get("eval_count"),
                    "done_reason": chunk.get("done_reason"),
                }

            # Check for reasoning/thinking in the chunk
            if self.model_setting.get("reasoning", {}).get("enabled") and (
                "thinking" in message and message["thinking"]
            ):
                thinking_chunk = message["thinking"]
                received_any_content = True

                # Start reasoning block if not started
                if not reasoning_started:
                    reasoning_started = True
                    reasoning_index = 0

                    if self.enable_timeline_log and self.logger.isEnabledFor(
                        logging.INFO
                    ):
                        elapsed = self._get_elapsed_time()
                        self.logger.info(
                            f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning started"
                        )

                # Accumulate reasoning text
                accumulated_reasoning_parts.append(thinking_chunk)
                accumulated_partial_reasoning_text += thinking_chunk

                # Process and send reasoning text
                reasoning_index, accumulated_partial_reasoning_text = (
                    self.process_text_content(
                        reasoning_index,
                        accumulated_partial_reasoning_text,
                        output_format,
                        suffix=f"rs#{reasoning_no}",
                    )
                )

            # Check if reasoning block has ended (content starts arriving)
            if reasoning_started and "content" in message and message.get("content"):
                # End reasoning block
                if len(accumulated_partial_reasoning_text) > 0:
                    self.send_data_to_stream(
                        index=reasoning_index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_reasoning_text,
                        suffix=f"rs#{reasoning_no}",
                    )
                    accumulated_partial_reasoning_text = ""
                    reasoning_index += 1

                reasoning_started = False

                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning ended")

            # Check for tool calls in streaming chunks (Ollama v0.8.0+)
            # Convert Pydantic objects to plain dicts so they can be serialized
            # back into messages for the next ollama.chat() call
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    tc = tool_call if isinstance(tool_call, dict) else dict(tool_call)
                    if "function" in tc and not isinstance(tc["function"], dict):
                        tc["function"] = dict(tc["function"])
                    accumulated_tool_calls.append(tc)
                    received_any_content = True

            # Get the content from the chunk
            chunk_content = message.get("content")

            # Skip empty chunks
            if not chunk_content:
                continue

            received_any_content = True

            # Ollama streaming format: {"message": {"role": "assistant", "content": "..."}, "done": false, ...}
            # Generate message_id on first chunk
            if not message_id:
                # Sync index with reasoning_index when starting content after reasoning
                if index == 0 and reasoning_index > 0:
                    index = reasoning_index + 1

                self.send_data_to_stream(
                    index=index,
                    data_format=output_format,
                )
                index += 1

                timestamp = pendulum.now("UTC").int_timestamp
                # Optimized UUID generation
                message_id = f"msg-ollama-{self.model_setting.get('model')}-{timestamp}-{uuid.uuid4().hex[:8]}"

            # Accumulate in list (performance optimization)
            accumulated_text_parts.append(chunk_content)

            if output_format in ["json_object", "json_schema"]:
                accumulated_partial_json += chunk_content
                # Temporarily build accumulated_text for processing
                temp_accumulated_text = "".join(accumulated_text_parts)
                index, temp_accumulated_text, accumulated_partial_json = (
                    self.process_and_send_json(
                        index,
                        temp_accumulated_text,
                        accumulated_partial_json,
                        output_format,
                    )
                )
            else:
                accumulated_partial_text += chunk_content
                # Check if text contains XML-style tags and update format
                index, accumulated_partial_text = self.process_text_content(
                    index, accumulated_partial_text, output_format
                )

        if len(accumulated_partial_text) > 0:
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta=accumulated_partial_text,
            )
            accumulated_partial_text = ""
            index += 1

        # Build final accumulated text from parts (performance optimization)
        final_accumulated_text = "".join(accumulated_text_parts)

        # Store accumulated reasoning summary if present
        if accumulated_reasoning_parts:
            final_reasoning_text = "".join(accumulated_reasoning_parts)
            if self.final_output.get("reasoning_summary"):
                # Accumulate reasoning from multiple rounds (e.g., function calls)
                self.final_output["reasoning_summary"] = (
                    self.final_output["reasoning_summary"] + "\n" + final_reasoning_text
                )
            else:
                self.final_output["reasoning_summary"] = final_reasoning_text

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[handle_stream] Stored reasoning summary: {final_reasoning_text[:100]}..."
                )

        # Scenario 1: accumulated tool calls — single-turn (plan D2):
        # return the requested tool calls to agent_engine instead of
        # executing in-handler (ids synthesized for OpenAI-function pairing).
        if accumulated_tool_calls:
            neutral_tool_calls = []
            for tc in accumulated_tool_calls:
                function = tc.get("function") or {}
                arguments = function.get("arguments") or {}
                arguments = (
                    json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str)
                    else arguments
                )
                neutral_tool_calls.append(
                    {
                        "id": f"call-{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": function.get("name") or "",
                            "arguments": arguments or "{}",
                        },
                    }
                )
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                is_message_end=True,
            )
            self.final_output.update(
                {
                    "message_id": (
                        message_id
                        or f"msg-{pendulum.now('UTC').int_timestamp}-{uuid.uuid4().hex[:8]}"
                    ),
                    "role": "assistant",
                    "content": final_accumulated_text,
                    "tool_calls": neutral_tool_calls,
                    "finish_reason": "tool_calls",
                    "usage": self._usage_from_counts(
                        (stream_usage or {}).get("prompt_eval_count"),
                        (stream_usage or {}).get("eval_count"),
                        (stream_usage or {}).get("done_reason"),
                    ),
                }
            )
            if stream_event:
                stream_event.set()
            return

        # Scenario 2: Empty stream - retry
        if not received_any_content:
            self.logger.warning(
                f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
            )
            next_response = self.invoke_model(
                **{"input_messages": input_messages, "stream": True}
            )
            self.handle_stream(
                next_response,
                input_messages,
                stream_event=stream_event,
                retry_count=retry_count + 1,
            )
            return

        # Scenario 3: Valid stream - finalize
        # Send final message end signal
        self.send_data_to_stream(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        # Set final output (plan §5.3: usage + finish_reason added)
        self.final_output.update(
            {
                "message_id": (
                    message_id
                    if message_id
                    # Optimized UUID generation
                    else f"msg-{pendulum.now('UTC').int_timestamp}-{uuid.uuid4().hex[:8]}"
                ),
                "role": "assistant",
                "content": final_accumulated_text,
                "finish_reason": (stream_usage or {}).get("done_reason") or "stop",
                "usage": self._usage_from_counts(
                    (stream_usage or {}).get("prompt_eval_count"),
                    (stream_usage or {}).get("eval_count"),
                    (stream_usage or {}).get("done_reason"),
                ),
            }
        )

        # Store accumulated_text for backward compatibility
        self.accumulated_text = final_accumulated_text

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
