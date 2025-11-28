import os
import json
import openai
from typing import Any, Dict, List, Optional

from lllm.core.models import Message, Prompt, FunctionCall, AgentException, TokenLogprob
from lllm.core.const import Roles, Modalities, APITypes, Providers, Features, find_model_card
from lllm.providers.base import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _ = config or {}
        assert os.getenv('OPENAI_API_KEY') is not None, "OPENAI_API_KEY is not set"
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')) # Preserving env var name
        # Support for other base_urls (e.g. Together AI)
        if os.getenv('TOGETHER_API_KEY') is not None:
            self.together_client = openai.OpenAI(api_key=os.getenv('TOGETHER_API_KEY'), base_url='https://api.together.xyz/v1')
        else:
            self.together_client = None
            print("TOGETHER_API_KEY is not set, cannot use Together AI models")

    def _get_client(self, model: str):
        model_card = find_model_card(model)
        if model_card.base_url is not None:
            if 'together' in model_card.base_url:
                return self.together_client
            else:
                # Generic base_url support could be added here
                return openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=model_card.base_url)
        return self.client

    def _convert_dialog(self, dialog: Any) -> List[Dict[str, Any]]:
        """Convert internal Dialog state into OpenAI-compatible messages."""
        messages: List[Dict[str, Any]] = []
        for message in dialog.messages:
            if message.role in (Roles.ASSISTANT, Roles.TOOL_CALL):
                assistant_entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.function_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id": fc.id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.arguments),
                            },
                        }
                        for fc in message.function_calls
                    ]
                messages.append(assistant_entry)
                continue

            if message.role == Roles.TOOL:
                tool_call_id = message.extra.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError(
                        "Tool call id is not found in the message extra for tool message: "
                        f"{message}"
                    )
                messages.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": tool_call_id,
                    }
                )
                continue

            if message.modality == Modalities.IMAGE:
                content_parts = []
                if "caption" in message.extra:
                    content_parts.append({"type": "text", "text": message.extra["caption"]})
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{message.content}"}}
                )
                messages.append({"role": message.role.openai, "content": content_parts})
                continue

            if message.modality == Modalities.TEXT:
                messages.append({"role": message.role.openai, "content": message.content})
                continue

            raise ValueError(f"Unsupported modality: {message.modality}")

        return messages

    def _call_chat_api(
        self,
        dialog: Any,
        prompt: Prompt,
        model: str,
        model_card,
        client,
        payload_args: Dict[str, Any],
        parser_args: Dict[str, Any],
        responder: str,
        extra: Dict[str, Any],
    ) -> Message:
        tools = self._build_tools(prompt)
        call_args = dict(payload_args)

        if prompt.format is None:
            call_fn = client.chat.completions.create
        else:
            call_fn = client.beta.chat.completions.parse
            call_args['response_format'] = prompt.format

        if model_card.is_reasoning:
            call_args['temperature'] = call_args.get('temperature', 1)

        completion = call_fn(
            model=model,
            messages=self._convert_dialog(dialog),
            tools=tools if tools else None,
            **call_args,
        )

        choice = completion.choices[0]
        usage = json.loads(completion.usage.model_dump_json())

        if choice.finish_reason == 'tool_calls':
            role = Roles.TOOL_CALL
            logprobs = None
            parsed = None
            errors: List[Exception] = []
            function_calls = [
                FunctionCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in choice.message.tool_calls
            ]
            content = 'Tool calls:\n\n' + '\n'.join(
                [
                    f'{idx}. {tool_call.function.name}: {tool_call.function.arguments}'
                    for idx, tool_call in enumerate(choice.message.tool_calls)
                ]
            )
        else:
            role = Roles.ASSISTANT
            errors = []
            function_calls = []

            if prompt.format is None:
                content = choice.message.content
                raw_logprobs = choice.logprobs.content if choice.logprobs is not None else None
                if raw_logprobs is not None:
                    converted = []
                    for logprob in raw_logprobs:
                        payload = logprob.model_dump() if hasattr(logprob, "model_dump") else logprob
                        converted.append(TokenLogprob.model_validate(payload))
                    logprobs = converted
                else:
                    logprobs = None
                try:
                    parsed = prompt.parser(content, **parser_args) if prompt.parser is not None else None
                except Exception as exc:
                    errors.append(exc)
                    parsed = {'raw': content}
            else:
                if choice.message.refusal:
                    raise ValueError(choice.message.refusal)
                content = str(choice.message.parsed.json())
                parsed = json.loads(content)
                logprobs = None

            if 'response_format' in call_args and prompt.format is not None:
                call_args['response_format'] = prompt.format.model_json_schema()

        return Message(
            role=role,
            raw_response=completion,
            creator=responder,
            function_calls=function_calls,
            content=content,
            logprobs=logprobs or [],
            model=model,
            model_args=call_args,
            usage=usage,
            parsed=parsed or {},
            extra=extra,
            execution_errors=errors,
            api_type=APITypes.COMPLETION,
        )

    def _call_response_api(
        self,
        dialog: Any,
        prompt: Prompt,
        model: str,
        model_card,
        client,
        payload_args: Dict[str, Any],
        parser_args: Dict[str, Any],
        responder: str,
        extra: Dict[str, Any],
    ) -> Message:
        if prompt.format is not None:
            raise ValueError("Response API does not support structured output. Remove 'format' or use the completion API.")

        tools = self._build_tools(prompt)
        if prompt.allow_web_search and Features.WEB_SEARCH in model_card.features:
            tools.append({"type": "web_search_preview"})
        if prompt.computer_use_config and Features.COMPUTER_USE in model_card.features:
            cfg = prompt.computer_use_config
            tools.append(
                {
                    "type": "computer_use_preview",
                    "display_width": cfg.get("display_width", 1280),
                    "display_height": cfg.get("display_height", 800),
                    "environment": cfg.get("environment", "browser"),
                }
            )

        call_args = dict(payload_args)
        max_output_tokens = call_args.pop('max_output_tokens', call_args.pop('max_completion_tokens', 32000))
        truncation = call_args.pop('truncation', 'auto')
        tool_choice = call_args.pop('tool_choice', 'auto')

        response = client.responses.create(
            model=model,
            input=self._convert_dialog(dialog),
            tools=tools if tools else None,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            truncation=truncation,
            **call_args,
        )

        usage = json.loads(response.usage.model_dump_json())
        outputs = getattr(response, "output", []) or []
        function_calls: List[FunctionCall] = []

        for item in outputs:
            if getattr(item, "type", None) == "function_call":
                arguments = getattr(item, "arguments", "{}")
                try:
                    parsed_args = json.loads(arguments)
                except Exception:
                    parsed_args = {}
                function_calls.append(
                    FunctionCall(
                        id=getattr(item, "call_id", getattr(item, "id", "tool_call")),
                        name=getattr(item, "name", "function"),
                        arguments=parsed_args,
                    )
                )

        logprobs = None
        errors: List[Exception] = []
        if function_calls:
            role = Roles.TOOL_CALL
            parsed = None
            content = 'Tool calls:\n\n' + '\n'.join(
                [f'{idx}. {call.name}: {json.dumps(call.arguments)}' for idx, call in enumerate(function_calls)]
            )
        else:
            role = Roles.ASSISTANT
            content = getattr(response, "output_text", None)
            if not content:
                text_chunks = []
                for item in outputs:
                    if getattr(item, "type", None) == "output_text":
                        chunk = getattr(item, "text", None)
                        if chunk:
                            text_chunks.append(chunk)
                content = '\n'.join(text_chunks).strip()
            try:
                parsed = prompt.parser(content, **parser_args) if prompt.parser is not None else None
            except Exception as exc:
                errors.append(exc)
                parsed = {'raw': content}

        extra_payload = dict(extra)
        reasoning = getattr(response, "reasoning", None)
        if reasoning is not None:
            try:
                extra_payload['reasoning'] = reasoning.model_dump_json()
            except Exception:
                extra_payload['reasoning'] = str(reasoning)
        extra_payload.setdefault('api_type', APITypes.RESPONSE.value)

        message = Message(
            role=role,
            raw_response=response,
            creator=responder,
            function_calls=function_calls,
            content=content,
            logprobs=logprobs or [],
            model=model,
            model_args=call_args,
            usage=usage,
            parsed=parsed or {},
            extra=extra_payload,
            execution_errors=errors,
            api_type=APITypes.RESPONSE,
        )
        return message

    def call(
        self,
        dialog: Any,
        prompt: Prompt,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = 'assistant',
        extra: Optional[Dict[str, Any]] = None,
        api_type: APITypes = APITypes.COMPLETION,
    ) -> Message:
        model_card = find_model_card(model)
        client = self._get_client(model)
        payload_args = dict(model_args) if model_args else {}
        parser_args = dict(parser_args) if parser_args else {}
        extra_payload = dict(extra) if extra else {}

        if api_type == APITypes.RESPONSE:
            return self._call_response_api(
                dialog,
                prompt,
                model,
                model_card,
                client,
                payload_args,
                parser_args,
                responder,
                extra_payload,
            )
        return self._call_chat_api(
            dialog,
            prompt,
            model,
            model_card,
            client,
            payload_args,
            parser_args,
            responder,
            extra_payload,
        )

    def stream(self, *args, **kwargs):
        raise NotImplementedError("Streaming not yet implemented for OpenAIProvider")

    def _build_tools(self, prompt: Prompt) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for func in prompt.functions.values():
            tool = func.to_tool(Providers.OPENAI)
            if tool:
                tools.append(tool)
        for server in prompt.mcp_servers.values():
            tool = server.to_tool(Providers.OPENAI)
            if tool:
                tools.append(tool)
        return tools
