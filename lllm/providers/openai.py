import os
import json
import openai
from typing import Any, Dict, Generator, List
from lllm.core.models import Message, Prompt, FunctionCall, AgentException
from lllm.core.const import Roles, Modalities, APITypes, Providers, find_model_card, Features
from lllm.providers.base import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any] = {}):
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

    def call(self, 
             dialog: Any, 
             prompt: Prompt,
             model: str, 
             model_args: Dict[str, Any] = {}, 
             parser_args: Dict[str, Any] = {},
             responder: str = 'assistant', 
             extra: Dict[str, Any] = {}) -> Message:
        
        model_card = find_model_card(model)
        client = self._get_client(model)
        
        # Determine if we are using Chat Completion or Response API (if applicable)
        # For now, following logic in LLMCaller._call_openai
        
        funcs = [func.to_tool(Providers.OPENAI) for func in prompt.functions.values()]
        call_args = model_args.copy()
        
        if prompt.format is None:
            call_fn = client.chat.completions.create
            api_type = APITypes.COMPLETION
        else:
            call_fn = client.beta.chat.completions.parse
            call_args = call_args.copy()
            call_args['response_format'] = prompt.format
            api_type = APITypes.RESPONSE

        if model_card.is_reasoning:
            call_args['temperature'] = call_args.get('temperature', 1)

        # Prepare messages
        openai_messages = self._convert_dialog(dialog)

        completion = call_fn(
            model=model,
            messages=openai_messages,
            tools=funcs if funcs else None, # Only pass tools if there are any
            **call_args
        )
        
        choice = completion.choices[0]
        usage = json.loads(completion.usage.model_dump_json())

        if choice.finish_reason == 'tool_calls':
            role = Roles.TOOL_CALL
            logprobs = None
            parsed = None
            errors = []
            function_calls = [FunctionCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments)
            ) for tool_call in choice.message.tool_calls]   
            content = 'Tool calls:\n\n'+'\n'.join([f'{idx}. {tool_call.function.name}: {tool_call.function.arguments}' for idx, tool_call in enumerate(choice.message.tool_calls)])
        else:
            role = Roles.ASSISTANT
            errors = []
            if prompt.format is None:
                content = choice.message.content
                logprobs = choice.logprobs.content if choice.logprobs is not None else None
                if logprobs is not None:
                    logprobs = [logprob.model_dump() for logprob in logprobs]
                try:
                    parsed = prompt.parser(content, **parser_args) if prompt.parser is not None else None
                except Exception as e: # Catching generic exception as ParseError might be imported differently
                    errors.append(e)
                    parsed = {'raw': content}
            else:
                if choice.message.refusal:
                    raise ValueError(choice.message.refusal)
                content = str(choice.message.parsed.json())
                logprobs = None
                parsed = json.loads(content)
            
            if 'response_format' in call_args and prompt.format is not None:
                 # convert the format uninstantiated class to a json string
                 call_args['response_format'] = prompt.format.model_json_schema()
            function_calls = []

        response = Message(
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
            api_type=api_type,
        )
        return response

    def stream(self, *args, **kwargs):
        raise NotImplementedError("Streaming not yet implemented for OpenAIProvider")
