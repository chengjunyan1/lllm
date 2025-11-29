import json
import types
from typing import Any, Iterable, List, Optional


class MockUsage:
    def __init__(self, data: Optional[dict] = None):
        self._data = data or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cached_prompt_tokens": 0,
        }

    def model_dump_json(self) -> str:
        return json.dumps(self._data)


class MockParsedObject:
    """Mimics the object returned by the parse API for structured outputs."""

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> str:
        return json.dumps(self._payload)


class MockToolFunction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class MockToolCall:
    def __init__(self, call_id: str, fn_name: str, arguments: dict):
        self.id = call_id
        self.function = MockToolFunction(fn_name, arguments)


class MockMessage:
    def __init__(self, content=None, tool_calls=None, parsed=None, refusal=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.parsed = parsed
        self.refusal = refusal


class MockChoice:
    def __init__(self, finish_reason, message, logprobs=None):
        self.finish_reason = finish_reason
        self.message = message
        self.logprobs = logprobs


class MockCompletion:
    def __init__(self, choice: MockChoice, usage: MockUsage):
        self.choices = [choice]
        self.usage = usage


class MockChatAPI:
    def __init__(self, queue: List[Any]):
        self._queue = queue

    def create(self, *args, **kwargs):
        if not self._queue:
            raise AssertionError("No scripted OpenAI chat responses remaining")
        return self._queue.pop(0)


class MockResponseAPI:
    def __init__(self, queue: List[Any]):
        self._queue = queue

    def create(self, *args, **kwargs):
        if not self._queue:
            raise AssertionError("No scripted OpenAI response payloads remaining")
        return self._queue.pop(0)


class MockOpenAIClient:
    def __init__(self, chat_scripts: Iterable[Any] = None, response_scripts: Iterable[Any] = None):
        chat_queue = list(chat_scripts or [])
        response_queue = list(response_scripts or [])

        chat_api = MockChatAPI(chat_queue)
        self.chat = types.SimpleNamespace(completions=chat_api)
        self.beta = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=chat_api)
        )
        self.responses = MockResponseAPI(response_queue)


class MockResponseOutput:
    def __init__(self, type_: str, text: Optional[str] = None, name: Optional[str] = None, arguments: Optional[str] = None, call_id: str = "call_1"):
        self.type = type_
        self.text = text
        self.name = name
        self.arguments = arguments
        self.call_id = call_id
        self.id = call_id


class MockResponse:
    def __init__(self, outputs: Optional[List[MockResponseOutput]] = None, output_text: Optional[str] = None, usage: Optional[MockUsage] = None, reasoning: Any = None):
        self.output = outputs or []
        self.output_text = output_text
        self.usage = usage or MockUsage()
        self.reasoning = reasoning


class MockReasoning:
    def __init__(self, payload: Optional[dict] = None):
        self._payload = payload or {"steps": []}

    def model_dump_json(self) -> str:
        return json.dumps(self._payload)


def tool_call_completion(tool_name: str, arguments: dict, call_id: str = "call_1") -> MockCompletion:
    tool_call = MockToolCall(call_id, tool_name, arguments)
    message = MockMessage(content=None, tool_calls=[tool_call])
    choice = MockChoice(finish_reason="tool_calls", message=message)
    return MockCompletion(choice, MockUsage())


def text_completion(content: str) -> MockCompletion:
    message = MockMessage(content=content, tool_calls=[])
    choice = MockChoice(finish_reason="stop", message=message)
    return MockCompletion(choice, MockUsage())


def parsed_completion(payload: dict) -> MockCompletion:
    message = MockMessage(content=json.dumps(payload), tool_calls=[], parsed=MockParsedObject(payload))
    choice = MockChoice(finish_reason="stop", message=message)
    return MockCompletion(choice, MockUsage())


def response_text_completion(content: str) -> MockResponse:
    outputs = [MockResponseOutput("output_text", text=content)]
    return MockResponse(outputs=outputs, output_text=content)


def response_tool_call(tool_name: str, arguments: dict, call_id: str = "call_r1") -> MockResponse:
    outputs = [MockResponseOutput("function_call", name=tool_name, arguments=json.dumps(arguments), call_id=call_id)]
    return MockResponse(outputs=outputs)


def completion_from_dict(data: dict) -> MockCompletion:
    finish_reason = data.get("finish_reason", "stop")
    usage = MockUsage(data.get("usage"))
    tool_calls = []
    if finish_reason == "tool_calls":
        for call in data.get("tool_calls", []):
            call_id = call.get("id", "call_recorded")
            fn = call.get("function", {})
            tool_calls.append(MockToolCall(call_id, fn.get("name", ""), fn.get("arguments", {})))
        message = MockMessage(content=None, tool_calls=tool_calls)
    else:
        message = MockMessage(content=data.get("message_content"), tool_calls=[])
    choice = MockChoice(finish_reason=finish_reason, message=message)
    return MockCompletion(choice, usage)


def load_recorded_completions(path) -> List[MockCompletion]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [completion_from_dict(entry) for entry in data]
