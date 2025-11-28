import json
import types


class MockUsage:
    def __init__(self, data=None):
        self._data = data or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cached_prompt_tokens": 0,
        }

    def model_dump_json(self):
        return json.dumps(self._data)


class MockToolFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = json.dumps(arguments)


class MockToolCall:
    def __init__(self, call_id, fn_name, arguments):
        self.id = call_id
        self.function = MockToolFunction(fn_name, arguments)


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


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
    def __init__(self, queue):
        self._queue = queue

    def create(self, *args, **kwargs):
        if not self._queue:
            raise AssertionError("No scripted OpenAI responses remaining")
        return self._queue.pop(0)


class MockChatCompletions:
    def __init__(self, queue):
        self._api = MockChatAPI(queue)

    def create(self, *args, **kwargs):
        return self._api.create(*args, **kwargs)


class MockOpenAIClient:
    def __init__(self, scripts):
        # scripts must be a list so they can be popped in order
        self._queue = list(scripts)
        chat_api = MockChatAPI(self._queue)
        self.chat = types.SimpleNamespace(completions=chat_api)
        self.beta = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=chat_api)
        )


def tool_call_completion(tool_name: str, arguments: dict, call_id: str = "call_1") -> MockCompletion:
    tool_call = MockToolCall(call_id, tool_name, arguments)
    message = MockMessage(content=None, tool_calls=[tool_call])
    choice = MockChoice(finish_reason="tool_calls", message=message)
    return MockCompletion(choice, MockUsage())


def text_completion(content: str) -> MockCompletion:
    message = MockMessage(content=content, tool_calls=[])
    choice = MockChoice(finish_reason="stop", message=message)
    return MockCompletion(choice, MockUsage())
