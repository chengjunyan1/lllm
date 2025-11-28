import uuid

import pytest

from lllm.core.const import Roles
from lllm.core.models import Function, FunctionCall, Message, Prompt
from lllm.providers.base import BaseProvider
from tests.helpers.agent_utils import make_agent


class FakeProvider(BaseProvider):
    """Provider that returns preseeded Message objects for each call."""

    def __init__(self, responses):
        self._responses = list(responses)

    def call(self, dialog, prompt, model, model_args=None, parser_args=None, responder='assistant', extra=None):
        if not self._responses:
            raise AssertionError("FakeProvider received more calls than responses")
        response = self._responses.pop(0)
        return response

    def stream(self, *args, **kwargs):
        raise NotImplementedError


def test_agent_call_returns_message_without_tools(log_config):
    system_prompt = Prompt(path="tests/system", prompt="You are a tester.")
    query_prompt = Prompt(path="tests/query", prompt="User task: {task}")

    provider = FakeProvider(
        [
            Message(
                role=Roles.ASSISTANT,
                creator="assistant",
                content="All done!",
                model="gpt-4o-mini",
            )
        ]
    )

    agent = make_agent(system_prompt, provider, log_config)
    dialog = agent.init_dialog()
    dialog.send_message(query_prompt, {"task": "demo"})

    response, dialog, interrupts = agent.call(dialog)
    assert response.content == "All done!"
    assert interrupts == []
    assert dialog.tail == response


def test_agent_call_executes_registered_function(log_config):
    calls = []

    def _echo(value: str) -> str:
        calls.append(value)
        return f"echo:{value}"

    tool = Function(
        name="echo",
        description="Echo input",
        properties={"value": {"type": "string"}},
        required=["value"],
    )
    tool.link_function(_echo)

    system_prompt = Prompt(path="tests/tool/system", prompt="Use tools when needed.")
    task_prompt = Prompt(path="tests/tool/query", prompt="Task: {task}", functions_list=[tool])

    tool_call_message = Message(
        role=Roles.TOOL_CALL,
        creator="assistant",
        content="Calling echo",
        function_calls=[
            FunctionCall(id=str(uuid.uuid4()), name="echo", arguments={"value": "foo"})
        ],
        model="gpt-4o-mini",
    )
    final_message = Message(
        role=Roles.ASSISTANT,
        creator="assistant",
        content="Tool results processed.",
        model="gpt-4o-mini",
    )

    provider = FakeProvider([tool_call_message, final_message])
    agent = make_agent(system_prompt, provider, log_config)

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"task": "use the tool"})

    response, dialog, interrupts = agent.call(dialog)

    assert response.content == "Tool results processed."
    assert len(interrupts) == 1
    assert calls == ["foo"]
    assert interrupts[0].result == "echo:foo"
