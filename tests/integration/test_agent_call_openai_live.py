import os

import pytest

from lllm.core.const import APITypes, Roles
from lllm.core.models import Function, Prompt
from lllm.providers.openai import OpenAIProvider
from tests.helpers.agent_utils import make_agent
from tests.helpers.mock_openai import (
    MockOpenAIClient,
    response_text_completion,
    response_tool_call,
    text_completion,
    tool_call_completion,
)


def _build_openai_provider(monkeypatch, run_real: bool, *, chat_scripts=None, response_scripts=None):
    if run_real:
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Set OPENAI_API_KEY to run live OpenAI tests.")
        return OpenAIProvider({})

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chat_payload = list(chat_scripts or [])
    response_payload = list(response_scripts or [])

    def fake_client(*args, **kwargs):
        return MockOpenAIClient(chat_payload.copy(), response_payload.copy())

    import openai

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    return OpenAIProvider({})


def _make_weather_tool():
    calls = []

    def get_forecast(city: str, unit: str = "celsius") -> str:
        summary = f"{city}:{unit}"
        calls.append(summary)
        return summary

    tool = Function(
        name="get_forecast",
        description="Return a normalized weather payload for the requested city.",
        properties={
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        required=["city"],
        strict=False,
    )
    tool.link_function(get_forecast)
    return tool, calls


def test_agent_call_openai_completion_optional_live(monkeypatch, log_config, real_openai_enabled):
    scripts = [text_completion("Task acknowledged: document the repo.")]
    provider = _build_openai_provider(monkeypatch, real_openai_enabled, chat_scripts=scripts)

    system_prompt = Prompt(
        path="live/system",
        prompt="You always respond with 'Task acknowledged: ' followed by the provided task verbatim.",
    )
    task_prompt = Prompt(
        path="live/query",
        prompt="Perform task: {task}",
    )

    agent = make_agent(system_prompt, provider, log_config, model_args={"temperature": 0})
    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"task": "document the repo"})

    response, dialog, interrupts = agent.call(dialog)

    content = (response.content or "").lower()
    assert "task acknowledged:" in content
    assert "document the repo" in content
    assert response.api_type == APITypes.COMPLETION
    assert response.usage
    assert interrupts == []
    assert dialog.tail == response


def test_agent_call_openai_tool_flow_optional_live(monkeypatch, log_config, real_openai_enabled):
    tool, calls = _make_weather_tool()
    scripts = [
        tool_call_completion("get_forecast", {"city": "Lisbon", "unit": "celsius"}),
        text_completion("Tool call handled for Lisbon."),
    ]
    provider = _build_openai_provider(monkeypatch, real_openai_enabled, chat_scripts=scripts)

    system_prompt = Prompt(
        path="live/tool/system",
        prompt="You MUST call the tool to fetch the forecast before replying and always request it in celsius.",
    )
    task_prompt = Prompt(
        path="live/tool/query",
        prompt="City to inspect: {city}",
        functions_list=[tool],
        interrupt_prompt="Tool output: {call_results}. Provide the final answer immediately.",
        interrupt_final_prompt="All tool calls handled. Respond now.",
    )

    agent = make_agent(
        system_prompt,
        provider,
        log_config,
        model_args={"tool_choice": {"type": "function", "function": {"name": "get_forecast"}}},
    )

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"city": "Lisbon"})

    response, dialog, interrupts = agent.call(dialog)

    if real_openai_enabled:
        assert calls and calls[0].startswith("Lisbon:")
    else:
        assert calls == ["Lisbon:celsius"]
    assert len(interrupts) == 1
    assert interrupts[0].name == "get_forecast"
    assert interrupts[0].result == "Lisbon:celsius"
    assert "lisbon" in (response.content or "").lower()
    assert response.api_type == APITypes.COMPLETION


def test_agent_call_openai_response_api_optional_live(monkeypatch, log_config, real_openai_enabled):
    response_scripts = [response_text_completion("Response API acknowledged: minimal summary.")]
    provider = _build_openai_provider(monkeypatch, real_openai_enabled, response_scripts=response_scripts)

    system_prompt = Prompt(
        path="live/response/system",
        prompt="Respond with 'Response API acknowledged: ' plus a concise summary.",
    )
    task_prompt = Prompt(
        path="live/response/query",
        prompt="Summarize: {topic}",
    )

    agent = make_agent(
        system_prompt,
        provider,
        log_config,
        model="gpt-4.1-mini",
        api_type=APITypes.RESPONSE,
        model_args={"max_output_tokens": 200},
    )

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"topic": "portability"})

    response, dialog, interrupts = agent.call(dialog)

    assert response.api_type == APITypes.RESPONSE
    assert "response api acknowledged" in (response.content or "").lower()
    assert response.usage
    assert interrupts == []
    assert dialog.tail == response


def test_agent_call_openai_response_tool_flow_optional_live(monkeypatch, log_config, real_openai_enabled):
    tool, calls = _make_weather_tool()
    response_scripts = [
        response_tool_call("get_forecast", {"city": "Berlin", "unit": "fahrenheit"}),
        response_text_completion("Tool results received for Berlin."),
    ]
    provider = _build_openai_provider(
        monkeypatch,
        real_openai_enabled,
        response_scripts=response_scripts,
    )

    system_prompt = Prompt(
        path="live/response/tool/system",
        prompt="Always invoke the tool before final reply, requesting data in fahrenheit, and then echo the tool output afterwards.",
        functions_list=[tool],
    )
    task_prompt = Prompt(
        path="live/response/tool/query",
        prompt="Fetch data for {city}",
        functions_list=[tool],
        interrupt_prompt="Tool output: {call_results}. Provide the final response immediately.",
        interrupt_final_prompt="All tool calls complete. Reply now.",
    )

    agent = make_agent(
        system_prompt,
        provider,
        log_config,
        model="gpt-4.1-mini",
        api_type=APITypes.RESPONSE,
        model_args={
            "tool_choice": {"type": "function", "function": {"name": "get_forecast"}},
            "max_output_tokens": 200,
        },
    )

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"city": "Berlin"})

    response, dialog, interrupts = agent.call(dialog)

    if real_openai_enabled:
        assert calls and calls[0].startswith("Berlin:")
    else:
        assert calls == ["Berlin:fahrenheit"]
    assert len(interrupts) == 1
    assert interrupts[0].name == "get_forecast"
    assert interrupts[0].result == "Berlin:fahrenheit"
    assert response.api_type == APITypes.RESPONSE
    assert "berlin" in (response.content or "").lower()
    tool_messages = [msg for msg in dialog.messages if msg.role == Roles.USER and msg.creator == "function"]
    assert tool_messages, "Response API tool outputs should surface as user-role entries"
