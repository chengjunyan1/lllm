import uuid

import pytest

from lllm.core.models import PROMPT_REGISTRY, Prompt, Message
from lllm.core.const import APITypes, Roles, find_model_card
from lllm.llm import Prompts, register_prompt
from lllm.proxies import BaseProxy, Proxy, PROXY_REGISTRY, ProxyRegistrator


@pytest.fixture
def prompt_registry_cleanup():
    before = set(PROMPT_REGISTRY.keys())
    yield
    for key in list(PROMPT_REGISTRY.keys()):
        if key not in before:
            PROMPT_REGISTRY.pop(key, None)


@pytest.fixture
def proxy_registry_cleanup():
    before = set(PROXY_REGISTRY.keys())
    yield
    for key in list(PROXY_REGISTRY.keys()):
        if key not in before:
            PROXY_REGISTRY.pop(key, None)


def test_prompts_helper_and_handlers(prompt_registry_cleanup):
    path = f"test/{uuid.uuid4().hex}"
    prompt = Prompt(path=path, prompt="Hello!")
    register_prompt(prompt)

    helper = Prompts("test")
    resolved = helper(path.split("/", 1)[1])
    assert resolved.path == path
    assert resolved.interrupt_handler.prompt == prompt.interrupt_prompt
    assert resolved.interrupt_handler_final.prompt == prompt.interrupt_final_prompt


def test_message_cost_uses_model_card():
    usage = {"prompt_tokens": 1000, "completion_tokens": 500, "cached_prompt_tokens": 250}
    msg = Message(
        role=Roles.ASSISTANT,
        content="ok",
        creator="test",
        model="gpt-4o-mini",
        usage=usage,
        api_type=APITypes.COMPLETION,
    )
    assert msg.cost.cost > 0


def test_model_card_classifier_bias():
    card = find_model_card("gpt-4o-mini")
    args = card.make_classifier(["YES", "NO"], strength=15)
    assert args["max_tokens"] == 1
    assert len(args["logit_bias"]) == 2


def test_proxy_registration_and_dispatch(proxy_registry_cleanup):
    path = f"test/proxy/{uuid.uuid4().hex}"

    @ProxyRegistrator(path=path, name="Test Proxy", description="For tests")
    class _TProxy(BaseProxy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def echo(self, payload):
            return {"payload": payload, "cutoff": self.cutoff_date is not None}

    proxy = Proxy(activate_proxies=[path])
    response = proxy(f"{path}.echo", payload=123)
    assert response["payload"] == 123
