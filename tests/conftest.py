import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--real-openai",
        action="store_true",
        default=False,
        help="Run tests that hit the live OpenAI APIs (requires OPENAI_API_KEY).",
    )


def _real_api_flag_from_env() -> bool:
    value = os.getenv("LLLM_RUN_REAL_API_TESTS", "")
    return value.lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session")
def real_openai_enabled(pytestconfig) -> bool:
    """Flag indicating whether live OpenAI API calls should be executed."""
    return pytestconfig.getoption("--real-openai") or _real_api_flag_from_env()


@pytest.fixture
def log_config(tmp_path):
    return {
        "log_dir": tmp_path.as_posix(),
        "name": "test_agent",
        "log_type": "none",
    }
