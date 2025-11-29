import pytest


@pytest.fixture
def log_config(tmp_path):
    return {
        "log_dir": tmp_path.as_posix(),
        "name": "test_agent",
        "log_type": "none",
    }
