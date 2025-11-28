import queue
import types
from typing import List, Tuple

import pytest
import nbformat

from lllm.sandbox.jupyter import JupyterCellType, JupyterSession, ProgrammingLanguage


@pytest.fixture
def session_dir(tmp_path):
    return tmp_path / "sandbox"


@pytest.fixture
def session_metadata(tmp_path):
    return {
        "project_root": tmp_path.as_posix(),
        "proxy": {
            "activate_proxies": [],
            "cutoff_date": None,
            "deploy_mode": False,
        },
    }


@pytest.fixture(autouse=True)
def patch_kernel(monkeypatch):
    """Stub KernelManager/BlockingKernelClient to avoid launching a real kernel."""

    class DummyClient:
        def __init__(self):
            self.started = False
            self.exec_counter = 0
            self.iopub_msgs = []
            self.shell_msgs = []
            self.result_factory = None

        def start_channels(self):
            self.started = True

        def wait_for_ready(self, timeout=10):
            if not self.started:
                raise RuntimeError("Channels not started")

        def stop_channels(self):
            self.started = False

        def _default_factory(self, msg_id):
            status_msg = {
                "parent_header": {"msg_id": msg_id},
                "header": {"msg_type": "status"},
                "content": {"execution_state": "idle"},
            }
            shell_msg = {
                "parent_header": {"msg_id": msg_id},
                "content": {"status": "ok", "execution_count": 1},
            }
            return [status_msg], shell_msg

        def execute(self, code, store_history=True):
            msg_id = f"msg-{self.exec_counter}"
            self.exec_counter += 1
            factory = self.result_factory or self._default_factory
            iopub_msgs, shell_msg = factory(msg_id)
            self.iopub_msgs.extend(iopub_msgs)
            self.shell_msgs.append(shell_msg)
            return msg_id

        def get_iopub_msg(self, timeout):
            if not self.iopub_msgs:
                raise queue.Empty
            return self.iopub_msgs.pop(0)

        def get_shell_msg(self, timeout):
            if not self.shell_msgs:
                raise queue.Empty
            return self.shell_msgs.pop(0)

    class DummyKernelManager:
        transport = "ipc"

        def __init__(self, *_, **__):
            self._alive = False
            self.kernel_id = "dummy"
            self._client = DummyClient()

        def is_alive(self):
            return self._alive

        def start_kernel(self):
            self._alive = True

        def shutdown_kernel(self, now=True):
            self._alive = False

        def client(self):
            return self._client

    monkeypatch.setattr("lllm.sandbox.jupyter.KernelManager", DummyKernelManager)
    monkeypatch.setattr(
        "lllm.sandbox.jupyter.BlockingKernelClient", DummyClient
    )

    def _noop_run_all(self, *args, **kwargs):
        return 0

    monkeypatch.setattr(
        "lllm.sandbox.jupyter.JupyterSession.run_all_cells", _noop_run_all, raising=False
    )


def test_session_creates_notebook_and_cells(session_dir, session_metadata):
    session_dir.mkdir(parents=True, exist_ok=True)
    js = JupyterSession(
        name="unit",
        dir=session_dir.as_posix(),
        metadata=session_metadata,
        programming_language=ProgrammingLanguage.PYTHON,
    )

    assert js.notebook_file is not None
    nb = nbformat.read(js.notebook_file, as_version=4)
    assert nb.cells[0].source.startswith("# INIT CODE")

    idx = js.append_markdown_cell("## Notes")
    assert isinstance(idx, int)

    nb = nbformat.read(js.notebook_file, as_version=4)
    assert nb.cells[idx].source == "## Notes"


def test_start_and_shutdown_kernel(session_dir, session_metadata):
    session_dir.mkdir(parents=True, exist_ok=True)
    js = JupyterSession(
        name="kernel",
        dir=session_dir.as_posix(),
        metadata=session_metadata,
        programming_language=ProgrammingLanguage.PYTHON,
    )

    started = js.start_kernel()
    assert started is True
    js.shutdown_kernel()
    # shutting down twice should be safe
    js.shutdown_kernel()


def test_directory_tree(session_dir, session_metadata):
    (session_dir / "nested").mkdir(parents=True, exist_ok=True)
    js = JupyterSession(
        name="tree",
        dir=session_dir.as_posix(),
        metadata=session_metadata,
        programming_language=ProgrammingLanguage.PYTHON,
    )
    tree = js.directory_tree
    assert "nested" in tree


def test_run_cell_records_execute_result(session_dir, session_metadata):
    session_dir.mkdir(parents=True, exist_ok=True)
    js = JupyterSession(
        name="runner",
        dir=session_dir.as_posix(),
        metadata=session_metadata,
        programming_language=ProgrammingLanguage.PYTHON,
    )

    cell_index = js.append_code_cell("42")

    def result_factory(msg_id):
        execute_result = {
            "parent_header": {"msg_id": msg_id},
            "header": {"msg_type": "execute_result"},
            "content": {
                "data": {"text/plain": "42"},
                "metadata": {},
                "execution_count": 1,
            },
            "metadata": {},
        }
        status_msg = {
            "parent_header": {"msg_id": msg_id},
            "header": {"msg_type": "status"},
            "content": {"execution_state": "idle"},
        }
        shell_msg = {
            "parent_header": {"msg_id": msg_id},
            "content": {"status": "ok", "execution_count": 1},
        }
        return [execute_result, status_msg], shell_msg

    js.start_kernel()
    js.kernel_client.result_factory = result_factory
    success = js.run_cell(cell_index)
    assert success

    nb = nbformat.read(js.notebook_file, as_version=4)
    outputs = nb.cells[cell_index].outputs
    assert outputs
    assert outputs[0]["data"]["text/plain"] == "42"
