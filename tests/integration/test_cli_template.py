import subprocess
import sys


def test_lllm_cli_create(tmp_path):
    project_name = "demo_system"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lllm.cli",
            "create",
            "--name",
            project_name,
            "--template",
            "init_template",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    system_dir = tmp_path / project_name / "system"
    assert system_dir.exists()
    assert (tmp_path / project_name / "lllm.toml").exists()
    assert (system_dir / "system.py").exists()
