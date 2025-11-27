from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".ini",
}

PLACEHOLDERS = {
    "__project_name__": "{name}",
    "{{project_name}}": "{name}",
    "{{PROJECT_NAME}}": "{name_upper}",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="lllm", description="LLLM helper CLI.")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Create a new LLLM project scaffold.")
    create_parser.add_argument(
        "--name",
        default="system",
        help="Name of the project folder to create (default: system).",
    )
    create_parser.add_argument(
        "--template",
        default="init_template",
        help="Template folder inside template/ to use (default: init_template).",
    )

    args = parser.parse_args()
    if args.command == "create":
        try:
            create_project(args.name, args.template)
        except Exception as exc:  # pragma: no cover
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


def create_project(name: str, template_name: str) -> None:
    target_dir = Path.cwd() / name
    if target_dir.exists():
        raise FileExistsError(f"Path '{target_dir}' already exists.")

    template_dir = _resolve_template(template_name)
    if template_dir is None:
        raise FileNotFoundError(f"Template '{template_name}' not found.")

    replacements = {
        key: value.format(name=name, name_upper=name.upper())
        for key, value in PLACEHOLDERS.items()
    }
    _copy_template(template_dir, target_dir, replacements)
    print(f"Created project at {target_dir}")


def _resolve_template(template_name: str) -> Path | None:
    package_root = Path(__file__).resolve().parent.parent
    repo_template = package_root / "template" / template_name
    if repo_template.exists():
        return repo_template
    return None


def _copy_template(src_dir: Path, dst_dir: Path, replacements: Dict[str, str]) -> None:
    for path in src_dir.rglob("*"):
        relative = path.relative_to(src_dir)
        rendered_relative = _render_path(relative, replacements)
        target_path = dst_dir / rendered_relative
        if path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in TEXT_EXTENSIONS:
            content = path.read_text(encoding="utf-8")
            for placeholder, value in replacements.items():
                content = content.replace(placeholder, value)
            target_path.write_text(content, encoding="utf-8")
        else:
            shutil.copy2(path, target_path)


def _render_path(path: Path, replacements: Dict[str, str]) -> Path:
    parts = []
    for part in path.parts:
        new_part = part
        for placeholder, value in replacements.items():
            new_part = new_part.replace(placeholder, value)
        parts.append(new_part)
    return Path(*parts)


if __name__ == "__main__":
    main()
