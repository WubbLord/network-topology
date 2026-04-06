from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Callable
from typing import Any

from .yaml_utils import load_yaml

MappingRunner = Callable[[str, str], Any]

_registered_runner: MappingRunner | None = None


def register_mapping_runner(runner: MappingRunner) -> None:
    global _registered_runner
    _registered_runner = runner


def clear_mapping_runner() -> None:
    global _registered_runner
    _registered_runner = None


def run_mapping(arch_yaml_path: str, workload_yaml_path: str) -> Any:
    if _registered_runner is not None:
        return _registered_runner(arch_yaml_path, workload_yaml_path)

    command_template = os.environ.get("ACCELFORGE_MAPPER_CMD")
    if command_template:
        return _run_external_command(command_template, arch_yaml_path, workload_yaml_path)

    raise RuntimeError(
        "No AccelForge mapper is configured. Register a Python runner with "
        "register_mapping_runner(...) or set ACCELFORGE_MAPPER_CMD."
    )


def _run_external_command(command_template: str, arch_yaml_path: str, workload_yaml_path: str) -> Any:
    command = command_template.format(
        arch=shlex.quote(arch_yaml_path),
        workload=shlex.quote(workload_yaml_path),
    )
    completed = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Mapper command failed with return code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )

    payload = completed.stdout.strip()
    if not payload:
        raise RuntimeError("Mapper command completed successfully but produced no stdout payload.")

    if payload.startswith("{") or payload.startswith("["):
        return json.loads(payload)

    if os.path.exists(payload):
        if payload.endswith((".yaml", ".yml")):
            return load_yaml(payload)
        with open(payload, "r", encoding="utf-8") as handle:
            return json.load(handle)

    raise RuntimeError(
        "Mapper command output was neither inline JSON nor a readable JSON/YAML file path."
    )
