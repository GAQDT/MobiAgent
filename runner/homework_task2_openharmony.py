#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework Task 2 runner based on runner/run.py interfaces.

This script uses the unified runner entry helpers and the verified HDC-based
Harmony device backend from device.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from types import SimpleNamespace

from run import (
    create_device,
    execute_batch_tasks,
    execute_single_task,
    load_tasks,
    setup_logging,
)


DEFAULT_API_BASE = "http://0.tcp.jp.ngrok.io:23909/v1"
DEFAULT_MODEL = "MobiMind-Mixed-4B-1031"
DEFAULT_TASK_FILE = os.path.join(os.path.dirname(__file__), "homework_task2_tasks.json")


def find_hdc_executable() -> str:
    env_path = os.environ.get("HDC_PATH")
    if env_path:
        return env_path

    found = shutil.which("hdc")
    if found:
        return found

    sdk_hdc = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "ohos-sdk-full",
        "windows",
        "toolchains",
        "hdc.exe",
    )
    if os.path.exists(sdk_hdc):
        return sdk_hdc

    return "hdc"


def resolve_device_id(device_id: str | None) -> str | None:
    if device_id:
        return device_id

    hdc_path = find_hdc_executable()
    proc = subprocess.run(
        [hdc_path, "list", "targets"],
        text=True,
        capture_output=True,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to get hdc targets.\n"
            f"stdout: {proc.stdout.strip()}\n"
            f"stderr: {proc.stderr.strip()}"
        )

    targets = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and line.strip() != "[Empty]"
    ]
    if not targets:
        raise RuntimeError("No OpenHarmony device found by `hdc list targets`.")
    if len(targets) > 1:
        raise RuntimeError(
            "Multiple hdc targets found: "
            + ", ".join(targets)
            + ". Please pass --device-id explicitly."
        )
    return targets[0]


def normalize_api_base(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return DEFAULT_API_BASE
    if not url.endswith("/v1"):
        return url.rstrip("/") + "/v1"
    return url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenHarmony homework task 2 runner")
    parser.add_argument("--task", type=str, default=None, help="Single task description")
    parser.add_argument(
        "--task-file",
        type=str,
        default=DEFAULT_TASK_FILE,
        help="Task file for batch execution",
    )
    parser.add_argument("--device-id", type=str, default=None, help="hdc target serial")
    parser.add_argument("--output-dir", type=str, default="results_homework_task2", help="Output directory")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="OpenAI-compatible model base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per task")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--draw", action="store_true", default=False, help="Draw action visualizations")
    parser.add_argument("--enable-planning", action="store_true", default=False, help="Enable planner stage")
    parser.add_argument("--use-experience", action="store_true", default=False, help="Enable experience rewriting")
    parser.add_argument(
        "--use-e2e",
        action="store_true",
        default=False,
        help="Use end-to-end mode that requires the model to return bbox directly",
    )
    return parser.parse_args()


def build_runner_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        provider="mobiagent_step",
        device_type="Harmony",
        device_id=cli_args.device_id,
        max_steps=cli_args.max_steps,
        log_level=cli_args.log_level,
        task=cli_args.task,
        task_file=cli_args.task_file,
        output_dir=cli_args.output_dir,
        api_base=normalize_api_base(cli_args.api_base),
        api_key="",
        model=cli_args.model,
        temperature=0.1,
        service_ip="localhost",
        decider_port=8000,
        grounder_port=8001,
        planner_port=8080,
        planner_model=cli_args.model,
        enable_planning=cli_args.enable_planning,
        use_e2e=cli_args.use_e2e,
        decider_model=cli_args.model,
        grounder_model=cli_args.model,
        use_experience=cli_args.use_experience,
        step_delay=2.0,
        draw=cli_args.draw,
    )


def unlock_device(device) -> None:
    logging.info("Waking and unlocking OpenHarmony device before task execution")
    device.wakeup()
    time.sleep(1.0)
    device.swipe("up")
    time.sleep(1.0)
    try:
        device.keyevent("HOME")
        time.sleep(0.5)
    except Exception as e:
        logging.warning(f"Failed to return to home after unlock: {e}")


def main() -> int:
    cli_args = parse_args()
    cli_args.device_id = resolve_device_id(cli_args.device_id)
    args = build_runner_args(cli_args)

    setup_logging(args.log_level)
    print(f"Using OpenHarmony target: {args.device_id}")

    device = create_device(args.device_type, args.device_id)
    unlock_device(device)

    if args.task:
        result = execute_single_task(
            provider=args.provider,
            task_description=args.task,
            device=device,
            output_dir=args.output_dir,
            device_type=args.device_type,
            args=args,
        )
        return 0 if result.get("status") != "error" else 1

    if not os.path.exists(args.task_file):
        raise FileNotFoundError(f"Task file not found: {args.task_file}")

    tasks = load_tasks(args.task_file)
    summary = execute_batch_tasks(
        provider=args.provider,
        tasks=tasks,
        device=device,
        output_dir=args.output_dir,
        device_type=args.device_type,
        args=args,
    )
    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
