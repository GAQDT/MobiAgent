#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from device import HdcHarmonyDevice


def save_screenshot(device, out_dir, name):
    path = out_dir / name
    device.screenshot(path)
    print(f"[ok] screenshot -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Smoke test for the hdc-backed OpenHarmony device.")
    parser.add_argument("--hdc-path", default=None, help="Path to hdc/hdc.exe. Defaults to HDC_PATH, PATH, or local SDK.")
    parser.add_argument("--target", default=None, help="Optional hdc target id if multiple devices are connected.")
    parser.add_argument("--out-dir", default="hdc_test_results", help="Directory for screenshots.")
    parser.add_argument("--tap-x", type=int, default=100, help="X coordinate used by the tap test.")
    parser.add_argument("--tap-y", type=int, default=100, help="Y coordinate used by the tap test.")
    parser.add_argument("--swipe", choices=["up", "down", "left", "right"], default="up", help="Swipe direction to test.")
    parser.add_argument("--input-text", default=None, help="Optional text input test. Focus a text field first.")
    parser.add_argument("--package", default=None, help="Optional bundle name to launch, e.g. com.huawei.hmos.settings.")
    parser.add_argument("--skip-actions", action="store_true", help="Only test connection and screenshots.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.hdc_path:
        os.environ["HDC_PATH"] = args.hdc_path

    print("[info] creating HdcHarmonyDevice")
    device = HdcHarmonyDevice(hdc_path=args.hdc_path, target=args.target)
    print(f"[ok] hdc path: {device.hdc_path}")

    print("[info] wakeup")
    device.wakeup()
    save_screenshot(device, out_dir, "01_wakeup.jpeg")

    if args.skip_actions:
        print("[done] skipped action tests")
        return

    print(f"[info] tap at ({args.tap_x}, {args.tap_y})")
    device.click(args.tap_x, args.tap_y)
    save_screenshot(device, out_dir, "02_after_tap.jpeg")

    print(f"[info] swipe {args.swipe}")
    device.swipe(args.swipe)
    save_screenshot(device, out_dir, "03_after_swipe.jpeg")

    if args.input_text is not None:
        print(f"[info] input text: {args.input_text!r}")
        device.input(args.input_text)
        save_screenshot(device, out_dir, "04_after_input.jpeg")
    else:
        print("[skip] input text test; pass --input-text after focusing a text field")

    print("[info] back")
    device.keyevent("BACK")
    save_screenshot(device, out_dir, "05_after_back.jpeg")

    print("[info] home")
    device.keyevent("HOME")
    save_screenshot(device, out_dir, "06_after_home.jpeg")

    if args.package:
        print(f"[info] start app package: {args.package}")
        device.app_start(args.package)
        save_screenshot(device, out_dir, "07_after_app_start.jpeg")
    else:
        print("[skip] app start test; pass --package com.huawei.hmos.settings")

    hierarchy = device.dump_hierarchy()
    hierarchy_path = out_dir / "hierarchy.json"
    hierarchy_path.write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] hierarchy probe -> {hierarchy_path}")
    print("[done] hdc device smoke test completed")


if __name__ == "__main__":
    main()
