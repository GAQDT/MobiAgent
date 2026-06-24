from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from PIL import Image


MOBIAGENT_ROOT = Path(__file__).resolve().parents[1]
MOBIAGENT_RUNNER = MOBIAGENT_ROOT / "runner"
PROMPT_DIR = MOBIAGENT_RUNNER / "providers" / "mobiagent" / "prompts"
if str(MOBIAGENT_RUNNER) not in sys.path:
    sys.path.insert(0, str(MOBIAGENT_RUNNER))


def load_prompt(name: str) -> str:
    with open(PROMPT_DIR / name, "r", encoding="utf-8") as f:
        return f.read()


class MobiAgentBridge:
    def __init__(
        self,
        decider_api_base: str,
        grounder_api_base: str,
        decider_model: str,
        grounder_model: str,
        use_e2e: bool = False,
    ) -> None:
        self.decider_client = OpenAI(api_key="0", base_url=self._normalize_api_base(decider_api_base))
        self.grounder_client = OpenAI(api_key="0", base_url=self._normalize_api_base(grounder_api_base))
        self.decider_model = decider_model
        self.grounder_model = grounder_model
        self.use_e2e = use_e2e

        if use_e2e:
            self.decider_prompt = load_prompt("e2e_qwen3.md")
            self.grounder_prompt_bbox = ""
        else:
            self.decider_prompt = load_prompt("decider_v2.md")
            self.grounder_prompt_bbox = load_prompt("grounder_qwen3_bbox.md")

    def decide(self, task: str, history: List[str], image_base64: str) -> Dict[str, Any]:
        img = self._open_image(image_base64)
        prompt = self.decider_prompt.format(
            task=task,
            history="\n".join(f"{idx}. {item}" for idx, item in enumerate(history, 1)) or "(No history)",
        )
        decider = self._call_decider(image_base64, prompt)
        reasoning = decider.get("reasoning", "")
        action = decider.get("action", "wait")
        params = decider.get("parameters", {}) or {}

        if action == "done":
            return {
                "action": {"id": self._action_id(), "action": "done", "status": params.get("status", "success")},
                "reasoning": reasoning,
                "raw": decider,
            }
        if action == "wait":
            return {
                "action": {"id": self._action_id(), "action": "wait", "duration": params.get("duration", 1000)},
                "reasoning": reasoning,
                "raw": decider,
            }
        if action == "input":
            return {
                "action": {"id": self._action_id(), "action": "input", "text": params.get("text", "")},
                "reasoning": reasoning,
                "raw": decider,
            }
        if action == "swipe":
            mapped = self._map_swipe(params, img)
            mapped["id"] = self._action_id()
            return {"action": mapped, "reasoning": reasoning, "raw": decider}
        if action == "click":
            point = self._resolve_click(params, reasoning, image_base64, img)
            return {
                "action": {"id": self._action_id(), "action": "click", "x": point[0], "y": point[1]},
                "reasoning": reasoning,
                "raw": decider,
            }

        return {
            "action": {"id": self._action_id(), "action": "wait", "duration": 1000},
            "reasoning": f"Unsupported decider action {action}; waiting.",
            "raw": decider,
        }

    def _call_decider(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        response = self.decider_client.chat.completions.create(
            model=self.decider_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.1,
            timeout=60,
            max_tokens=512,
            response_format={"type": "json_object"},
        ).choices[0].message.content
        return self._parse_json(response)

    def _call_grounder(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        response = self.grounder_client.chat.completions.create(
            model=self.grounder_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0,
            timeout=60,
            max_tokens=256,
            response_format={"type": "json_object"},
        ).choices[0].message.content
        return self._parse_json(response)

    def _resolve_click(self, params: Dict[str, Any], reasoning: str, image_base64: str, img: Image.Image) -> List[int]:
        if self.use_e2e:
            bbox = params.get("bbox")
            if not bbox:
                raise ValueError("E2E decider did not return bbox")
            x1, y1, x2, y2 = self._convert_coords(bbox, img.width, img.height, True)
            return [(x1 + x2) // 2, (y1 + y2) // 2]

        target = params.get("target_element", "")
        prompt = self.grounder_prompt_bbox.format(reasoning=reasoning, description=target)
        grounded = self._call_grounder(image_base64, prompt)
        bbox = None
        for key, value in grounded.items():
            if key.lower() in {"bbox", "bbox_2d", "bbox-2d", "bbox2d"}:
                bbox = value
                break
        if bbox is None:
            raise ValueError(f"Grounder response missing bbox: {grounded}")
        x1, y1, x2, y2 = self._convert_coords(bbox, img.width, img.height, True)
        return [(x1 + x2) // 2, (y1 + y2) // 2]

    def _map_swipe(self, params: Dict[str, Any], img: Image.Image) -> Dict[str, Any]:
        if params.get("start_coords") and params.get("end_coords"):
            start = self._convert_coords(params["start_coords"], img.width, img.height, False)
            end = self._convert_coords(params["end_coords"], img.width, img.height, False)
            return {"action": "swipe", "from": start, "to": end, "duration": 500}

        direction = params.get("direction", "up")
        cx, cy = img.width // 2, img.height // 2
        delta = int(min(img.width, img.height) * 0.35)
        vectors = {
            "up": ([cx, cy + delta], [cx, cy - delta]),
            "down": ([cx, cy - delta], [cx, cy + delta]),
            "left": ([cx + delta, cy], [cx - delta, cy]),
            "right": ([cx - delta, cy], [cx + delta, cy]),
        }
        start, end = vectors.get(direction, vectors["up"])
        return {"action": "swipe", "from": start, "to": end, "duration": 500}

    def _open_image(self, image_base64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(image_base64)))

    def _convert_coords(self, coords: List[int], width: int, height: int, is_bbox: bool) -> List[int]:
        if is_bbox:
            x1, y1, x2, y2 = coords
            return [int(x1 / 1000 * width), int(y1 / 1000 * height), int(x2 / 1000 * width), int(y2 / 1000 * height)]
        x, y = coords
        return [int(x / 1000 * width), int(y / 1000 * height)]

    def _parse_json(self, text: Optional[str]) -> Dict[str, Any]:
        if not text:
            raise ValueError("empty model response")
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _normalize_api_base(self, url: str) -> str:
        url = url.rstrip("/")
        return url if url.endswith("/v1") else f"{url}/v1"

    def _action_id(self) -> str:
        return f"act-{int(time.time() * 1000)}"
