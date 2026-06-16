# -*- coding: utf-8 -*-
"""
OpenHarmony plugin for AgentRR.

Goal:
- Keep agent_rr core unchanged.
- Adapt OpenHarmony device + MobiMind decider/grounder to AgentRR's
  Environment / Agent / ActionTree interfaces.
"""

import base64
import hashlib
import io
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_rr.agent.agent import Agent, ReplayLevel
from agent_rr.agent.env import MultiLevelGeneralEnvironment
from agent_rr.action_cache.action import GeneralAgentAction
from agent_rr.action_cache.tree import ActionTree, MatchMode, Task

logger = logging.getLogger(__name__)


def normalize_base_url(url: str) -> str:
    url = str(url).rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def safe_task_key(task_description: str) -> str:
    normalized = re.sub(r"\s+", "_", task_description.strip())[:50]
    safe = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", normalized)
    digest = hashlib.md5(task_description.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


def parse_json_loose(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def normalize_action_name(name: str) -> str:
    name = str(name or "wait").lower()
    if name == "tap":
        return "click"
    if name == "scroll":
        return "swipe"
    if name in {"type", "input_text", "set_text"}:
        return "input"
    if name in {"finished", "complete", "end"}:
        return "done"
    return name


def action_to_dict(action: GeneralAgentAction) -> Dict[str, Any]:
    return {
        "name": action.name,
        "param": action.param or {},
        "extra": action.extra or {},
    }


def action_from_dict(data: Dict[str, Any]) -> GeneralAgentAction:
    return GeneralAgentAction(
        name=data["name"],
        param=data.get("param", {}) or {},
        extra=data.get("extra", {}) or {},
    )


class MobiMindOpenHarmonyAgent(Agent):
    """
    AgentRR Agent adapter.

    Input:
        agent_input = {
            "image": PIL.Image,
            "query": prompt string,
            optional "replay_level"
        }

    Output:
        AgentRR-compatible action dict:
        {
            "name": "click",
            "param": {"target_element": "...", "bbox": [...]},
            "extra": {"reasoning": "...", "decider_raw_output": "..."}
        }
    """

    def __init__(
        self,
        decider_base_url: str,
        grounder_base_url: Optional[str],
        decider_model: str,
        grounder_model: str,
        api_key: str = "0",
    ):
        super().__init__()
        self.decider_client = OpenAI(
            api_key=api_key or "0",
            base_url=normalize_base_url(decider_base_url),
        )
        self.grounder_client = OpenAI(
            api_key=api_key or "0",
            base_url=normalize_base_url(grounder_base_url or decider_base_url),
        )
        self.decider_model = decider_model
        self.grounder_model = grounder_model

        self.model_calls = 0
        self.decider_calls = 0
        self.grounder_calls = 0
        self.generated_actions: List[Dict[str, Any]] = []

    def _image_to_b64(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate(self, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        image: Image.Image = agent_input["image"]
        query: str = agent_input["query"]
        replay_level = agent_input.get("replay_level", ReplayLevel.ALL)

        image_b64 = self._image_to_b64(image)

        # Used only by AgentRR's ReplayLevel.REASONING.
        # You can keep it for compatibility, even if current experiment does not use it.
        if replay_level == ReplayLevel.REASONING:
            self.model_calls += 1
            self.grounder_calls += 1

            response = self.grounder_client.chat.completions.create(
                model=self.grounder_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": query},
                    ],
                }],
                temperature=0,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            obj = parse_json_loose(raw)
            return {
                "name": "click",
                "param": {
                    "target_element": "re-grounded target",
                    "bbox": obj.get("bbox") or obj.get("bbox_2d") or obj.get("bbox-2d"),
                },
                "extra": {
                    "reasoning": "re-grounded cached reasoning",
                    "decider_raw_output": raw,
                },
            }

        # Normal generation: decider first.
        self.model_calls += 1
        self.decider_calls += 1

        response = self.decider_client.chat.completions.create(
            model=self.decider_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": query},
                ],
            }],
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        decider_raw = response.choices[0].message.content
        decider_obj = parse_json_loose(decider_raw)

        reasoning = decider_obj.get("reasoning", "")
        action = normalize_action_name(decider_obj.get("action", "wait"))
        param = decider_obj.get("parameters", {}) or {}

        if action in {"click", "longclick"}:
            target_element = (
                param.get("target_element")
                or param.get("description")
                or param.get("text")
                or "target element"
            )
            bbox = param.get("bbox") or param.get("bbox_2d") or param.get("bbox-2d")

            # If decider does not provide bbox, call grounder.
            if bbox is None:
                grounder_prompt = (
                    "Based on the screenshot, user's intent and the description of the target UI element, "
                    "provide the bounding box of the element using coordinates in the 0-1000 range.\n"
                    f"User's intent: {reasoning}\n"
                    f"Target element's description: {target_element}\n"
                    "Your output should be a JSON object with the following format: "
                    "{\"bbox\": [x1, y1, x2, y2]}"
                )

                self.model_calls += 1
                self.grounder_calls += 1

                grounder_response = self.grounder_client.chat.completions.create(
                    model=self.grounder_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            {"type": "text", "text": grounder_prompt},
                        ],
                    }],
                    temperature=0,
                    max_tokens=128,
                    response_format={"type": "json_object"},
                )

                grounder_raw = grounder_response.choices[0].message.content
                grounder_obj = parse_json_loose(grounder_raw)
                bbox = (
                    grounder_obj.get("bbox")
                    or grounder_obj.get("bbox_2d")
                    or grounder_obj.get("bbox-2d")
                    or grounder_obj.get("point")
                )

            final_param = {
                "target_element": target_element,
                "bbox": bbox,
            }

        elif action == "swipe":
            final_param = {
                "direction": str(param.get("direction", "UP")).upper()
            }

        elif action == "input":
            final_param = {
                "text": param.get("text", param.get("content", ""))
            }

        elif action in {"back", "home", "wait", "done"}:
            final_param = param or {}

        else:
            action = "wait"
            final_param = {"seconds": 1}

        action_dict = {
            "name": action,
            "param": final_param,
            "extra": {
                "reasoning": reasoning,
                "decider_raw_output": decider_raw,
            },
        }

        self.generated_actions.append(action_dict)
        return action_dict


class OpenHarmonyAgentRREnvironment(MultiLevelGeneralEnvironment):
    """
    AgentRR Environment adapter for runner/device.py's Device interface.

    Required by ActionTree:
    - get_agent_input(history, task_description)
    - execute(action)
    """

    def __init__(
        self,
        device,
        agent: MobiMindOpenHarmonyAgent,
        output_dir: str,
        replay_level=ReplayLevel.ALL,
        action_sleep: float = 1.0,
    ):
        super().__init__(agent, replay_level=replay_level)
        self.device = device
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.action_sleep = action_sleep
        self.step = 0
        self.last_screenshot: Optional[Image.Image] = None

        self.executed_actions: List[Dict[str, Any]] = []
        self.replayed_actions = 0
        self.generated_actions = 0
        self._last_agent_generated_len = 0

        self.decider_prompt_fmt = '''You are a phone-use GUI agent controlling an OpenHarmony device. Now your task is "{task}".
Your action history is:
{history}

Please provide the next action based on the screenshot and your action history.
Your action space includes:
- click: parameters target_element, and optionally bbox in the 0-1000 coordinate range.
- swipe: parameter direction, one of UP, DOWN, LEFT, RIGHT.
- input: parameter text.
- back: no parameters.
- home: no parameters.
- wait: optional parameter seconds.
- done: parameters status, usually success.

Return ONLY one JSON object:
{{"reasoning": "...", "action": "click/input/swipe/back/home/wait/done", "parameters": {{...}}}}'''

    def get_screenshot(self) -> Image.Image:
        self.step += 1
        path = self.output_dir / f"agentrr_step_{self.step:03d}.jpg"

        self.device.screenshot(str(path))
        image = Image.open(path).convert("RGB")
        self.last_screenshot = image
        return image

    def _bbox_to_abs_center(self, bbox: Any) -> List[int]:
        """
        Support:
        - [x, y]
        - [x1, y1, x2, y2]
        - {"bbox": [...]}
        - {"point": [...]}
        - {"x": x, "y": y}
        Coordinates are treated as 0-1000 normalized if within 0-1000.
        """
        if bbox is None:
            raise ValueError("bbox is None")

        if isinstance(bbox, dict):
            if "bbox" in bbox:
                return self._bbox_to_abs_center(bbox["bbox"])
            if "point" in bbox:
                return self._bbox_to_abs_center(bbox["point"])
            if "x" in bbox and "y" in bbox:
                return [int(round(float(bbox["x"]))), int(round(float(bbox["y"])))]

        if isinstance(bbox, (list, tuple)) and len(bbox) == 1 and isinstance(bbox[0], (list, tuple)):
            return self._bbox_to_abs_center(bbox[0])

        if not isinstance(bbox, (list, tuple)):
            raise ValueError(f"Unsupported bbox type: {type(bbox)}, value={bbox}")

        if len(bbox) == 2:
            x, y = bbox
            return [int(round(float(x))), int(round(float(y)))]

        if len(bbox) != 4:
            raise ValueError(f"Unsupported bbox format: {bbox}")

        x1, y1, x2, y2 = map(float, bbox)

        if self.last_screenshot is None:
            raise RuntimeError("No screenshot available for bbox conversion")

        w, h = self.last_screenshot.size

        # MobiMind usually returns 0-1000 coordinates.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000:
            x1 = x1 / 1000.0 * w
            x2 = x2 / 1000.0 * w
            y1 = y1 / 1000.0 * h
            y2 = y2 / 1000.0 * h

        x = int(round((x1 + x2) / 2.0))
        y = int(round((y1 + y2) / 2.0))
        return [x, y]

    def execute(self, action: GeneralAgentAction):
        # If the action was not generated in the current loop, it came from ActionTree cache.
        if len(self.agent.generated_actions) == self._last_agent_generated_len:
            self.replayed_actions += 1
        else:
            self.generated_actions += 1
            self._last_agent_generated_len = len(self.agent.generated_actions)

        name = normalize_action_name(action.name)
        param = action.param or {}

        logger.info("AgentRR executing: %s(%s)", name, param)

        if name in {"click", "longclick"}:
            bbox = param.get("bbox")
            if bbox is not None:
                x, y = self._bbox_to_abs_center(bbox)
            elif "coordinate" in param:
                x, y = param["coordinate"]
            else:
                raise ValueError(f"Click action missing bbox/coordinate: {param}")

            if name == "longclick":
                self.device.long_click(x, y)
            else:
                self.device.click(x, y)

        elif name == "swipe":
            self.device.swipe(
                str(param.get("direction", "UP")).lower(),
                param.get("scale", 0.5),
            )

        elif name == "input":
            self.device.input(str(param.get("text", "")))

        elif name == "back":
            self.device.keyevent("BACK")

        elif name == "home":
            self.device.keyevent("HOME")

        elif name == "wait":
            time.sleep(float(param.get("seconds", 1)))

        else:
            logger.warning("Unsupported AgentRR action: %s", name)

        self.executed_actions.append(action_to_dict(action))
        time.sleep(self.action_sleep)


class AgentRRTraceStore:
    """
    Persistent memory layer.

    This is not AgentRR core logic; it only persists successful trajectories
    and rebuilds ActionTree when a new process starts.
    """

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.trace_dir = self.memory_dir / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def load_traces(self) -> List[Dict[str, Any]]:
        traces = []
        for path in sorted(self.trace_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    traces.append(json.load(f))
            except Exception as e:
                logger.warning("Failed to load trace %s: %s", path, e)
        return traces

    def rebuild_tree(self, tree: ActionTree):
        count = 0
        for trace in self.load_traces():
            task_description = trace.get("task_description")
            if not task_description:
                continue

            task = Task(task_description)
            node = tree.root

            for action_data in trace.get("actions", []):
                try:
                    node = node.add_child(action_from_dict(action_data), task)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to add action from trace: %s", e)

        logger.info("AgentRR tree rebuilt from traces: %d actions", count)

    def save_trace(
        self,
        task_description: str,
        actions: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> Optional[str]:
        if not actions:
            logger.warning("Skip saving empty AgentRR trace: %s", task_description)
            return None

        payload = {
            "task_description": task_description,
            "match_mode": "EXACT",
            "actions": actions,
            "meta": meta,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        path = self.trace_dir / f"{safe_task_key(task_description)}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("AgentRR trace saved: %s", path)
        return str(path)


def run_agentrr_oh_task(
    task_description: str,
    device,
    device_type: str,
    output_dir: str,
    args,
) -> Dict[str, Any]:
    """
    Main plugin entry.

    This is the only function run.py needs to call.
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decider_url = getattr(args, "agentrr_decider_base_url", None)
    grounder_url = getattr(args, "agentrr_grounder_base_url", None)

    if not decider_url:
        host = getattr(args, "service_ip", "127.0.0.1")
        port = getattr(args, "decider_port", 8000)
        decider_url = f"http://{host}:{port}"

    if not grounder_url:
        grounder_url = decider_url

    store = AgentRRTraceStore(args.agentrr_memory_dir)

    agent = MobiMindOpenHarmonyAgent(
        decider_base_url=decider_url,
        grounder_base_url=grounder_url,
        decider_model=args.decider_model,
        grounder_model=args.grounder_model,
        api_key=getattr(args, "api_key", "0"),
    )

    env = OpenHarmonyAgentRREnvironment(
        device=device,
        agent=agent,
        output_dir=str(output_dir),
        replay_level=ReplayLevel.ALL,
        action_sleep=float(getattr(args, "agentrr_action_sleep", 1.0)),
    )

    tree = ActionTree(
        env=env,
        agent=agent,
        action_class=GeneralAgentAction,
        done=lambda a: normalize_action_name(a.name) == "done",
        mode=MatchMode.EXACT,
        enable_ui_detection=False,
    )

    # Load JSON traces into the original AgentRR ActionTree.
    store.rebuild_tree(tree)

    # Important: generate shortcuts from existing traces before executing.
    # Otherwise shortcuts generated at the end of a previous run are not
    # available after process restart.
    try:
        tree.generate_shortcuts()
        logger.info("AgentRR shortcuts generated from memory: %d", len(tree.shortcuts))
    except Exception as e:
        logger.warning("AgentRR shortcut generation failed: %s", e)

    status = "success"
    error = None

    try:
        tree.execute(task_description)
    except Exception as e:
        logger.exception("AgentRR OpenHarmony execution failed")
        status = "error"
        error = str(e)

    elapsed = time.time() - start_time

    # If agent.generated_actions is empty, this was a pure cache replay.
    # Do not overwrite memory in that case.
    generated_actions = list(agent.generated_actions)
    trace_actions = generated_actions if generated_actions else list(env.executed_actions)

    if status == "success" and generated_actions:
        store.save_trace(
            task_description=task_description,
            actions=generated_actions,
            meta={
                "device_type": device_type,
                "output_dir": str(output_dir),
                "elapsed_time": elapsed,
                "model_calls": agent.model_calls,
                "decider_calls": agent.decider_calls,
                "grounder_calls": agent.grounder_calls,
                "executed_actions": len(env.executed_actions),
                "generated_actions": env.generated_actions,
                "replayed_actions": env.replayed_actions,
            },
        )

    result = {
        "status": status,
        "error": error,
        "task_description": task_description,
        "device_type": device_type,
        "output_dir": str(output_dir),
        "agentrr_enabled": True,
        "agentrr_plugin": "openharmony",
        "agentrr_match_mode": "EXACT",
        "elapsed_time": elapsed,
        "model_calls": agent.model_calls,
        "decider_calls": agent.decider_calls,
        "grounder_calls": agent.grounder_calls,
        "executed_actions": len(env.executed_actions),
        "generated_actions": env.generated_actions,
        "replayed_actions": env.replayed_actions,
        "cache_hit_rate": (
            env.replayed_actions / len(env.executed_actions)
            if env.executed_actions else 0.0
        ),
        "shortcuts": len(getattr(tree, "shortcuts", [])),
        "trace_actions": trace_actions,
    }

    with open(output_dir / "agentrr_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result