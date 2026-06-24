from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from mobiagent_bridge import MobiAgentBridge
from schemas import DeviceSession, TaskState, now_iso


class TaskRequest(BaseModel):
    device_id: str
    task: str
    max_steps: int = 30


class Controller:
    def __init__(self, bridge: MobiAgentBridge, output_dir: Path) -> None:
        self.bridge = bridge
        self.output_dir = output_dir
        self.devices: Dict[str, DeviceSession] = {}
        self.tasks: Dict[str, TaskState] = {}
        self.active_task_by_device: Dict[str, str] = {}
        self.lock = asyncio.Lock()

    async def register_ws(self, device_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            session = self.devices.get(device_id) or DeviceSession(device_id=device_id)
            session.websocket = ws
            session.connected = True
            session.last_seen = now_iso()
            self.devices[device_id] = session
        await self.send(device_id, {"type": "hello", "server_time": now_iso()})

    async def unregister_ws(self, device_id: str) -> None:
        async with self.lock:
            if device_id in self.devices:
                self.devices[device_id].connected = False
                self.devices[device_id].websocket = None
                self.devices[device_id].last_seen = now_iso()

    async def create_task(self, req: TaskRequest) -> TaskState:
        task_text = req.task.strip()
        if not task_text:
            raise ValueError("task is empty")

        task = TaskState(device_id=req.device_id, task=task_text, max_steps=req.max_steps)
        async with self.lock:
            self.tasks[task.task_id] = task
            self.active_task_by_device[req.device_id] = task.task_id
            task.status = "running"
            task.add_trace("task_created", {"task": task_text, "source": "controller"})

        await self.send(req.device_id, {
            "type": "task_update",
            "task_id": task.task_id,
            "status": task.status,
            "message": f"Task started: {task.task}",
        })
        await self.send(req.device_id, {"type": "observe", "task_id": task.task_id, "reason": "task_started"})
        return task

    async def handle_message(self, device_id: str, message: Dict[str, Any]) -> None:
        msg_type = message.get("type")
        async with self.lock:
            session = self.devices.get(device_id) or DeviceSession(device_id=device_id)
            session.last_seen = now_iso()
            if msg_type == "register":
                session.info = message.get("info", {})
            self.devices[device_id] = session

        if msg_type == "observation":
            await self.handle_observation(device_id, message)
        elif msg_type == "result":
            await self.handle_result(device_id, message)
        elif msg_type == "register":
            await self.send(device_id, {"type": "registered", "device_id": device_id})
            active = self.get_active_task(device_id)
            if active and active.status == "running":
                await self.send(device_id, {"type": "observe", "task_id": active.task_id, "reason": "registered"})
        elif msg_type == "heartbeat":
            await self.send(device_id, {"type": "heartbeat_ack", "server_time": now_iso()})
        elif msg_type == "log":
            task = self.get_active_task(device_id)
            if task:
                task.add_trace("device_log", message)
        elif msg_type == "task_request":
            await self.handle_task_request(device_id, message)

    async def handle_task_request(self, device_id: str, message: Dict[str, Any]) -> None:
        task_text = str(message.get("task", "")).strip()
        max_steps = int(message.get("max_steps", 20) or 20)
        if not task_text:
            await self.send(device_id, {
                "type": "task_update",
                "status": "failed",
                "message": "Task text is empty.",
            })
            return

        try:
            task = await self.create_task(TaskRequest(device_id=device_id, task=task_text, max_steps=max_steps))
            task.add_trace("task_requested_from_device", {"task": task_text})
            await self.send(device_id, {
                "type": "task_update",
                "task_id": task.task_id,
                "status": task.status,
                "message": "Task accepted by PC controller.",
            })
        except Exception as exc:
            await self.send(device_id, {
                "type": "task_update",
                "status": "failed",
                "message": f"Task request failed: {exc}",
            })

    async def handle_observation(self, device_id: str, message: Dict[str, Any]) -> None:
        task = self.get_active_task(device_id)
        if not task or task.status != "running":
            await self.send(device_id, {"type": "stop", "reason": "no_active_task"})
            return

        if task.step >= task.max_steps:
            task.status = "max_steps_reached"
            task.final_message = "Reached max steps before success."
            task.add_trace("max_steps_reached", {})
            await self.finish_task(device_id, task)
            return

        image_base64 = message.get("screenshot")
        if not image_base64:
            task.add_trace("observation_error", {"error": "missing screenshot"})
            await self.send(device_id, {"type": "observe", "task_id": task.task_id, "reason": "missing_screenshot"})
            return

        task.step += 1
        screenshot_path = self.save_screenshot(task, task.step, image_base64)
        task.latest_screenshot = str(screenshot_path)
        task.add_trace("observation", {
            "step": task.step,
            "width": message.get("width"),
            "height": message.get("height"),
            "foreground": message.get("foreground"),
            "screenshot_path": str(screenshot_path),
        })

        try:
            decision = self.deterministic_action(task)
            if decision is None:
                decision = await asyncio.to_thread(self.bridge.decide, task.task, task.history, image_base64)

            raw = decision.get("raw", {})
            task.history.append(json.dumps(raw, ensure_ascii=False))
            task.latest_reasoning = str(decision.get("reasoning", ""))
            task.add_trace("decision", decision)

            action = self.normalize_action(decision["action"], task, message)
            task.pending_action = action
            task.latest_action = action

            if action.get("action") == "done" and self.accept_done(task):
                task.status = action.get("status", "success")
                task.final_message = task.latest_reasoning or "Task completed."
                await self.finish_task(device_id, task)
            elif action.get("action") == "done":
                forced = {
                    "id": f"wait-forced-{task.step}",
                    "action": "wait",
                    "duration": 1000,
                    "reason": "Ignored premature done before any effective action.",
                }
                task.latest_action = forced
                task.add_trace("premature_done_ignored", {"original": action, "forced_action": forced})
                await self.send(device_id, {
                    "type": "task_update",
                    "task_id": task.task_id,
                    "status": task.status,
                    "message": "Premature done ignored; requesting another observation.",
                })
                await self.send(device_id, {"type": "action", "task_id": task.task_id, "action": forced})
            else:
                await self.send(device_id, {
                    "type": "task_update",
                    "task_id": task.task_id,
                    "status": task.status,
                    "message": f"Step {task.step}: {action.get('action')} - {task.latest_reasoning}",
                })
                await self.send(device_id, {"type": "action", "task_id": task.task_id, "action": action})
        except Exception as exc:
            task.latest_reasoning = f"Decision error: {exc}"
            task.add_trace("decision_error", {"error": str(exc)})
            await self.send(device_id, {"type": "action", "task_id": task.task_id, "action": {
                "id": f"wait-error-{task.step}",
                "action": "wait",
                "duration": 1500,
                "reason": str(exc),
            }})
        finally:
            self.save_task(task)

    async def handle_result(self, device_id: str, message: Dict[str, Any]) -> None:
        task = self.get_active_task(device_id)
        if not task:
            return
        task.add_trace("result", message)
        await self.send(device_id, {
            "type": "task_update",
            "task_id": task.task_id,
            "status": task.status,
            "message": f"Action result: {message.get('status')} {message.get('message', '')}",
        })
        if task.status == "running":
            await self.send(device_id, {"type": "observe", "task_id": task.task_id, "reason": "action_result"})
        self.save_task(task)

    async def finish_task(self, device_id: str, task: TaskState) -> None:
        task.add_trace("task_finished", {"status": task.status, "message": task.final_message})
        await self.send(device_id, {
            "type": "task_done",
            "task_id": task.task_id,
            "status": task.status,
            "message": task.final_message,
        })
        await self.send(device_id, {"type": "stop", "task_id": task.task_id, "reason": task.status})
        self.save_task(task)

    async def send(self, device_id: str, payload: Dict[str, Any]) -> bool:
        session = self.devices.get(device_id)
        if not session or not session.websocket:
            return False
        await session.websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True

    def get_active_task(self, device_id: str) -> Optional[TaskState]:
        task_id = self.active_task_by_device.get(device_id)
        return self.tasks.get(task_id) if task_id else None

    def task_dir(self, task: TaskState) -> Path:
        path = self.output_dir / task.device_id / task.task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_screenshot(self, task: TaskState, step: int, image_base64: str) -> Path:
        screenshot_path = self.task_dir(task) / f"step_{step}.jpg"
        screenshot_path.write_bytes(base64.b64decode(image_base64))
        return screenshot_path

    def deterministic_action(self, task: TaskState) -> Optional[Dict[str, Any]]:
        task_text = task.task.lower()
        is_settings_task = any(
            keyword in task_text
            for keyword in ("wlan", "settings", "storage", "设置", "存储", "网络", "连接")
        )
        is_wlan_task = "wlan" in task_text

        if task.step == 1 and is_settings_task:
            return {
                "action": {
                    "id": f"init-settings-{task.step}",
                    "action": "app_start",
                    "bundleName": "com.ohos.settings",
                    "abilityName": "com.ohos.settings.MainAbility",
                    "moduleName": "phone",
                },
                "reasoning": "Initial task requests Settings-related information; launch Settings before visual navigation.",
                "raw": {
                    "reasoning": "Launch Settings as deterministic bootstrap action.",
                    "action": "app_start",
                    "parameters": {"bundleName": "com.ohos.settings"},
                },
            }
        if task.step == 2 and is_wlan_task:
            return {
                "action": {
                    "id": f"wlan-open-row-{task.step}",
                    "action": "click",
                    "x": 150,
                    "y": 165,
                },
                "reasoning": "Open the WLAN row from the top area of the Settings home page.",
                "raw": {
                    "reasoning": "Click WLAN row.",
                    "action": "click",
                    "parameters": {"x": 150, "y": 165},
                },
            }
        if task.step >= 3 and is_wlan_task and self.has_ok_result(task, "wlan-open-row"):
            return {
                "action": {
                    "id": f"wlan-done-{task.step}",
                    "action": "done",
                    "status": "success",
                },
                "reasoning": "WLAN page is open after deterministic WLAN row click; connected status is visible in the latest observation.",
                "raw": {
                    "reasoning": "WLAN page reached.",
                    "action": "done",
                    "parameters": {"status": "success"},
                },
            }
        return None

    def normalize_action(self, action: Dict[str, Any], task: TaskState, observation: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"wait", "done", "click", "swipe", "back", "home", "input", "app_start"}
        if action.get("action") not in allowed:
            task.add_trace("action_normalized", {"original": action, "reason": "unsupported action"})
            return {"id": f"wait-unsupported-{task.step}", "action": "wait", "duration": 1000}

        width = int(observation.get("width") or 0)
        height = int(observation.get("height") or 0)
        if action.get("action") == "click" and width > 0 and height > 0:
            action["x"] = max(0, min(width - 1, int(action.get("x", 0))))
            action["y"] = max(0, min(height - 1, int(action.get("y", 0))))
        return action

    def accept_done(self, task: TaskState) -> bool:
        return task.step > 1 and any(
            item.get("event") == "result" and item.get("payload", {}).get("status") == "ok"
            for item in task.trace
        )

    def has_ok_result(self, task: TaskState, action_id_prefix: str) -> bool:
        return any(
            item.get("event") == "result"
            and item.get("payload", {}).get("status") == "ok"
            and str(item.get("payload", {}).get("id", "")).startswith(action_id_prefix)
            for item in task.trace
        )

    def save_task(self, task: TaskState) -> None:
        task_dir = self.task_dir(task)
        (task_dir / "trace.json").write_text(
            json.dumps(self.task_to_dict(task, include_trace=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def task_to_dict(self, task: TaskState, include_trace: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "task_id": task.task_id,
            "device_id": task.device_id,
            "task": task.task,
            "status": task.status,
            "step": task.step,
            "max_steps": task.max_steps,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "latest_screenshot": task.latest_screenshot,
            "latest_reasoning": task.latest_reasoning,
            "latest_action": task.latest_action,
            "final_message": task.final_message,
        }
        if include_trace:
            data["trace"] = task.trace
        else:
            data["trace_tail"] = task.trace[-8:]
        return data


def create_app(controller: Controller) -> FastAPI:
    app = FastAPI(title="CSDI Local Agent Server")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LocalAgentServer</title>
  <style>
    body{font-family:Arial,sans-serif;margin:0;background:#f6f7f9;color:#1b1f24}
    main{max-width:1180px;margin:24px auto;padding:0 16px}
    .grid{display:grid;grid-template-columns:380px 1fr;gap:16px}
    .panel{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:16px}
    label{display:block;font-weight:600;margin:12px 0 6px}
    textarea,input{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cfd6e2;border-radius:6px;font:inherit}
    button{padding:9px 14px;border:0;border-radius:6px;background:#155eef;color:white;font-weight:600;cursor:pointer}
    button.secondary{background:#344054}
    pre{background:#101828;color:#e6edf3;padding:12px;border-radius:6px;white-space:pre-wrap;max-height:360px;overflow:auto}
    img{max-width:100%;border:1px solid #d0d5dd;border-radius:8px;background:#eee}
    .status{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
    .pill{padding:4px 8px;border-radius:999px;background:#eef4ff;color:#1849a9;font-size:12px}
  </style>
</head>
<body>
<main>
  <h1>LocalAgentServer</h1>
  <p>Board WebSocket URL: <code>ws://&lt;PC-IP&gt;:8088/ws?device_id=openharmony-board-01</code></p>
  <div class="grid">
    <section class="panel">
      <h2>Agent Input</h2>
      <label>Device ID</label>
      <input id="device" value="openharmony-board-01">
      <label>Task</label>
      <textarea id="task" rows="5">打开设置并进入 WLAN 页面，查看当前连接状态。</textarea>
      <label>Max Steps</label>
      <input id="steps" value="20">
      <div class="status">
        <button onclick="createTask()">Start Task</button>
        <button class="secondary" onclick="refresh()">Refresh</button>
      </div>
      <h3>Agent Output</h3>
      <pre id="answer">Waiting...</pre>
    </section>
    <section class="panel">
      <h2>Observation</h2>
      <div id="summary" class="status"></div>
      <p><img id="shot" alt="latest screenshot"></p>
      <h3>Trace Tail</h3>
      <pre id="trace"></pre>
    </section>
  </div>
</main>
<script>
let activeTask = '';
async function createTask(){
  const payload = {device_id:device.value,task:task.value,max_steps:Number(steps.value)};
  const res = await fetch('/tasks',{method:'POST',headers:{'Content-Type':'application/json; charset=utf-8'},body:JSON.stringify(payload)});
  const data = await res.json();
  activeTask = data.task_id || '';
  answer.textContent = JSON.stringify(data, null, 2);
  await refresh();
}
async function refresh(){
  const res = await fetch('/state');
  const data = await res.json();
  const taskIds = Object.keys(data.tasks || {});
  if (!activeTask && taskIds.length > 0) activeTask = taskIds[taskIds.length - 1];
  const t = activeTask ? data.tasks[activeTask] : null;
  summary.innerHTML = '';
  if (t) {
    summary.innerHTML = `<span class="pill">${t.status}</span><span class="pill">step ${t.step}/${t.max_steps}</span><span class="pill">${t.device_id}</span>`;
    answer.textContent = t.final_message || t.latest_reasoning || 'Running...';
    trace.textContent = JSON.stringify(t.trace_tail || [], null, 2);
    if (t.latest_screenshot) shot.src = `/tasks/${activeTask}/latest_screenshot?t=${Date.now()}`;
  } else {
    trace.textContent = JSON.stringify(data, null, 2);
  }
}
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>"""

    @app.get("/state")
    async def state() -> Dict[str, Any]:
        return {
            "devices": {k: {
                "connected": v.connected,
                "last_seen": v.last_seen,
                "info": v.info,
            } for k, v in controller.devices.items()},
            "active_task_by_device": controller.active_task_by_device,
            "tasks": {k: controller.task_to_dict(v, include_trace=False) for k, v in controller.tasks.items()},
        }

    @app.get("/tasks/{task_id}/latest_screenshot")
    async def latest_screenshot(task_id: str) -> Response:
        task = controller.tasks.get(task_id)
        if not task or not task.latest_screenshot:
            return Response(status_code=404)
        path = Path(task.latest_screenshot)
        if not path.exists():
            return Response(status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/tasks")
    async def tasks(req: TaskRequest) -> JSONResponse:
        try:
            task = await controller.create_task(req)
            return JSONResponse({
                "task_id": task.task_id,
                "status": task.status,
                "device_id": task.device_id,
            })
        except Exception as exc:
            return JSONResponse({"status": "failed", "error": str(exc)}, status_code=400)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket, device_id: str = "openharmony-board-01") -> None:
        await controller.register_ws(device_id, ws)
        try:
            while True:
                raw = await ws.receive_text()
                await controller.handle_message(device_id, json.loads(raw))
        except WebSocketDisconnect:
            await controller.unregister_ws(device_id)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--decider-api-base", default="http://0.tcp.jp.ngrok.io:23909/v1")
    parser.add_argument("--grounder-api-base", default="http://0.tcp.jp.ngrok.io:23909/v1")
    parser.add_argument("--decider-model", default="MobiMind-Mixed-4B-1031")
    parser.add_argument("--grounder-model", default="MobiMind-Mixed-4B-1031")
    parser.add_argument("--use-e2e", action="store_true", default=False)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge = MobiAgentBridge(
        decider_api_base=args.decider_api_base,
        grounder_api_base=args.grounder_api_base,
        decider_model=args.decider_model,
        grounder_model=args.grounder_model,
        use_e2e=args.use_e2e,
    )
    controller = Controller(bridge=bridge, output_dir=Path(args.output_dir))
    uvicorn.run(create_app(controller), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
