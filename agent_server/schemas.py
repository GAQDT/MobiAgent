from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class DeviceSession:
    device_id: str
    connected: bool = False
    last_seen: str = field(default_factory=now_iso)
    info: Dict[str, Any] = field(default_factory=dict)
    websocket: Any = None


@dataclass
class TaskState:
    device_id: str
    task: str
    max_steps: int = 30
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "queued"
    step: int = 0
    history: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    pending_action: Optional[Dict[str, Any]] = None
    latest_screenshot: Optional[str] = None
    latest_reasoning: str = ""
    latest_action: Optional[Dict[str, Any]] = None
    final_message: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def add_trace(self, event: str, payload: Dict[str, Any]) -> None:
        self.updated_at = now_iso()
        self.trace.append({
            "time": self.updated_at,
            "event": event,
            "payload": payload,
        })
