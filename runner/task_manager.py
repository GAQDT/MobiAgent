# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import logging
from importlib import import_module
from typing import Dict


class TaskManager:
    """
    Task manager that creates and executes tasks for different providers.
    """

    def __init__(
        self,
        provider: str,
        task_description: str,
        device,
        data_dir: str,
        device_type: str = "Android",
        max_steps: int = 40,
        draw: bool = False,
        **kwargs,
    ):
        self.provider = provider
        self.task_description = task_description
        self.device = device
        self.data_dir = data_dir
        self.device_type = device_type
        self.max_steps = max_steps
        self.kwargs = kwargs

        self.task_map = self._get_task_map()
        if provider not in self.task_map:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available providers: {list(self.task_map.keys())}"
            )

        task_class = self._load_task_class(provider)
        self.task = task_class(
            task_description=task_description,
            device=device,
            data_dir=data_dir,
            device_type=device_type,
            max_steps=max_steps,
            draw=draw,
            **kwargs,
        )

        logging.info(f"TaskManager initialized with provider: {provider}")

    def _get_task_map(self) -> Dict[str, tuple[str, str]]:
        return {
            "mobiagent": ("providers.mobiagent.mobile_task", "MobiAgentStepTask"),
            "mobiagent_step": ("providers.mobiagent.mobile_task", "MobiAgentStepTask"),
            "uitars": ("providers.uitars.uitars_task", "UITARSTask"),
            "qwen": ("providers.qwen.qwen_task", "QwenTask"),
            "autoglm": ("providers.autoglm.autoglm_task", "AutoGLMTask"),
        }

    def _load_task_class(self, provider: str):
        module_name, class_name = self.task_map[provider]
        try:
            module = import_module(module_name)
            return getattr(module, class_name)
        except ModuleNotFoundError as e:
            missing_name = getattr(e, "name", "") or str(e)
            raise ModuleNotFoundError(
                f"Failed to load provider '{provider}' because dependency "
                f"'{missing_name}' is missing. Please install the required "
                f"package and try again."
            ) from e

    def execute(self) -> Dict:
        logging.info(f"Executing task with provider: {self.provider}")
        return self.task.execute()

    def get_task_info(self) -> Dict:
        return {
            "provider": self.provider,
            "task_description": self.task_description,
            "device_type": self.device_type,
            "max_steps": self.max_steps,
            "data_dir": self.data_dir,
        }
