"""Singleton pipeline manager for concurrent request handling."""

import asyncio
from enum import Enum
from typing import Any, Dict, Optional

import psutil

from app.config import settings
from app.exceptions import ModelNotReadyError
from app.logger import get_core_logger
from app.service import TranslationPipelineCLI

logger = get_core_logger("pipeline_manager")


class PipelineStatus(str, Enum):
    """Pipeline initialization status."""

    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"


class PipelineManager:
    """Singleton manager for the translation pipeline to avoid cold starts."""

    _instance: Optional["PipelineManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "PipelineManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._pipeline: Optional[TranslationPipelineCLI] = None
        self._status: PipelineStatus = PipelineStatus.INITIALIZING
        self._error_message: Optional[str] = None
        self._initialization_task: Optional[asyncio.Task] = None

    @property
    def is_ready(self) -> bool:
        """Check if pipeline is ready to process requests."""
        return self._status == PipelineStatus.READY

    def get_status(self) -> Dict[str, Any]:
        """Get detailed pipeline status information."""
        status_dict = {
            "status": self._status.value,
            "ready": self.is_ready,
        }

        if self._status == PipelineStatus.FAILED and self._error_message:
            status_dict["error"] = self._error_message

        if self._pipeline and self._pipeline._ready:
            try:
                # Get system memory info
                vm = psutil.virtual_memory()
                status_dict["system_memory_gb"] = round(vm.total / (1024**3), 2)
                status_dict["system_memory_used_gb"] = round(vm.used / (1024**3), 2)

                # Get GPU info if CUDA is in use
                if self._pipeline._device == "cuda":
                    try:
                        import torch

                        device_props = torch.cuda.get_device_properties(0)
                        status_dict["gpu_device"] = device_props.name
                        total_vram = device_props.total_memory / (1024**3)
                        status_dict["gpu_total_memory_gb"] = round(total_vram, 2)
                    except Exception:
                        pass

                status_dict["detector_device"] = self._pipeline._device
            except Exception as exc:
                logger.warning(f"Failed to get detailed status: {exc}")

        return status_dict

    async def initialize(self) -> None:
        """Initialize the pipeline synchronously, blocking until ready."""
        async with self._lock:
            if self._status == PipelineStatus.READY:
                logger.info("Pipeline already initialized")
                return

            if self._status == PipelineStatus.FAILED:
                raise RuntimeError(f"Pipeline initialization failed: {self._error_message}")

            try:
                logger.info("Starting pipeline initialization...")
                self._pipeline = TranslationPipelineCLI()
                await self._pipeline.initialize()

                self._status = PipelineStatus.READY
                logger.info("Pipeline initialization completed successfully")

            except Exception as exc:
                self._status = PipelineStatus.FAILED
                self._error_message = str(exc)
                logger.error(f"Pipeline initialization failed: {exc}")
                raise

    async def translate(
        self, image_path: str, post_id: str = "0", target_lang: str = "ENG", **optional_settings
    ) -> Dict[str, Any]:
        """
        Translate an image using the pipeline.

        Args:
            image_path: Absolute path to the image file
            post_id: Optional metadata ID
            target_lang: Target translation language (e.g., "ENG", "VIE")
            **optional_settings: Optional CLI flags to override defaults
                - device: torch device (cpu, cuda, mps, auto)
                - detectionSize: Detection input size
                - textThreshold: Detection text threshold
                - boxThreshold: Detection box threshold
                - unclipRatio: Detection unclip ratio
                - detInvert: Invert detection input
                - detGammaCorrect: Apply gamma correction
                - detRotate: Enable detection rotation
                - detAutoRotate: Enable detection auto-rotation

        Returns:
            Dictionary with translation results

        Raises:
            ModelNotReadyError: If pipeline is not ready
            InvalidInputError: If image path is invalid
        """
        if not self.is_ready:
            raise ModelNotReadyError(f"Pipeline is not ready: {self._status.value}")

        # Apply optional settings temporarily for this request
        original_settings = {}
        try:
            # Map request field names to settings attribute names
            settings_map = {
                "device": "device",
                "detectionSize": "detection_size",
                "textThreshold": "text_threshold",
                "boxThreshold": "box_threshold",
                "unclipRatio": "unclip_ratio",
                "detInvert": "det_invert",
                "detGammaCorrect": "det_gamma_correct",
                "detRotate": "det_rotate",
                "detAutoRotate": "det_auto_rotate",
                "verbose": "verbose",
            }

            # Apply overrides from request
            for request_key, value in optional_settings.items():
                if request_key in settings_map and value is not None:
                    settings_attr = settings_map[request_key]
                    original_settings[settings_attr] = getattr(settings, settings_attr)
                    setattr(settings, settings_attr, value)

            # Override target language
            original_settings["target_lang"] = settings.target_lang
            settings.target_lang = target_lang

            # Execute translation
            result = await self._pipeline.translate_image(image_path, post_id, target_lang)
            return result

        finally:
            # Restore original settings
            for attr_name, original_value in original_settings.items():
                setattr(settings, attr_name, original_value)
