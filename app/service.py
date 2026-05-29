import time

import torch
from manga_translator.detection.default import DefaultDetector
from manga_translator.ocr import Model48pxOCR, Ocr, OcrConfig
from manga_translator.textline_merge import dispatch as dispatch_textline_merge
from manga_translator.translators.deep_translator_wrapper import DeepTranslatorWrapper
from manga_translator.utils import load_image
from PIL import Image

from app.config import settings
from app.exceptions import ModelNotReadyError
from app.logger import get_core_logger

logger = get_core_logger("service")


class TranslationPipelineCLI:
    def __init__(self) -> None:
        self._ready = False
        self._device: str = "cpu"

        self._detector: DefaultDetector = None
        self._ocr: Model48pxOCR = None
        self._translator = None
        self._ocr_config: OcrConfig = None

    @staticmethod
    def _resolve_device(torch_module, requested: str) -> str:
        req = (requested or "auto").strip().lower()
        if req not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("SIDECAR_DEVICE must be one of: auto, cpu, cuda, mps")

        if req == "cpu":
            return "cpu"

        if req == "cuda":
            if torch_module.cuda.is_available():
                return "cuda"
            raise RuntimeError("Requested CUDA device is not available in this environment")

        if req == "mps":
            if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                return "mps"
            raise RuntimeError("Requested MPS device is not available in this environment")

        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _to_overlay_item(self, idx: int, region) -> dict:
        xywh = region.xywh
        min_rect = region.min_rect[0]

        confidence = float(region.prob if region.prob is not None else 0.0)
        confidence = max(0.0, min(1.0, confidence))

        polygon = [
            {"x": float(min_rect[0][0]), "y": float(min_rect[0][1])},
            {"x": float(min_rect[1][0]), "y": float(min_rect[1][1])},
            {"x": float(min_rect[2][0]), "y": float(min_rect[2][1])},
            {"x": float(min_rect[3][0]), "y": float(min_rect[3][1])},
        ]

        return {
            "id": f"r-{idx}",
            "xywh": {
                "x": float(xywh[0]),
                "y": float(xywh[1]),
                "width": float(xywh[2]),
                "height": float(xywh[3]),
            },
            "polygon": polygon,
            "sourceText": str(region.text or ""),
            "translatedText": str(region.translation or ""),
            "confidence": confidence,
            "sourceLang": "JPN",
        }

    async def initialize(self) -> None:
        """Asynchronously load and prepare the pipeline detector, OCR, and GGUF LLM models."""
        try:

            self._device = self._resolve_device(torch, settings.device)

            self._ocr_config = OcrConfig(
                ocr=Ocr.ocr48px,
                min_text_length=1,
                ignore_bubble=0,
            )

            self._detector = DefaultDetector()
            self._ocr = Model48pxOCR()
            self._translator = DeepTranslatorWrapper()
            logger.info("Using DeepTranslator backend (Google Translate)")

            await self._detector.load(self._device)
            await self._ocr.load(self._device)
            await self._translator.load("auto", settings.target_lang, self._device)

            self._ready = True
            logger.info(f"Models successfully initialized: device={self._device}")

        except Exception as exc:
            self._ready = False
            logger.error(f"Initialization failed: {exc}")
            raise

    async def translate_image(self, image_path: str, post_id: str, target_lang: str) -> dict:
        """
        Loads the image, runs detection -> OCR -> Paragraph Merging -> LLM Translation -> Normalization,
        returning a standard output dictionary.
        """
        if not self._ready:
            raise ModelNotReadyError("Pipeline is not ready")

        started = time.perf_counter()

        # 1. Load PIL Image
        t0 = time.perf_counter()
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to open image at {image_path}: {exc}")
        image_load_ms = int((time.perf_counter() - t0) * 1000)

        # 2. Decode using engine utilities
        t0 = time.perf_counter()
        img_rgb, _ = load_image(pil_image)
        image_decode_ms = int((time.perf_counter() - t0) * 1000)

        img_height, img_width = int(img_rgb.shape[0]), int(img_rgb.shape[1])

        # 3. Detect textlines
        t0 = time.perf_counter()
        textlines, _, _ = await self._detector.detect(
            img_rgb,
            settings.detection_size,
            settings.text_threshold,
            settings.box_threshold,
            settings.unclip_ratio,
            settings.det_invert,
            settings.det_gamma_correct,
            settings.det_rotate,
            settings.det_auto_rotate,
            settings.verbose,
        )
        detect_ms = int((time.perf_counter() - t0) * 1000)
        detected_textlines = len(textlines)

        def build_response(overlays: list, timings_dict: dict) -> dict:
            return {
                "postId": post_id,
                "imagePath": image_path,
                "originalSize": {"width": img_width, "height": img_height},
                "translator": "deep-translator",
                "elapsedMs": int((time.perf_counter() - started) * 1000),
                "timings": timings_dict,
                "overlays": overlays,
            }

        # Handle case when no text is detected
        if not textlines:
            total_ms = int((time.perf_counter() - started) * 1000)
            return build_response(
                [],
                {
                    "totalMs": total_ms,
                    "imageLoadMs": image_load_ms,
                    "imageDecodeMs": image_decode_ms,
                    "detectMs": detect_ms,
                    "ocrMs": 0,
                    "mergeMs": 0,
                    "translateMs": 0,
                    "detectorDevice": self._device,
                    "detectedTextlines": 0,
                    "recognizedTextlines": 0,
                    "mergedRegions": 0,
                },
            )

        # 4. Recognize characters
        t0 = time.perf_counter()
        textlines = await self._ocr.recognize(img_rgb, textlines, self._ocr_config, settings.verbose)
        ocr_ms = int((time.perf_counter() - t0) * 1000)

        # Filter out empty texts
        textlines = [q for q in textlines if getattr(q, "text", "")]
        recognized_textlines = len(textlines)

        if not textlines:
            total_ms = int((time.perf_counter() - started) * 1000)
            return build_response(
                [],
                {
                    "totalMs": total_ms,
                    "imageLoadMs": image_load_ms,
                    "imageDecodeMs": image_decode_ms,
                    "detectMs": detect_ms,
                    "ocrMs": ocr_ms,
                    "mergeMs": 0,
                    "translateMs": 0,
                    "detectorDevice": self._device,
                    "detectedTextlines": detected_textlines,
                    "recognizedTextlines": 0,
                    "mergedRegions": 0,
                },
            )

        # 5. Merge aligned textlines
        t0 = time.perf_counter()
        text_regions = await dispatch_textline_merge(
            textlines,
            width=img_width,
            height=img_height,
            verbose=settings.verbose,
        )
        merge_ms = int((time.perf_counter() - t0) * 1000)
        merged_regions = len(text_regions)

        # 6. Translation
        queries = [str(region.text or "") for region in text_regions]
        t0 = time.perf_counter()
        translations = await self._translator.translate("auto", target_lang, queries, use_mtpe=False)
        translate_ms = int((time.perf_counter() - t0) * 1000)

        # 7. Build final overlays list
        overlays = []
        for idx, region in enumerate(text_regions):
            region.translation = translations[idx] if idx < len(translations) else ""
            overlays.append(self._to_overlay_item(idx, region))

        total_ms = int((time.perf_counter() - started) * 1000)
        return build_response(
            overlays,
            {
                "totalMs": total_ms,
                "imageLoadMs": image_load_ms,
                "imageDecodeMs": image_decode_ms,
                "detectMs": detect_ms,
                "ocrMs": ocr_ms,
                "mergeMs": merge_ms,
                "translateMs": translate_ms,
                "detectorDevice": self._device,
                "detectedTextlines": detected_textlines,
                "recognizedTextlines": recognized_textlines,
                "mergedRegions": merged_regions,
            },
        )
