"""Shared Data Transfer Objects for CLI and Server modes."""
from typing import Dict, List

from pydantic import BaseModel, Field


class TextOverlay(BaseModel):
    """Translated text overlay region."""

    id: str = Field(..., description="Region identifier (r-{index})")
    xywh: Dict[str, float] = Field(..., description="Bounding box: x, y, width, height")
    polygon: List[Dict[str, float]] = Field(..., description="4-point polygon coordinates")
    sourceText: str = Field(..., description="Original detected text")
    translatedText: str = Field(..., description="Translated text")
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")
    sourceLang: str = Field(..., description="Source language code")


class TranslationTimings(BaseModel):
    """Timing breakdown for the translation pipeline."""

    totalMs: int = Field(..., description="Total execution time in milliseconds")
    imageLoadMs: int = Field(..., description="PIL Image.open() time")
    imageDecodeMs: int = Field(..., description="load_image() conversion time")
    detectMs: int = Field(..., description="Text detection time")
    ocrMs: int = Field(..., description="OCR recognition time")
    mergeMs: int = Field(..., description="Paragraph merging time")
    translateMs: int = Field(..., description="Translation time")
    device: str = Field(..., description="Device used for execution (cpu/cuda/mps)")
    detectedTextlines: int = Field(..., description="Text regions detected")
    recognizedTextlines: int = Field(..., description="Text regions recognized by OCR")
    mergedRegions: int = Field(..., description="Text regions after merging")


class TranslationResponse(BaseModel):
    """Complete translation response (used by both CLI verbose mode and Server mode)."""

    postId: str = Field(..., description="Post/metadata ID")
    imagePath: str = Field(..., description="Path to the translated image")
    originalSize: Dict[str, int] = Field(..., description="Image dimensions: width, height")
    timings: TranslationTimings = Field(..., description="Timing breakdown")
    overlays: List[TextOverlay] = Field(..., description="Translated text regions")


class TranslationResponseMinimal(BaseModel):
    """Minimal translation response (used by CLI non-verbose mode)."""

    postId: str = Field(..., description="Post/metadata ID")
    imagePath: str = Field(..., description="Path to the translated image")
    overlays: List[TextOverlay] = Field(..., description="Translated text regions")
