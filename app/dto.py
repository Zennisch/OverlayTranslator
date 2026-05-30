"""Shared Data Transfer Objects for CLI and Server modes."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TranslationTextOverlay(BaseModel):
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


class TranslationRequest(BaseModel):
    """Translation request payload."""

    imagePath: str = Field(..., description="Absolute path to the image file to translate")
    postId: str = Field("0", description="Optional post/metadata ID")
    sourceLang: str = Field("JPN", description="Source language (JPN, ENG, etc. or 'auto' for auto-detection)")
    targetLang: str = Field("ENG", description="Target translation language (e.g., ENG, VIE)")

    # Detection parameters
    detectionSize: Optional[int] = Field(None, description="Detection input size")
    textThreshold: Optional[float] = Field(None, description="Detection text threshold")
    boxThreshold: Optional[float] = Field(None, description="Detection box threshold")
    unclipRatio: Optional[float] = Field(None, description="Detection unclip ratio")
    detInvert: Optional[bool] = Field(None, description="Invert detection input")
    detGammaCorrect: Optional[bool] = Field(None, description="Apply gamma correction to detection")
    detRotate: Optional[bool] = Field(None, description="Enable detection rotation")
    detAutoRotate: Optional[bool] = Field(None, description="Enable detection auto-rotation")

    # Misc
    verbose: Optional[bool] = Field(None, description="Enable verbose logging")


class TranslationResponse(BaseModel):
    """Complete translation response (used by both CLI and Server modes)."""

    postId: str = Field(..., description="Post/metadata ID")
    imagePath: str = Field(..., description="Path to the translated image")
    originalSize: Optional[Dict[str, int]] = Field(None, description="Image dimensions: width, height")
    timings: Optional[TranslationTimings] = Field(None, description="Timing breakdown")
    overlays: List[TranslationTextOverlay] = Field(..., description="Translated text regions")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Pipeline status: initializing, ready, or failed")
    ready: bool = Field(..., description="True if pipeline is ready to handle requests")
    error: Optional[str] = Field(None, description="Error message if status is failed")
    device: Optional[str] = Field(None, description="Device used for execution (always cpu)")
    system_memory_gb: Optional[float] = Field(None, description="Total system RAM in GB")
    system_memory_used_gb: Optional[float] = Field(None, description="Used system RAM in GB")


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str = Field(..., description="Error message")
    errorCode: str = Field(..., description="Error code (e.g., INVALID_INPUT, MODEL_NOT_READY)")
    status: str = Field("error", description="Status indicator")
    retryable: Optional[bool] = Field(None, description="Whether the request can be retried")
