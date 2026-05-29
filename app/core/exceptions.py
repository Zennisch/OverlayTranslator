from typing import Any, Optional


class SidecarError(Exception):
    """Base exception for all sidecar related errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        retryable: bool = True,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.retryable = retryable
        self.extra = extra or {}


class ModelNotReadyError(SidecarError):
    """Raised when the model is queried but has not finished loading."""

    def __init__(self, message: str = "Translation model is not ready yet", extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="MODEL_NOT_READY",
            status_code=503,
            retryable=True,
            extra=extra,
        )


class TranslationTimeoutError(SidecarError):
    """Raised when translation pipeline operations exceed configured limits."""

    def __init__(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="TIMEOUT",
            status_code=504,
            retryable=True,
            extra=extra,
        )


class UnsupportedMediaError(SidecarError):
    """Raised when an unsupported image type or format is loaded."""

    def __init__(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="UNSUPPORTED_MEDIA",
            status_code=415,
            retryable=False,
            extra=extra,
        )


class InvalidInputError(SidecarError):
    """Raised when input parameters, paths or arguments are malformed."""

    def __init__(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="INVALID_INPUT",
            status_code=400,
            retryable=False,
            extra=extra,
        )


class SidecarUnavailableError(SidecarError):
    """Raised when required sidecar resources/subprocesses are completely missing."""

    def __init__(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="SIDECAR_UNAVAILABLE",
            status_code=503,
            retryable=True,
            extra=extra,
        )


class InternalPipelineError(SidecarError):
    """Raised on unforeseen internal pipeline/inference failures."""

    def __init__(self, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code="INTERNAL_ERROR",
            status_code=500,
            retryable=True,
            extra=extra,
        )
