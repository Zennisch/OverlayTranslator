"""FastAPI application for OverlayTranslator server mode."""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.dto import ErrorResponse, HealthResponse, TranslateRequest, TranslateResponse
from app.exceptions import GlobalError
from app.logger import get_core_logger
from app.server.pipeline_manager import PipelineManager

logger = get_core_logger("fastapi_app")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="OverlayTranslator API",
        description="Server for manga/comic image translation with text detection and LLM translation",
        version="1.0.0",
    )

    pipeline_manager = PipelineManager()

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """Check server and pipeline health status."""
        status = pipeline_manager.get_status()

        response = HealthResponse(
            status=status["status"],
            ready=status["ready"],
            error=status.get("error"),
            device=status.get("device"),
            system_memory_gb=status.get("system_memory_gb"),
            system_memory_used_gb=status.get("system_memory_used_gb"),
        )

        # Return 503 if not ready
        if not pipeline_manager.is_ready:
            raise HTTPException(status_code=503, detail=response.dict())

        return response

    @app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
    async def translate(request: TranslateRequest) -> TranslateResponse:
        """
        Translate Japanese manga/comic image to target language.

        Returns detected text regions with translations and overlay coordinates.
        """
        try:
            # Validate image path exists
            if not os.path.exists(request.imagePath):
                raise HTTPException(
                    status_code=400,
                    detail=ErrorResponse(
                        error=f"Image file not found: {request.imagePath}",
                        errorCode="INVALID_INPUT",
                    ).dict()
                )

            # Build optional settings from request
            optional_settings = {}
            for field_name in request.model_fields:
                if field_name not in ["imagePath", "postId", "targetLang"]:
                    value = getattr(request, field_name)
                    if value is not None:
                        optional_settings[field_name] = value

            # Execute translation
            result = await pipeline_manager.translate(
                image_path=request.imagePath,
                post_id=request.postId,
                target_lang=request.targetLang,
                **optional_settings
            )

            return TranslateResponse(**result)

        except GlobalError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=ErrorResponse(
                    error=exc.message,
                    errorCode=exc.error_code,
                    retryable=exc.retryable,
                ).dict()
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during translation")
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=str(exc),
                    errorCode="INTERNAL_ERROR",
                    retryable=True,
                ).dict()
            )

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle unexpected exceptions."""
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(exc),
                errorCode="INTERNAL_ERROR",
                retryable=True,
            ).dict()
        )

    return app


# Create the app instance
app = create_app()
