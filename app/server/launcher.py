"""Server launcher module for OverlayTranslator."""

import uvicorn

from app.config import settings
from app.logger import get_core_logger
from app.server.pipeline_manager import PipelineManager

logger = get_core_logger("launcher")


async def run_server(
    host: str = "127.0.0.1",
    port: int = 7861,
    blocking_init: bool = True,
) -> None:
    """
    Run the OverlayTranslator API server.

    Args:
        host: Server bind address
        port: Server listen port
        blocking_init: If True, blocks until pipeline is initialized before accepting requests
    """
    pipeline_manager = PipelineManager()

    if blocking_init:
        try:
            logger.info(f"Initializing pipeline before starting server (blocking)...")
            await pipeline_manager.initialize()
            logger.info("Pipeline initialized, starting server...")
        except Exception as exc:
            logger.error(f"Pipeline initialization failed, server will not accept requests: {exc}")
            # Continue anyway - server will return 503 on requests

    # Import app here to avoid circular imports
    from app.server.app import app

    # Configure and run uvicorn server
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=settings.log_level,
        access_log=True,
    )
    server = uvicorn.Server(config)

    logger.info(f"Starting server on {host}:{port}")
    await server.serve()
