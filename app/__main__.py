import argparse
import asyncio
import json
import os
import sys

from app.config import settings
from app.exceptions import SidecarError
from app.logger import get_core_logger, setup_logging

setup_logging()
logger = get_core_logger("cli")


async def async_main():
    parser = argparse.ArgumentParser(description="Overlay Translator - CLI Engine or API Server")

    # Server mode flag
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run in server mode (API server) instead of CLI mode",
    )
    parser.add_argument(
        "--serverPort",
        type=int,
        default=7861,
        help="Server port when running in server mode (default: 7861)",
    )

    # CLI mode arguments (only required when not in server mode)
    parser.add_argument(
        "--imagePath",
        required=False,
        help="Absolute path to the Japanese image (required for CLI mode)",
    )
    parser.add_argument(
        "--postId",
        default="0",
        help="Optional post/metadata ID",
    )

    parser.add_argument(
        "--targetLang",
        default="ENG",
        help="Target translation language",
    )
    parser.add_argument(
        "--sourceLang",
        default="JPN",
        help="Source language (JPN, ENG, etc. or 'auto' for auto-detection)",
    )

    parser.add_argument(
        "--logLevel",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set logging level (debug, info, warning, error, critical)",
    )

    parser.add_argument(
        "--detectionSize",
        type=int,
        help="Detection input size (default: 2048)",
    )
    parser.add_argument(
        "--textThreshold",
        type=float,
        help="Detection text threshold (default: 0.5)",
    )
    parser.add_argument(
        "--boxThreshold",
        type=float,
        help="Detection box threshold (default: 0.7)",
    )
    parser.add_argument(
        "--unclipRatio",
        type=float,
        help="Detection unclip ratio (default: 2.3)",
    )
    parser.add_argument(
        "--detInvert",
        action="store_true",
        help="Invert detection input image",
    )
    parser.add_argument(
        "--detGammaCorrect",
        action="store_true",
        help="Apply gamma correction to detection input",
    )
    parser.add_argument(
        "--detRotate",
        action="store_true",
        help="Enable detection rotation",
    )
    parser.add_argument(
        "--detAutoRotate",
        action="store_true",
        help="Enable detection auto-rotation",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if not args.server and not args.imagePath:
        parser.error("--imagePath is required for CLI mode (or use --server for server mode)")

    # Sync CLI args with settings
    settings.target_lang = args.targetLang
    settings.source_lang = args.sourceLang

    if args.logLevel:
        settings.log_level = args.logLevel
        setup_logging()

    # If server mode, launch server and exit
    if args.server:
        logger.info(f"Starting server mode on port {args.serverPort}")
        from app.server.launcher import run_server

        await run_server(port=args.serverPort)
        return

    if args.detectionSize is not None:
        settings.detection_size = args.detectionSize
    if args.textThreshold is not None:
        settings.text_threshold = args.textThreshold
    if args.boxThreshold is not None:
        settings.box_threshold = args.boxThreshold
    if args.unclipRatio is not None:
        settings.unclip_ratio = args.unclipRatio
    if args.detInvert:
        settings.det_invert = True
    if args.detGammaCorrect:
        settings.det_gamma_correct = True
    if args.detRotate:
        settings.det_rotate = True
    if args.detAutoRotate:
        settings.det_auto_rotate = True

    if args.verbose:
        settings.verbose = True

    logger.info(f"Starting standalone translation pipeline for image: {args.imagePath} (postId: {args.postId})")

    if not os.path.exists(args.imagePath):
        err_json = {
            "postId": args.postId,
            "imagePath": args.imagePath,
            "error": f"Image file not found: {args.imagePath}",
            "errorCode": "INVALID_INPUT",
            "status": "error",
        }
        # Print directly to stdout for Electron to capture
        print(json.dumps(err_json, indent=2))
        sys.exit(1)

    try:
        from app.service import TranslationPipelineCLI

        pipeline = TranslationPipelineCLI()

        logger.info("Initializing models (Detector, OCR, Translator)...")
        await pipeline.initialize()

        logger.info("Executing image translation pipeline...")
        result = await pipeline.translate_image(image_path=args.imagePath, post_id=args.postId, target_lang=args.targetLang)

        # Return pure JSON to stdout
        print(json.dumps(result, indent=2))

        # Wait for any active background probe thread to finish before exiting (so cache can be written)
        import threading

        for t in threading.enumerate():
            if t.name == "BackgroundProbe":
                logger.info("Waiting for background layer probing to complete before exiting...")
                t.join()

        sys.exit(0)

    except SidecarError as exc:
        logger.exception("Pipeline failed with SidecarError")
        err_json = {
            "postId": args.postId,
            "imagePath": args.imagePath,
            "error": exc.message,
            "errorCode": exc.error_code,
            "status": "error",
        }
        print(json.dumps(err_json, indent=2))
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed with unexpected exception")
        err_json = {
            "postId": args.postId,
            "imagePath": args.imagePath,
            "error": str(exc),
            "errorCode": "INTERNAL_ERROR",
            "status": "error",
        }
        print(json.dumps(err_json, indent=2))
        sys.exit(1)


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
