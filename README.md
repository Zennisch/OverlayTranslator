# OverlayTranslator

**Detect, OCR, and translate Japanese manga/comic images with text overlay coordinates for in-game/in-app rendering.**

OverlayTranslator is a specialized tool that automatically detects text regions in manga images, extracts text via OCR, translates to your target language, and returns precise overlay coordinates for rendering translated text. Designed for Emiru overlay system integration.

**Latest Version:** 0.1.0 | **Python:** 3.11-3.12 | **License:** GPL v3.0

---

## 🔗 Fork Attribution

This project is a **heavily-modified fork** of [manga-image-translator](https://github.com/zyddnys/manga-image-translator) by [zyddnys](https://github.com/zyddnys), licensed under **GPL v3.0**.

### Key Modifications from Original

The original `manga-image-translator` is a comprehensive manga translation pipeline with advanced GPU optimization. OverlayTranslator **simplifies and refactors** this for a specific use case: CPU-first deployment with streamlined translation.

| Aspect | Original | OverlayTranslator |
|--------|----------|-------------------|
| **Translation Backend** | Local LLM (llama.cpp + GGUF models) | Google Translate (deep-translator) |
| **Device Support** | CUDA GPU-optimized, VRAM management | CPU-first (GPU optional, PyTorch auto-detect) |
| **Complexity** | Advanced: LLM layer optimization, contention handling, GGML tuning | Simplified: 4-step pipeline, no model management |
| **Model Distribution** | 1.5GB+ GGUF models for LLM | Pre-trained detection/OCR models only |
| **Execution Modes** | CLI only | CLI + REST API (singleton pattern) |
| **Dependencies** | llama-cpp-python, complex CUDA setup | deep-translator, FastAPI, minimal setup |

**Why these changes?**
- **deep-translator over LLM**: Eliminates model distribution, download delays, and VRAM management complexity. Google Translate is fast, free, and requires zero local models.
- **CPU-first over GPU-optimized**: Reduces deployment friction. Most users run on CPU; GPU is optional bonus, not prerequisite.
- **REST API addition**: Enables easy integration with game overlays, Discord bots, and other services without subprocess management.

---

## ⚡ Quick Start

### Prerequisites
- **Python 3.11 or 3.12** (3.13+ not supported due to torch)
- **~200MB disk** for pre-trained models (auto-downloaded on first run)
- **No CUDA/GPU required** (works 100% on CPU, GPU accelerates ~2-3x if available)

### Install (2 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/OverlayTranslator.git
cd OverlayTranslator

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Translate Your First Image (CLI)

```bash
python -m app --imagePath "/path/to/manga.jpg" --targetLang "ENG"
```

**Output:**
```json
{
  "postId": "0",
  "imagePath": "/path/to/manga.jpg",
  "overlays": [
    {
      "id": "0",
      "xywh": {"x": 150, "y": 200, "w": 180, "h": 40},
      "polygon": [
        {"x": 150, "y": 200},
        {"x": 330, "y": 200},
        {"x": 330, "y": 240},
        {"x": 150, "y": 240}
      ],
      "sourceText": "こんにちは",
      "translatedText": "Hello",
      "confidence": 0.92,
      "sourceLang": "ja"
    }
  ],
  "translator": "deep-translator",
  "elapsedMs": 2340
}
```

---

## 📦 Installation

### System Requirements
- **Python:** 3.11.x or 3.12.x
- **RAM:** 2GB minimum (4GB+ recommended)
- **Disk:** 200MB for models + virtual environment
- **OS:** Windows, Linux, macOS

### Development Setup

```bash
# Clone and enter directory
git clone https://github.com/yourusername/OverlayTranslator.git
cd OverlayTranslator

# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m app --help
```

### Model Auto-Download

Models are automatically downloaded on first run to:
- Detection model: `app/models/detection/detect-20241225.ckpt`
- OCR model: `app/models/ocr/ocr_ar_48px.ckpt`
- OCR alphabet: `app/models/ocr/alphabet-all-v7.txt`

If downloads fail, manually download from the original [manga-image-translator](https://github.com/zyddnys/manga-image-translator) repository.

### Optional: GPU Acceleration (CUDA)

PyTorch auto-detects NVIDIA GPUs. To enable:

```bash
# Install CUDA-enabled PyTorch (replaces CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

GPU speeds up detection/OCR ~2-3x. No model changes needed.

---

## 🏗️ Architecture

### 4-Step Pipeline

```
Input Image
    ↓
[1] TEXT DETECTION
    └─ DefaultDetector: Finds text bounding boxes
    └─ Model: detect-20241225.ckpt (EAST-based)
    └─ Output: Quadrilateral coordinates
    ↓
[2] OCR (Character Recognition)
    └─ Model48pxOCR: Recognizes characters in each box
    └─ Model: ocr_ar_48px.ckpt (48-pixel height)
    └─ Output: Character strings per region
    ↓
[3] TEXTLINE MERGING
    └─ dispatch_textline_merge: Groups nearby OCR results
    └─ Uses graph connectivity (NetworkX)
    └─ Output: TextBlock objects with merged geometry
    ↓
[4] TRANSLATION
    └─ DeepTranslatorWrapper: Batch translates via Google Translate
    └─ Supports 24+ languages
    └─ Output: Translated text per region
    ↓
JSON Response with Overlays
```

### Project Structure

```
OverlayTranslator/
├── app/
│   ├── __main__.py                 # CLI entry point with argparse
│   ├── bootstrap.py                # PyInstaller path configuration
│   ├── config.py                   # Global Settings class
│   ├── service.py                  # TranslationPipelineCLI (orchestrator)
│   ├── core/
│   │   ├── exceptions.py           # Error hierarchy (GlobalError, etc.)
│   │   └── logger.py               # Logging setup
│   ├── manga_translator/
│   │   ├── detection/
│   │   │   ├── default.py          # DefaultDetector class
│   │   │   └── default_utils/      # Detection utilities & model wrapper
│   │   ├── ocr/
│   │   │   ├── common.py           # CommonOCR base classes
│   │   │   ├── config.py           # OcrConfig dataclass
│   │   │   └── model_48px.py       # Model48pxOCR implementation
│   │   ├── textline_merge/         # Text region merging logic
│   │   ├── translators/
│   │   │   ├── common.py           # OfflineTranslator base class
│   │   │   └── deep_translator_wrapper.py  # Google Translate wrapper
│   │   └── utils/                  # Utility functions (inference, textblock, etc.)
│   ├── models/
│   │   ├── detection/detect-20241225.ckpt
│   │   └── ocr/ocr_ar_48px.ckpt + alphabet-all-v7.txt
│   └── server/
│       ├── app.py                  # FastAPI application
│       ├── launcher.py             # Server startup
│       ├── pipeline_manager.py     # Singleton pipeline (async init, status tracking)
│       └── schemas.py              # Pydantic request/response models
├── tests/
│   ├── conftest.py
│   ├── test_cli.py
│   └── test_server.py
├── build.py                        # PyInstaller build script
├── requirements.txt                # Python dependencies
└── pyproject.toml                  # Project metadata (Poetry)
```

### Execution Modes

#### CLI Mode (Single Image)
```
image path → load → detect → OCR → merge → translate → JSON stdout
```

Runs synchronously, single-threaded. Good for batch processing, automation, serverless.

#### Server Mode (REST API)
```
HTTP requests → [Async FastAPI] ← [Singleton Pipeline Manager]
                    ↓
            TranslationPipelineCLI
                    ↓
            Returns JSON response
```

Models loaded once on startup, reused across all requests. Async/await prevents blocking. Perfect for real-time integrations.

---

## 💻 Usage Examples

### CLI Mode: Single Image Translation

**Basic translation (English target):**
```bash
python -m app --imagePath "manga.jpg" --targetLang "ENG"
```

**With custom detection settings:**
```bash
python -m app \
  --imagePath "manga.jpg" \
  --targetLang "VIE" \
  --device "cpu" \
  --detectionSize 1024 \
  --textThreshold 0.6
```

**With verbose logging:**
```bash
python -m app \
  --imagePath "manga.jpg" \
  --logLevel "debug" \
  --verbose
```

### CLI Mode: Batch Processing

```bash
for img in *.jpg; do
  python -m app --imagePath "$img" --targetLang "ENG" > "${img%.jpg}.json"
done
```

### Server Mode: Start REST API

**Start server on default port (7861):**
```bash
python -m app --server
```

**Custom port:**
```bash
python -m app --server --serverPort 8080
```

**Server output:**
```
INFO: Started server process [12345]
INFO: Waiting for application startup.
INFO: Application startup complete
INFO: Uvicorn running on http://127.0.0.1:7861
```

### Server Mode: Health Check

Check server readiness before sending translation requests:

```bash
curl -X GET http://127.0.0.1:7861/health
```

**Response (ready):**
```json
{
  "status": "ready",
  "ready": true,
  "error": null,
  "detector_device": "cpu",
  "translator_device": "cpu",
  "gpu_device": null,
  "system_memory_gb": 16.0,
  "system_memory_used_gb": 4.2,
  "gpu_total_memory_gb": null
}
```

**Response (initializing):**
```json
{
  "status": "initializing",
  "ready": false,
  "error": null,
  "detector_device": null,
  "translator_device": null,
  "gpu_device": null,
  "system_memory_gb": null,
  "system_memory_used_gb": null,
  "gpu_total_memory_gb": null
}
```

HTTP status: **503 Service Unavailable** if not ready.

### Server Mode: Translate Image via API

**Python:**
```python
import requests
import json

response = requests.post(
    "http://127.0.0.1:7861/translate",
    json={
        "imagePath": "/absolute/path/to/manga.jpg",
        "postId": "123",
        "targetLang": "ENG",
        "device": "auto"
    }
)

result = response.json()
print(f"Translated {len(result['overlays'])} regions in {result['elapsedMs']}ms")

for overlay in result['overlays']:
    print(f"  {overlay['sourceText']} → {overlay['translatedText']}")
```

**cURL:**
```bash
curl -X POST http://127.0.0.1:7861/translate \
  -H "Content-Type: application/json" \
  -d '{
    "imagePath": "/absolute/path/to/manga.jpg",
    "targetLang": "ENG",
    "device": "auto"
  }'
```

**Request Payload:**
```json
{
  "imagePath": "/absolute/path/to/manga.jpg",
  "postId": "optional_metadata_id",
  "targetLang": "ENG",
  "device": "auto",
  "detectionSize": 2048,
  "textThreshold": 0.5,
  "boxThreshold": 0.7,
  "unclipRatio": 2.3,
  "verbose": false
}
```

**Response:**
```json
{
  "postId": "optional_metadata_id",
  "imagePath": "/absolute/path/to/manga.jpg",
  "originalSize": {
    "width": 1280,
    "height": 960
  },
  "translator": "deep-translator",
  "elapsedMs": 2340,
  "timings": {
    "totalMs": 2340,
    "imageLoadMs": 45,
    "imageDecodeMs": 120,
    "detectMs": 890,
    "ocrMs": 640,
    "mergeMs": 180,
    "translateMs": 410,
    "normalizeMs": 15,
    "detectorDevice": "cpu",
    "translatorDevice": "cpu",
    "detectedTextlines": 12,
    "recognizedTextlines": 12,
    "mergedRegions": 8
  },
  "overlays": [
    {
      "id": "0",
      "xywh": {
        "x": 150.5,
        "y": 200.0,
        "w": 180.3,
        "h": 40.2
      },
      "polygon": [
        {"x": 150.5, "y": 200.0},
        {"x": 330.8, "y": 200.0},
        {"x": 330.8, "y": 240.2},
        {"x": 150.5, "y": 240.2}
      ],
      "sourceText": "こんにちは",
      "translatedText": "Hello",
      "confidence": 0.92,
      "sourceLang": "ja"
    }
  ]
}
```

---

## ⚙️ Configuration

All settings are in [app/config.py](app/config.py) with static defaults. Override via CLI arguments.

### Global Settings (Defaults)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `log_level` | str | "info" | Logging verbosity (debug, info, warning, error, critical) |
| `target_lang` | str | "ENG" | Target translation language code |
| `device` | str | "auto" | Torch device for detection/OCR (cpu, cuda, mps, auto) |
| `detection_size` | int | 2048 | Input size for text detection model |
| `text_threshold` | float | 0.5 | Confidence threshold for text detection |
| `box_threshold` | float | 0.7 | Confidence threshold for bounding boxes |
| `unclip_ratio` | float | 2.3 | Expansion ratio for detected text regions |
| `det_invert` | bool | False | Invert image for detection (dark text on light) |
| `det_gamma_correct` | bool | False | Apply gamma correction to detection input |
| `det_rotate` | bool | False | Enable rotation augmentation in detection |
| `det_auto_rotate` | bool | False | Auto-detect and rotate image (SLOW) |
| `server_host` | str | "127.0.0.1" | Server bind address |
| `server_port` | int | 7861 | Server listen port |
| `server_blocking_init` | bool | True | Block /translate requests until models are loaded |

### Supported Languages

Google Translate (via deep-translator) supports 24+ languages:

- **East Asian:** JPN, CHS (Simplified Chinese), CHT (Traditional Chinese), KOR
- **European:** ENG, DEU (German), FRA (French), ESP (Spanish), ITA (Italian), NLD (Dutch), POL (Polish), RUS (Russian), UKR (Ukrainian), CSY (Czech), HUN (Hungarian), ROM (Romanian)
- **Southeast Asian:** VIE (Vietnamese), THA (Thai), IND (Indonesian), FIL (Filipino)
- **Middle East/Other:** ARA (Arabic), TRK (Turkish)

Use language code in `--targetLang` or `targetLang` request parameter.

### CLI Arguments

All CLI arguments override [app/config.py](app/config.py) defaults:

```
--server                    Run in server mode (default: CLI mode)
--serverPort PORT           Server port (default: 7861)
--imagePath PATH            Image path for CLI mode (REQUIRED for CLI)
--postId ID                 Optional metadata ID (default: "0")
--targetLang CODE           Target language code (default: "ENG")
--device DEVICE             Torch device: cpu/cuda/mps/auto (default: "auto")
--logLevel LEVEL            Log level: debug/info/warning/error/critical
--detectionSize SIZE        Detection model input size (default: 2048)
--textThreshold FLOAT       Text detection confidence (default: 0.5)
--boxThreshold FLOAT        Box detection confidence (default: 0.7)
--unclipRatio FLOAT         Text region expansion ratio (default: 2.3)
--detInvert                 Invert detection input
--detGammaCorrect           Apply gamma correction to detection
--detRotate                 Enable rotation augmentation
--detAutoRotate             Auto-detect and rotate image
--verbose                   Enable verbose logging
```

---

## 🌐 API Documentation

### Endpoints

#### `GET /health`
Health check and system status.

**Response:**
```json
{
  "status": "ready|initializing|failed",
  "ready": true,
  "error": null,
  "detector_device": "cpu|cuda|mps",
  "translator_device": "cpu|cuda|mps",
  "gpu_device": "NVIDIA RTX 4090|null",
  "system_memory_gb": 16.0,
  "system_memory_used_gb": 4.2,
  "gpu_total_memory_gb": 24.0
}
```

**Status Codes:**
- `200`: Ready
- `503`: Not ready (initializing or failed)

---

#### `POST /translate`
Translate image to target language.

**Request:**
```json
{
  "imagePath": "/path/to/image.jpg",
  "postId": "optional_id",
  "targetLang": "ENG",
  "device": "auto",
  "detectionSize": 2048,
  "textThreshold": 0.5,
  "boxThreshold": 0.7,
  "unclipRatio": 2.3,
  "verbose": false
}
```

**Response (success, 200):**
```json
{
  "postId": "optional_id",
  "imagePath": "/path/to/image.jpg",
  "originalSize": {"width": 1280, "height": 960},
  "translator": "deep-translator",
  "elapsedMs": 2340,
  "timings": {
    "totalMs": 2340,
    "imageLoadMs": 45,
    "imageDecodeMs": 120,
    "detectMs": 890,
    "ocrMs": 640,
    "mergeMs": 180,
    "translateMs": 410,
    "normalizeMs": 15,
    "detectorDevice": "cpu",
    "translatorDevice": "cpu",
    "detectedTextlines": 12,
    "recognizedTextlines": 12,
    "mergedRegions": 8
  },
  "overlays": [...]
}
```

**Response (error, 400/503/500):**
```json
{
  "detail": "Error message"
}
```

**Status Codes:**
- `200`: Translation successful
- `400`: Invalid request (missing imagePath, invalid language, etc.)
- `503`: Pipeline not ready (initializing or failed)
- `500`: Internal error (model error, translation error, etc.)

---

## 📋 Error Handling

Exception hierarchy in [app/core/exceptions.py](app/exceptions.py):

| Exception | HTTP | Retryable | Description |
|-----------|------|-----------|-------------|
| `GlobalError` | 500 | Yes | Base error; retry recommended |
| `ModelNotReadyError` | 503 | Yes | Pipeline still initializing |
| `InvalidInputError` | 400 | No | Bad image path or parameters |
| `TranslationTimeoutError` | 504 | Yes | Translation took too long |
| `UnsupportedMediaError` | 415 | No | Image format not supported |
| `InternalPipelineError` | 500 | Yes | Detection/OCR/merge error |

**Error Response:**
```json
{
  "detail": "Error message describing the issue"
}
```

---

## 📚 Technology Stack

- **PyTorch** — Deep learning framework (detection, OCR)
- **FastAPI + Uvicorn** — REST API server
- **Pydantic** — Request/response validation
- **OpenCV** — Image processing
- **Pillow** — Image loading/manipulation
- **deep-translator** — Google Translate backend
- **NetworkX** — Graph-based text merging
- **Shapely, pyclipper** — Geometric operations
- **NumPy, scikit-image, einops** — Numerical operations

---

## 📄 License

This project is licensed under **GPL v3.0** (inherited from [manga-image-translator](https://github.com/zyddnys/manga-image-translator)).

### Attribution

**OverlayTranslator** is a modified fork of **manga-image-translator** by [zyddnys](https://github.com/zyddnys). The original project provides advanced manga translation with local LLM support. OverlayTranslator simplifies this for a specific use case: CPU-first, API-first manga overlay translation.

**Modifications include:**
- Replaced local LLM translator with Google Translate (deep-translator)
- Removed complex GPU optimization (layer probing, VRAM contention handling)
- Added FastAPI REST server mode with singleton pattern
- Streamlined configuration and CLI arguments
- Focused on 4-step pipeline clarity over advanced features

See [COPYING](COPYING) for full GPL v3.0 license text.

---

## 🚀 Building Standalone Executable

Use PyInstaller to create a single executable (no Python installation needed):

```bash
pip install pyinstaller
python build.py
```

Output: `dist/OverlayTranslator.exe` (Windows) or `dist/OverlayTranslator` (Linux/macOS)

**Note:** Executable includes all dependencies and models (~500MB). No virtual environment needed on target machine.

---

## 🧪 Testing

Run test suite:

```bash
pytest tests/ -v
```

- **test_cli.py** — CLI mode functionality
- **test_server.py** — REST API server tests
- **conftest.py** — Shared test fixtures

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make changes with clear commit messages
4. Submit a pull request

Remember: This is a GPL v3.0 project. Any derivative work must also be GPL v3.0.

---

## 📞 Support

- **Original Project:** [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)
- **Issues:** GitHub Issues (this repository)
- **Questions:** Refer to Architecture section or inline code comments

---

**Made for Emiru overlay system** | **Fork of manga-image-translator (GPL v3.0)** | **Python 3.11-3.12**
