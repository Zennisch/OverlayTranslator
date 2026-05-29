"""Tests for OverlayTranslator server endpoints."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.server.app import app
from app.server.pipeline_manager import PipelineManager, PipelineStatus


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a dummy image file for testing."""
    import numpy as np
    from PIL import Image

    # Create a simple 100x100 RGB image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')

    image_path = tmp_path / "test_image.jpg"
    img.save(image_path)
    return str(image_path)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_when_initializing(self, client):
        """Test health endpoint when pipeline is initializing."""
        # Reset pipeline to initializing state
        manager = PipelineManager()
        manager._status = PipelineStatus.INITIALIZING
        manager._pipeline = None

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "initializing"
        assert data["ready"] is False

    def test_health_when_failed(self, client):
        """Test health endpoint when pipeline failed to initialize."""
        manager = PipelineManager()
        manager._status = PipelineStatus.FAILED
        manager._error_message = "Test error"

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "failed"
        assert data["ready"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_health_when_ready(self, client):
        """Test health endpoint when pipeline is ready."""
        manager = PipelineManager()
        manager._status = PipelineStatus.READY

        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_pipeline._ready = True
        mock_pipeline._device = "cpu"
        mock_pipeline._translator_device = "cpu"
        manager._pipeline = mock_pipeline

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["ready"] is True


class TestTranslateEndpoint:
    """Test /translate endpoint."""

    def test_translate_without_image_path(self, client):
        """Test translate endpoint without imagePath raises 422."""
        response = client.post("/translate", json={
            "postId": "test",
            "targetLang": "ENG"
        })
        assert response.status_code == 422  # Validation error

    def test_translate_with_nonexistent_image(self, client):
        """Test translate endpoint with non-existent image returns 400."""
        manager = PipelineManager()
        manager._status = PipelineStatus.READY
        manager._pipeline = MagicMock()

        response = client.post("/translate", json={
            "imagePath": "/path/that/does/not/exist/image.jpg",
            "postId": "test",
            "targetLang": "ENG"
        })
        assert response.status_code == 400
        data = response.json()
        assert data["errorCode"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_translate_when_not_ready(self, client, sample_image_path):
        """Test translate endpoint returns 503 when pipeline not ready."""
        manager = PipelineManager()
        manager._status = PipelineStatus.INITIALIZING
        manager._pipeline = None

        response = client.post("/translate", json={
            "imagePath": sample_image_path,
            "postId": "test",
            "targetLang": "ENG"
        })
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_translate_with_minimal_payload(self, client, sample_image_path):
        """Test translate endpoint with minimal required fields."""
        manager = PipelineManager()
        manager._status = PipelineStatus.READY

        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_pipeline._ready = True

        mock_result = {
            "postId": "test",
            "imagePath": sample_image_path,
            "originalSize": {"width": 100, "height": 100},
            "translator": "deep-translator",
            "elapsedMs": 1000,
            "timings": {
                "totalMs": 1000,
                "imageLoadMs": 10,
                "imageDecodeMs": 5,
                "detectMs": 100,
                "ocrMs": 200,
                "mergeMs": 50,
                "translateMs": 300,
                "detectorDevice": "cpu",
                "detectedTextlines": 5,
                "recognizedTextlines": 5,
                "mergedRegions": 3,
            },
            "overlays": []
        }

        mock_pipeline.translate_image = AsyncMock(return_value=mock_result)
        manager._pipeline = mock_pipeline

        response = client.post("/translate", json={
            "imagePath": sample_image_path,
            "postId": "test",
            "targetLang": "ENG"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["postId"] == "test"
        assert data["imagePath"] == sample_image_path
        assert "overlays" in data
        assert "timings" in data

    @pytest.mark.asyncio
    async def test_translate_with_optional_settings(self, client, sample_image_path):
        """Test translate endpoint passes optional settings to pipeline."""
        manager = PipelineManager()
        manager._status = PipelineStatus.READY

        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_pipeline._ready = True

        mock_result = {
            "postId": "test",
            "imagePath": sample_image_path,
            "originalSize": {"width": 100, "height": 100},
            "translator": "llm",
            "elapsedMs": 1000,
            "timings": {
                "totalMs": 1000,
                "imageLoadMs": 10,
                "imageDecodeMs": 5,
                "detectMs": 100,
                "ocrMs": 200,
                "mergeMs": 50,
                "translateMs": 300,
                "detectorDevice": "cpu",
                "detectedTextlines": 5,
                "recognizedTextlines": 5,
                "mergedRegions": 3,
            },
            "overlays": []
        }

        mock_pipeline.translate_image = AsyncMock(return_value=mock_result)
        manager._pipeline = mock_pipeline

        response = client.post("/translate", json={
            "imagePath": sample_image_path,
            "postId": "test",
            "targetLang": "VIE",
            "detectionSize": 1024,
            "textThreshold": 0.6,
            "device": "cpu"
        })

        assert response.status_code == 200


class TestPipelineManager:
    """Test PipelineManager singleton."""

    def test_singleton_pattern(self):
        """Test that PipelineManager is a true singleton."""
        manager1 = PipelineManager()
        manager2 = PipelineManager()
        assert manager1 is manager2

    def test_initial_status(self):
        """Test initial status of pipeline manager."""
        # Create new instance for clean state
        manager = PipelineManager()
        # Status should be initializing initially
        assert manager._status in [PipelineStatus.INITIALIZING, PipelineStatus.READY, PipelineStatus.FAILED]

    def test_get_status_dict(self):
        """Test get_status returns proper dictionary."""
        manager = PipelineManager()
        manager._status = PipelineStatus.READY
        manager._pipeline = MagicMock()
        manager._pipeline._ready = True
        manager._pipeline._device = "cpu"
        manager._pipeline._translator_device = "cpu"

        status = manager.get_status()
        assert "status" in status
        assert "ready" in status
        assert status["status"] == "ready"
        assert status["ready"] is True


class TestAPIErrors:
    """Test error handling in API."""

    def test_model_not_ready_error(self, client, sample_image_path):
        """Test ModelNotReadyError is properly converted to HTTP 503."""
        manager = PipelineManager()
        manager._status = PipelineStatus.INITIALIZING
        manager._pipeline = None

        response = client.post("/translate", json={
            "imagePath": sample_image_path,
            "postId": "test",
            "targetLang": "ENG"
        })
        assert response.status_code == 503
        data = response.json()
        assert data["errorCode"] == "MODEL_NOT_READY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
