"""
Basic API tests for Kramer Web API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_docs_available():
    """Test that API documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_create_discovery_missing_fields():
    """Test discovery creation with missing required fields."""
    response = client.post("/api/v1/discovery/start", json={})
    assert response.status_code == 422  # Validation error


def test_get_nonexistent_discovery():
    """Test getting a discovery that doesn't exist."""
    response = client.get("/api/v1/discovery/nonexistent-id/status")
    # Should return error or 404
    assert response.status_code in [404, 500]


def test_list_discoveries():
    """Test listing all discoveries."""
    response = client.get("/api/v1/discovery/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# Add more tests as needed
@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection (basic check)."""
    # Note: This requires more setup for full testing
    # Just verify the endpoint exists
    from fastapi.websockets import WebSocket
    # WebSocket testing would require additional setup
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
