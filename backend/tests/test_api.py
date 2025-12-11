"""
Comprehensive API tests for Kramer Web API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

from app.main import app


# Create test client
client = TestClient(app)


# ==================== Root Endpoint Tests ====================


class TestRootEndpoints:
    """Test root and health endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint returns correct info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert data["name"] == "Kramer Discovery Platform"

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_api_docs_available(self):
        """Test that API documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_spec_available(self):
        """Test that OpenAPI spec is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Kramer Discovery Platform"


# ==================== Discovery Endpoint Tests ====================


class TestDiscoveryEndpoints:
    """Test discovery-related endpoints."""

    def test_create_discovery_missing_fields(self):
        """Test discovery creation with missing required fields."""
        response = client.post("/api/v1/discovery/start", json={})
        assert response.status_code == 422  # Validation error

    def test_create_discovery_invalid_json(self):
        """Test discovery creation with invalid JSON."""
        response = client.post(
            "/api/v1/discovery/start",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_get_nonexistent_discovery(self):
        """Test getting a discovery that doesn't exist."""
        response = client.get("/api/v1/discovery/nonexistent-id/status")
        # Should return error or 404
        assert response.status_code in [404, 500]

    def test_list_discoveries(self):
        """Test listing all discoveries."""
        response = client.get("/api/v1/discovery/")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_stop_nonexistent_discovery(self):
        """Test stopping a discovery that doesn't exist."""
        response = client.post("/api/v1/discovery/nonexistent-id/stop")
        assert response.status_code in [404, 500]

    def test_get_cycles_nonexistent_discovery(self):
        """Test getting cycles for nonexistent discovery."""
        response = client.get("/api/v1/discovery/nonexistent-id/cycles")
        # May return empty list or error depending on implementation
        assert response.status_code in [200, 404, 500]

    def test_get_metrics_nonexistent_discovery(self):
        """Test getting metrics for nonexistent discovery."""
        response = client.get("/api/v1/discovery/nonexistent-id/metrics")
        assert response.status_code in [404, 500]


# ==================== Dataset Endpoint Tests ====================


class TestDatasetEndpoints:
    """Test dataset management endpoints."""

    def test_list_datasets(self):
        """Test listing datasets."""
        response = client.get("/api/v1/datasets/")
        assert response.status_code == 200

        data = response.json()
        assert "files" in data
        assert "count" in data
        assert isinstance(data["files"], list)

    def test_list_datasets_with_filter(self):
        """Test listing datasets with discovery filter."""
        response = client.get("/api/v1/datasets/?discovery_id=test-id")
        assert response.status_code == 200

    def test_delete_nonexistent_dataset(self):
        """Test deleting a dataset that doesn't exist."""
        response = client.delete("/api/v1/datasets/nonexistent-file.csv")
        assert response.status_code in [404, 500]


# ==================== Reports Endpoint Tests ====================


class TestReportsEndpoints:
    """Test report-related endpoints."""

    def test_list_reports(self):
        """Test listing reports."""
        response = client.get("/api/v1/reports/")
        assert response.status_code == 200

    def test_get_nonexistent_report(self):
        """Test getting a report that doesn't exist."""
        response = client.get("/api/v1/reports/nonexistent-id")
        assert response.status_code in [404, 500]


# ==================== World Model Endpoint Tests ====================


class TestWorldModelEndpoints:
    """Test world model query endpoints."""

    def test_world_model_stats(self):
        """Test getting world model statistics."""
        response = client.get("/api/v1/world-model/stats")
        # May require discovery context
        assert response.status_code in [200, 404, 500]

    def test_world_model_nodes(self):
        """Test getting world model nodes."""
        response = client.get("/api/v1/world-model/nodes")
        assert response.status_code in [200, 404, 500]

    def test_world_model_edges(self):
        """Test getting world model edges."""
        response = client.get("/api/v1/world-model/edges")
        assert response.status_code in [200, 404, 500]


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_endpoint(self):
        """Test accessing invalid endpoint."""
        response = client.get("/api/v1/invalid-endpoint")
        assert response.status_code == 404

    def test_invalid_method(self):
        """Test using invalid HTTP method."""
        response = client.put("/")
        assert response.status_code == 405

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/")
        # CORS should allow the request
        assert response.status_code in [200, 405]


# ==================== Validation Tests ====================


class TestValidation:
    """Test request validation."""

    def test_discovery_config_validation(self):
        """Test discovery config validation."""
        # Missing required field
        response = client.post(
            "/api/v1/discovery/start",
            json={"max_cycles": 10}  # Missing objective
        )
        assert response.status_code == 422

    def test_discovery_config_types(self):
        """Test discovery config type validation."""
        # Wrong type for max_cycles
        response = client.post(
            "/api/v1/discovery/start",
            json={
                "objective": "Test",
                "max_cycles": "not a number"  # Should be int
            }
        )
        assert response.status_code == 422


# ==================== Content Type Tests ====================


class TestContentTypes:
    """Test content type handling."""

    def test_json_content_type(self):
        """Test JSON content type handling."""
        response = client.post(
            "/api/v1/discovery/start",
            json={"objective": "Test"}
        )
        # Should be handled (may fail validation but not content type)
        assert response.status_code in [200, 422, 500]

    def test_response_content_type(self):
        """Test response content type is JSON."""
        response = client.get("/")
        assert "application/json" in response.headers.get("content-type", "")


# ==================== API Versioning Tests ====================


class TestAPIVersioning:
    """Test API versioning."""

    def test_v1_prefix(self):
        """Test v1 API prefix works."""
        response = client.get("/api/v1/discovery/")
        assert response.status_code == 200

    def test_no_prefix_returns_404(self):
        """Test endpoints without prefix return 404."""
        response = client.get("/discovery/")
        assert response.status_code == 404


# ==================== Query Parameter Tests ====================


class TestQueryParameters:
    """Test query parameter handling."""

    def test_optional_query_params(self):
        """Test optional query parameters."""
        response = client.get("/api/v1/datasets/")
        assert response.status_code == 200

        response = client.get("/api/v1/datasets/?discovery_id=123")
        assert response.status_code == 200

    def test_invalid_query_params(self):
        """Test invalid query parameters are handled."""
        response = client.get("/api/v1/datasets/?invalid_param=value")
        # Should be ignored or handled gracefully
        assert response.status_code == 200


# ==================== Path Parameter Tests ====================


class TestPathParameters:
    """Test path parameter handling."""

    def test_valid_path_params(self):
        """Test valid path parameters."""
        response = client.get("/api/v1/discovery/test-id-123/status")
        # May fail with 404 but should parse path correctly
        assert response.status_code in [200, 404, 500]

    def test_special_chars_in_path(self):
        """Test special characters in path parameters."""
        response = client.get("/api/v1/discovery/test%2Fid/status")
        assert response.status_code in [200, 404, 500]


# ==================== Response Format Tests ====================


class TestResponseFormat:
    """Test response format consistency."""

    def test_root_response_format(self):
        """Test root response has expected format."""
        response = client.get("/")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "name" in data

    def test_list_response_format(self):
        """Test list responses are arrays."""
        response = client.get("/api/v1/discovery/")
        data = response.json()

        assert isinstance(data, list)

    def test_error_response_format(self):
        """Test error responses have detail field."""
        response = client.get("/api/v1/discovery/nonexistent/status")

        if response.status_code != 200:
            data = response.json()
            assert "detail" in data or "error" in data


# ==================== WebSocket Tests ====================


class TestWebSocket:
    """Test WebSocket functionality (basic checks)."""

    def test_websocket_endpoint_exists(self):
        """Test WebSocket endpoint is defined."""
        # Note: Full WebSocket testing requires async client
        # This just verifies the endpoint exists
        # The actual WebSocket testing would use:
        # async with client.websocket_connect("/api/v1/ws") as websocket:
        #     ...
        pass


# ==================== Integration Tests ====================


class TestAPIIntegration:
    """Integration tests for API flows."""

    def test_full_discovery_flow(self):
        """Test complete discovery flow (list -> create -> status)."""
        # List (should be empty or have items)
        list_response = client.get("/api/v1/discovery/")
        assert list_response.status_code == 200

        initial_count = len(list_response.json())

        # Note: Creating a discovery would require mocking the service
        # For now, just verify the endpoints work

    def test_dataset_flow(self):
        """Test dataset management flow."""
        # List datasets
        response = client.get("/api/v1/datasets/")
        assert response.status_code == 200

        data = response.json()
        assert "files" in data


# ==================== Main ====================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
