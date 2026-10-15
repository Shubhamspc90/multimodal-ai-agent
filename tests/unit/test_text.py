from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_text_endpoint_accepts_valid_message() -> None:
    response = client.post(
        "/api/v1/text",
        json={"message": "  Hello agent  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "received_message": "Hello agent",
    }


def test_text_endpoint_rejects_empty_message() -> None:
    response = client.post(
        "/api/v1/text",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_text_endpoint_rejects_whitespace_only_message() -> None:
    response = client.post(
        "/api/v1/text",
        json={"message": "   "},
    )

    assert response.status_code == 422


def test_text_endpoint_rejects_oversized_message() -> None:
    response = client.post(
        "/api/v1/text",
        json={"message": "a" * 10_001},
    )

    assert response.status_code == 422
