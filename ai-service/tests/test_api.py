import pytest
import os
import json

os.environ["ENABLE_LOCAL_EMBEDDER"] = "true"

from main import app, init_embedder

@pytest.fixture
def client():
    app.config["TESTING"] = True
    init_embedder()
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["embedder"]["enabled"] is True
    assert data["embedder"]["status"] == "loaded"
    assert data["embedder"]["dimension"] == 384
    assert data["embedder"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"

def test_embed_endpoint_success(client):
    payload = {"text": "Senior Full-Stack Engineer with React, Node.js and PostgreSQL"}
    res = client.post("/embed", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "embedding" in data
    assert data["dim"] == 384
    assert len(data["embedding"]) == 384
    assert data["model"] == "sentence-transformers/all-MiniLM-L6-v2"

def test_embed_endpoint_empty_text(client):
    res = client.post("/embed", json={"text": ""})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data

    res_spaces = client.post("/embed", json={"text": "   "})
    assert res_spaces.status_code == 400

    res_missing = client.post("/embed", json={})
    assert res_missing.status_code == 400

def test_embed_batch_endpoint(client):
    payload = {"texts": ["Frontend Developer", "Backend Developer", "DevOps Engineer"]}
    res = client.post("/embed/batch", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["count"] == 3
    assert data["dim"] == 384
    assert len(data["embeddings"]) == 3

def test_embed_batch_empty_list(client):
    res = client.post("/embed/batch", json={"texts": []})
    assert res.status_code == 400

def test_skills_extract_fallback(client):
    res = client.post("/skills/extract", json={"text": "Looking for a developer with Java, Spring Boot, Docker, and React skills."})
    assert res.status_code == 200
    data = res.get_json()
    assert "skills" in data
    assert "Java" in data["skills"]
    assert "Spring Boot" in data["skills"]
    assert "Docker" in data["skills"]
    assert "React" in data["skills"]
