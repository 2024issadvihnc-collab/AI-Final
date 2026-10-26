import pytest
import numpy as np
import os

# Ensure ENABLE_LOCAL_EMBEDDER is true for tests
os.environ["ENABLE_LOCAL_EMBEDDER"] = "true"

from models.embedder import EmbedderModel

def test_model_loads_correctly():
    model = EmbedderModel()
    assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert model.dimension == 384
    assert model.backend in ("sentence-transformers", "onnxruntime")

def test_embedding_dimension_and_normalization():
    model = EmbedderModel()
    text = "Full Stack Java Developer proficient in Spring Boot, PostgreSQL, and React."
    vec = model.embed(text)
    assert len(vec) == 384
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)

    # Check L2 normalization (norm should be ~1.0)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0

def test_empty_input_rejection():
    model = EmbedderModel()
    with pytest.raises(ValueError):
        model.embed("")
    with pytest.raises(ValueError):
        model.embed("    \n\t  ")

def test_batch_embedding():
    model = EmbedderModel()
    texts = [
        "Backend engineer building microservices in Java",
        "Frontend developer building UI in React and TypeScript",
        "Data scientist training computer vision models"
    ]
    vecs = model.embed_batch(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 384
        assert pytest.approx(np.linalg.norm(v), abs=1e-3) == 1.0

def test_semantic_textual_similarity_relevance():
    """
    Verify that semantically related text has significantly higher cosine similarity
    than unrelated text.
    """
    model = EmbedderModel()
    resume_java = "Experienced Java software developer skilled in Spring Boot, PostgreSQL, Docker, and REST APIs building backend systems."
    job_java = "Senior Java Backend Engineer with Spring Boot and relational databases experience."
    job_chef = "Head Pastry Chef specializing in artisan sourdough breads and Italian desserts."
    job_designer = "Lead Graphic Designer skilled in Figma, typography, branding, and Adobe Illustrator."

    vec_resume = np.array(model.embed(resume_java))
    vec_job_java = np.array(model.embed(job_java))
    vec_job_chef = np.array(model.embed(job_chef))
    vec_job_designer = np.array(model.embed(job_designer))

    sim_java = float(np.dot(vec_resume, vec_job_java))
    sim_chef = float(np.dot(vec_resume, vec_job_chef))
    sim_designer = float(np.dot(vec_resume, vec_job_designer))

    print(f"\nJava Developer vs Java Job Cosine Similarity: {sim_java:.4f}")
    print(f"Java Developer vs Chef Job Cosine Similarity: {sim_chef:.4f}")
    print(f"Java Developer vs Designer Job Cosine Similarity: {sim_designer:.4f}")

    assert sim_java > 0.60, f"Expected strong similarity for matching job, got {sim_java}"
    assert sim_chef < 0.30, f"Expected weak similarity for unrelated job, got {sim_chef}"
    assert sim_designer < 0.40, f"Expected weak similarity for unrelated job, got {sim_designer}"
    assert sim_java > sim_designer > sim_chef or sim_java > sim_chef
    assert sim_java - sim_chef > 0.40, "Expected at least 0.40 delta between related and unrelated jobs"

def test_oversized_text_chunking():
    """
    Verify that long documents exceeding 1200 chars are chunked and mean-pooled
    without error, producing a valid 384-dimensional unit vector.
    """
    model = EmbedderModel()
    long_resume = ("Senior Software Architect with 10 years experience. " * 30) + (
        "Core skills: Python, Machine Learning, PyTorch, Sentence Transformers, PGVector. " * 10
    )
    assert len(long_resume) > 2000
    vec = model.embed(long_resume)
    assert len(vec) == 384
    assert pytest.approx(np.linalg.norm(vec), abs=1e-3) == 1.0
