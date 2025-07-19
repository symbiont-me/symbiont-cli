"""
Test configuration and fixtures for e2e tests
"""
import pytest
import tempfile
import shutil
import os
import json
import toml
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import time

@pytest.fixture
def runner():
    """CLI test runner"""
    return CliRunner()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_config():
    """Sample configuration for tests"""
    return {
        "general": {
            "llm_name": "gpt-4o-mini"
        },
        "embeddings": {
            "model": "huggingface",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "reranker": {
            "reranker": "huggingface"
        },
        "qdrant": {
            "host": "localhost", 
            "port": 6333
        }
    }

@pytest.fixture
def test_config_file(temp_dir, sample_config):
    """Create test config file"""
    config_path = os.path.join(temp_dir, "config.toml")
    with open(config_path, "w") as f:
        toml.dump(sample_config, f)
    return config_path

@pytest.fixture
def test_docs_dir(temp_dir):
    """Create test documents directory with sample PDF content"""
    docs_dir = os.path.join(temp_dir, "test_docs")
    os.makedirs(docs_dir)
    
    # Create a simple test PDF file using reportlab if available
    test_pdf_path = os.path.join(docs_dir, "test_document.pdf")
    
    # Create a minimal PDF content for testing
    # This is a simple approach - we'll use a text file with .pdf extension for basic testing
    with open(test_pdf_path, "w") as f:
        f.write("This is test PDF content for symbiont-cli testing.")
    
    return docs_dir

@pytest.fixture
def sample_questions():
    """Sample questions for testing"""
    return [
        "What is the main topic?",
        "How does this relate to the subject?",
        "What are the key points mentioned?"
    ]

@pytest.fixture
def test_questions_file(temp_dir, sample_questions):
    """Create test questions file"""
    questions_path = os.path.join(temp_dir, "questions.txt")
    with open(questions_path, "w") as f:
        for q in sample_questions:
            f.write(f"{q}\n")
    return questions_path

@pytest.fixture
def test_questions_json(temp_dir, sample_questions):
    """Create test questions JSON file"""
    questions_path = os.path.join(temp_dir, "questions.json")
    with open(questions_path, "w") as f:
        json.dump({"questions": sample_questions}, f)
    return questions_path

@pytest.fixture
def mock_openai_key():
    """Mock OpenAI API key"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        yield "test_key"

@pytest.fixture
def qdrant_client():
    """Real Qdrant client for integration tests"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        # Test connection
        client.get_collections()
        yield client
    except (ConnectionError, UnexpectedResponse):
        pytest.skip("Qdrant server not available")

@pytest.fixture
def test_collection_name():
    """Generate unique test collection name"""
    import uuid
    return f"test_collection_{uuid.uuid4().hex[:8]}"

@pytest.fixture
def cleanup_collections(qdrant_client):
    """Cleanup test collections after tests"""
    created_collections = []
    
    def _add_collection(name):
        created_collections.append(name)
    
    yield _add_collection
    
    # Cleanup
    for collection_name in created_collections:
        try:
            qdrant_client.delete_collection(collection_name)
        except:
            pass  # Collection might not exist

@pytest.fixture
def isolated_config(temp_dir, monkeypatch):
    """Run tests in isolated configuration directory"""
    original_cwd = os.getcwd()
    monkeypatch.chdir(temp_dir)
    yield temp_dir
    monkeypatch.chdir(original_cwd)

@pytest.fixture(scope="session")
def docker_qdrant():
    """Start Qdrant in Docker for isolated testing"""
    import subprocess
    import time
    
    # Check if docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Docker not available")
    
    # Start Qdrant container
    container_name = "test_qdrant_symbiont"
    try:
        # Stop existing container if running
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        
        # Start new container
        subprocess.run([
            "docker", "run", "-d", "--name", container_name,
            "-p", "6334:6333",  # Use different port to avoid conflicts
            "qdrant/qdrant"
        ], check=True, capture_output=True)
        
        # Wait for container to be ready
        time.sleep(10)
        
        # Test connection
        test_client = QdrantClient(host="localhost", port=6334)
        for _ in range(30):  # Wait up to 30 seconds
            try:
                test_client.get_collections()
                break
            except:
                time.sleep(1)
        else:
            raise Exception("Qdrant container failed to start")
        
        yield "localhost:6334"
        
    finally:
        # Cleanup
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

@pytest.fixture
def mock_expensive_operations():
    """Mock expensive operations for faster tests"""
    with patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("symbiont_cli.main.ChatOpenAI") as mock_llm, \
         patch("symbiont_cli.main.DirectoryLoader") as mock_loader:
        
        # Configure mocks to return reasonable test values
        mock_embeddings_instance = mock_embeddings.return_value
        mock_embeddings_instance.embed_documents.return_value = [[0.1] * 384]  # 384-dim vector
        mock_embeddings_instance.embed_query.return_value = [0.1] * 384
        
        mock_llm_instance = mock_llm.return_value
        mock_llm_instance.invoke.return_value.content = "Test response from LLM"
        
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.load.return_value = [
            type('Document', (), {
                'page_content': 'Test document content',
                'metadata': {'source': 'test.pdf', 'title': 'Test', 'page': 1}
            })()
        ]
        
        yield {
            "embeddings": mock_embeddings_instance,
            "llm": mock_llm_instance, 
            "loader": mock_loader_instance
        }