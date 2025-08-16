"""
Integration tests that use real external services
These tests require actual services to be running
"""
import pytest
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from symbiont_cli.main import SymbiontCLI


@pytest.mark.integration
class TestRealQdrantIntegration:
    """Tests that use a real Qdrant instance"""

    def test_create_and_use_real_collection(self, temp_dir, mock_openai_key, sample_config, 
                                           qdrant_client, test_collection_name, cleanup_collections):
        """Test creating a real collection in Qdrant"""
        cleanup_collections(test_collection_name)
        
        # Create test documents
        docs_dir = os.path.join(temp_dir, "real_docs")
        os.makedirs(docs_dir)
        
        # Create a simple text file (treating as PDF for testing)
        test_file = os.path.join(docs_dir, "test.pdf")
        with open(test_file, "w") as f:
            f.write("This is a test document about machine learning and artificial intelligence.")
        
        with patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm:
            
            # Mock document loader to return our test content
            from types import SimpleNamespace
            mock_doc = SimpleNamespace()
            mock_doc.page_content = "This is a test document about machine learning and artificial intelligence."
            mock_doc.metadata = {"source": test_file, "title": "Test Doc", "page": 1}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            
            # Mock LLM for faster testing
            mock_llm.return_value.invoke.return_value.content = "Machine learning is a subset of AI."
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Test collection creation and document processing
                cli = SymbiontCLI(
                    docs_directory=docs_dir,
                    collection_name=test_collection_name,
                    k_value=3,
                    llm_response="no",  # Disable LLM for faster testing
                    output_directory="search_results",
                    q_list=None
                )
                
                # Verify collection was created in real Qdrant
                assert qdrant_client.collection_exists(test_collection_name)
                
                # Verify collection has documents
                collection_info = qdrant_client.get_collection(test_collection_name)
                assert collection_info.vectors_count > 0
                
                # Test search functionality
                with patch("symbiont_cli.main.ContextualCompressionRetriever") as mock_compression:
                    mock_doc_result = SimpleNamespace()
                    mock_doc_result.page_content = "Machine learning content"
                    mock_doc_result.metadata = {
                        "source": test_file, 
                        "title": "Test Doc", 
                        "page": 1,
                        "relevance_score": 0.95
                    }
                    
                    mock_compression.return_value.invoke.return_value = [mock_doc_result]
                    
                    # Perform a search
                    cli.perform_search_and_qa("What is machine learning?")
                    
                    # Verify search was performed
                    mock_compression.return_value.invoke.assert_called_once()
                
            finally:
                os.chdir(original_cwd)

    def test_real_collection_persistence(self, temp_dir, mock_openai_key, sample_config,
                                        qdrant_client, test_collection_name, cleanup_collections):
        """Test that collections persist between CLI instances"""
        cleanup_collections(test_collection_name)
        
        docs_dir = os.path.join(temp_dir, "persistent_docs")
        os.makedirs(docs_dir)
        
        # Create test document
        test_file = os.path.join(docs_dir, "persistent.pdf")
        with open(test_file, "w") as f:
            f.write("Persistent test document content.")
        
        with patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm:
            
            # Mock document loader
            from types import SimpleNamespace
            mock_doc = SimpleNamespace()
            mock_doc.page_content = "Persistent test document content."
            mock_doc.metadata = {"source": test_file, "title": "Persistent Doc", "page": 1}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # First CLI instance - create collection
                cli1 = SymbiontCLI(
                    docs_directory=docs_dir,
                    collection_name=test_collection_name,
                    k_value=3,
                    llm_response="no",
                    output_directory="search_results",
                    q_list=None
                )
                
                # Verify collection exists
                assert qdrant_client.collection_exists(test_collection_name)
                initial_count = qdrant_client.get_collection(test_collection_name).vectors_count
                
                # Second CLI instance - should use existing collection
                cli2 = SymbiontCLI(
                    docs_directory=docs_dir,
                    collection_name=test_collection_name,
                    k_value=3,
                    llm_response="no",
                    output_directory="search_results",
                    q_list=None
                )
                
                # Verify collection still exists with same content
                assert qdrant_client.collection_exists(test_collection_name)
                final_count = qdrant_client.get_collection(test_collection_name).vectors_count
                assert final_count == initial_count  # No duplicate documents
                
            finally:
                os.chdir(original_cwd)

    def test_real_search_and_similarity(self, temp_dir, mock_openai_key, sample_config,
                                       qdrant_client, test_collection_name, cleanup_collections):
        """Test real similarity search with actual embeddings"""
        cleanup_collections(test_collection_name)
        
        docs_dir = os.path.join(temp_dir, "similarity_docs")
        os.makedirs(docs_dir)
        
        # Create multiple test documents with different content
        test_docs = [
            ("machine_learning.pdf", "Machine learning is a method of data analysis that automates analytical model building."),
            ("artificial_intelligence.pdf", "Artificial intelligence refers to the simulation of human intelligence in machines."),
            ("deep_learning.pdf", "Deep learning is a subset of machine learning that uses neural networks."),
            ("programming.pdf", "Programming is the process of creating computer software using programming languages.")
        ]
        
        for filename, content in test_docs:
            with open(os.path.join(docs_dir, filename), "w") as f:
                f.write(content)
        
        with patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm:
            
            # Mock document loader to return our test documents
            from types import SimpleNamespace
            mock_documents = []
            for filename, content in test_docs:
                mock_doc = SimpleNamespace()
                mock_doc.page_content = content
                mock_doc.metadata = {"source": os.path.join(docs_dir, filename), "title": filename, "page": 1}
                mock_documents.append(mock_doc)
            
            mock_loader.return_value.load.return_value = mock_documents
            
            # Create config file with real HuggingFace embeddings
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Create CLI with real embeddings (but mock LLM for speed)
                cli = SymbiontCLI(
                    docs_directory=docs_dir,
                    collection_name=test_collection_name,
                    k_value=3,
                    llm_response="no",
                    output_directory="search_results",
                    q_list=None
                )
                
                # Verify collection was created and populated
                assert qdrant_client.collection_exists(test_collection_name)
                collection_info = qdrant_client.get_collection(test_collection_name)
                assert collection_info.vectors_count == len(test_docs)
                
                # Test similarity search directly on vector store
                search_results = cli.vector_store.similarity_search("neural networks", k=2)
                
                # Verify search returns relevant results
                assert len(search_results) > 0
                
                # The most relevant result should be about deep learning or machine learning
                top_result = search_results[0]
                assert any(term in top_result.page_content.lower() 
                          for term in ["deep learning", "machine learning", "neural"])
                
            finally:
                os.chdir(original_cwd)

    def test_collection_management_operations(self, temp_dir, mock_openai_key, sample_config,
                                            qdrant_client, cleanup_collections):
        """Test collection management operations with real Qdrant"""
        test_collections = [f"mgmt_test_{i}" for i in range(3)]
        for collection_name in test_collections:
            cleanup_collections(collection_name)
        
        docs_dir = os.path.join(temp_dir, "mgmt_docs")
        os.makedirs(docs_dir)
        
        # Create test document
        test_file = os.path.join(docs_dir, "mgmt_test.pdf")
        with open(test_file, "w") as f:
            f.write("Management test document.")
        
        with patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm:
            
            # Mock document loader
            from types import SimpleNamespace
            mock_doc = SimpleNamespace()
            mock_doc.page_content = "Management test document."
            mock_doc.metadata = {"source": test_file, "title": "Mgmt Test", "page": 1}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Create multiple collections
                for collection_name in test_collections:
                    cli = SymbiontCLI(
                        docs_directory=docs_dir,
                        collection_name=collection_name,
                        k_value=3,
                        llm_response="no",
                        output_directory="search_results",
                        q_list=None
                    )
                    
                    # Verify collection was created
                    assert qdrant_client.collection_exists(collection_name)
                
                # Test listing collections
                collections = qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                for test_collection in test_collections:
                    assert test_collection in collection_names
                
                # Test collection info
                for collection_name in test_collections:
                    info = qdrant_client.get_collection(collection_name)
                    assert info.vectors_count > 0
                    assert info.status == "green"
                
                # Test collection deletion
                for collection_name in test_collections:
                    qdrant_client.delete_collection(collection_name)
                    assert not qdrant_client.collection_exists(collection_name)
                
            finally:
                os.chdir(original_cwd)


@pytest.mark.docker
class TestDockerQdrantIntegration:
    """Tests that use Qdrant running in Docker"""

    def test_docker_qdrant_basic_operations(self, docker_qdrant, temp_dir, mock_openai_key, sample_config):
        """Test basic operations with Qdrant in Docker"""
        # Update config to use Docker Qdrant port
        docker_config = sample_config.copy()
        docker_config["qdrant"]["port"] = 6334
        
        docs_dir = os.path.join(temp_dir, "docker_docs")
        os.makedirs(docs_dir)
        
        # Create test document
        test_file = os.path.join(docs_dir, "docker_test.pdf")
        with open(test_file, "w") as f:
            f.write("Docker test document content.")
        
        with patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm:
            
            # Mock document loader
            from types import SimpleNamespace
            mock_doc = SimpleNamespace()
            mock_doc.page_content = "Docker test document content."
            mock_doc.metadata = {"source": test_file, "title": "Docker Test", "page": 1}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(docker_config, f)
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Test with Docker Qdrant
                cli = SymbiontCLI(
                    docs_directory=docs_dir,
                    collection_name="docker_test_collection",
                    k_value=3,
                    llm_response="no",
                    output_directory="search_results",
                    q_list=None
                )
                
                # Verify we can connect to Docker Qdrant
                docker_client = QdrantClient(host="localhost", port=6334)
                assert docker_client.collection_exists("docker_test_collection")
                
                # Verify collection has content
                info = docker_client.get_collection("docker_test_collection")
                assert info.vectors_count > 0
                
                # Cleanup
                docker_client.delete_collection("docker_test_collection")
                
            finally:
                os.chdir(original_cwd)


@pytest.mark.openai
class TestOpenAIIntegration:
    """Tests that use real OpenAI API (requires API key)"""

    @pytest.fixture(autouse=True)
    def check_openai_key(self):
        """Skip tests if no real OpenAI API key"""
        if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test_key":
            pytest.skip("Real OpenAI API key required for this test")

    def test_real_openai_llm_response(self, temp_dir, sample_config, qdrant_client, 
                                     test_collection_name, cleanup_collections):
        """Test with real OpenAI LLM responses"""
        cleanup_collections(test_collection_name)
        
        docs_dir = os.path.join(temp_dir, "openai_docs")
        os.makedirs(docs_dir)
        
        # Create test document with specific content
        test_file = os.path.join(docs_dir, "openai_test.pdf")
        with open(test_file, "w") as f:
            f.write("Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991.")
        
        with patch("symbiont_cli.main.DirectoryLoader") as mock_loader:
            
            # Mock document loader
            from types import SimpleNamespace
            mock_doc = SimpleNamespace()
            mock_doc.page_content = "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991."
            mock_doc.metadata = {"source": test_file, "title": "Python Info", "page": 1}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            # Create logs directory
            logs_dir = os.path.join(temp_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Test with real OpenAI LLM
                cli = SymbiontCLI(
                    docs_directory=docs_dir,
                    collection_name=test_collection_name,
                    k_value=3,
                    llm_response="yes",  # Enable real LLM
                    output_directory="search_results",
                    q_list=None
                )
                
                # Ask a question that should get a real response
                cli.perform_search_and_qa("Who created Python?")
                
                # Check that log file contains a real response
                log_file = os.path.join(logs_dir, f"{test_collection_name}.txt")
                assert os.path.exists(log_file)
                
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    # Should contain the question and a response about Guido van Rossum
                    assert "Who created Python?" in log_content
                    assert "Guido" in log_content or "van Rossum" in log_content
                
            finally:
                os.chdir(original_cwd)