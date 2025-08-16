"""
End-to-end tests for document processing pipeline
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from qdrant_client import QdrantClient

from symbiont_cli.main import SymbiontCLI


class TestDocumentProcessing:
    """Test document processing pipeline end-to-end"""

    def test_vector_store_creation_new_collection(self, temp_dir, mock_openai_key, 
                                                 sample_config, test_docs_dir):
        """Test creating a new vector store collection"""
        collection_name = "test_new_collection"
        
        with patch("symbiont_cli.main.QdrantClient") as mock_client_class, \
             patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_embeddings, \
             patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm:
            
            # Configure mocks
            mock_client = mock_client_class.return_value
            mock_client.collection_exists.return_value = False
            mock_client.create_collection.return_value = None
            
            mock_embeddings_instance = mock_embeddings.return_value
            
            mock_loader_instance = mock_loader.return_value
            mock_loader_instance.load.return_value = [
                MagicMock(
                    page_content="Test document content",
                    metadata={"source": "test.pdf", "title": "Test Doc", "page": 1}
                )
            ]
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                cli = SymbiontCLI(
                    docs_directory=test_docs_dir,
                    collection_name=collection_name,
                    k_value=5,
                    llm_response="yes",
                    output_directory="search_results",
                    q_list=None
                )
                
                # Verify collection was created
                mock_client.create_collection.assert_called_once()
                mock_loader.assert_called_once()
                
            finally:
                os.chdir(original_cwd)

    def test_vector_store_existing_collection(self, temp_dir, mock_openai_key,
                                            sample_config, test_docs_dir):
        """Test using existing vector store collection"""
        collection_name = "existing_test_collection"
        
        with patch("symbiont_cli.main.QdrantClient") as mock_client_class, \
             patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_embeddings, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm:
            
            # Configure mocks
            mock_client = mock_client_class.return_value
            mock_client.collection_exists.return_value = True  # Collection exists
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                cli = SymbiontCLI(
                    docs_directory=test_docs_dir,
                    collection_name=collection_name,
                    k_value=5,
                    llm_response="yes",
                    output_directory="search_results",
                    q_list=None
                )
                
                # Verify collection was NOT created (since it exists)
                mock_client.create_collection.assert_not_called()
                
            finally:
                os.chdir(original_cwd)

    def test_search_and_qa_pipeline(self, temp_dir, mock_openai_key, sample_config, test_docs_dir):
        """Test the complete search and QA pipeline"""
        collection_name = "test_qa_collection"
        test_query = "What is the main topic?"
        
        with patch("symbiont_cli.main.QdrantClient") as mock_client_class, \
             patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_embeddings, \
             patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm, \
             patch("symbiont_cli.main.init_reranker") as mock_reranker, \
             patch("symbiont_cli.main.ContextualCompressionRetriever") as mock_compression:
            
            # Configure mocks
            mock_client = mock_client_class.return_value
            mock_client.collection_exists.return_value = False
            
            mock_loader_instance = mock_loader.return_value
            mock_loader_instance.load.return_value = [
                MagicMock(
                    page_content="This document discusses machine learning concepts",
                    metadata={"source": "ml.pdf", "title": "ML Guide", "page": 1}
                )
            ]
            
            # Mock search results
            mock_doc = MagicMock()
            mock_doc.page_content = "Machine learning is a subset of AI"
            mock_doc.metadata = {
                "source": "ml.pdf", 
                "title": "ML Guide", 
                "page": 1,
                "relevance_score": 0.95
            }
            
            mock_compression_instance = mock_compression.return_value
            mock_compression_instance.invoke.return_value = [mock_doc]
            
            # Mock LLM response
            mock_llm_instance = mock_llm.return_value
            mock_qa_chain = MagicMock()
            mock_qa_chain.run.return_value = "Machine learning is a field of AI that uses algorithms to learn patterns."
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            # Create logs directory
            logs_dir = os.path.join(temp_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                cli = SymbiontCLI(
                    docs_directory=test_docs_dir,
                    collection_name=collection_name,
                    k_value=5,
                    llm_response="yes",
                    output_directory="search_results",
                    q_list=None
                )
                
                # Mock the QA chain
                cli.qa_stuff = mock_qa_chain
                
                # Perform search and QA
                cli.perform_search_and_qa(test_query)
                
                # Verify operations were called
                mock_compression_instance.invoke.assert_called_once_with(test_query)
                mock_qa_chain.run.assert_called_once()
                
                # Verify log file was created
                log_file = os.path.join(logs_dir, f"{collection_name}.txt")
                assert os.path.exists(log_file)
                
                # Verify log content
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    assert test_query in log_content
                    assert "Machine learning is a field of AI" in log_content
                
            finally:
                os.chdir(original_cwd)

    def test_embedding_models_configuration(self, temp_dir, mock_openai_key):
        """Test different embedding model configurations"""
        test_docs_dir = os.path.join(temp_dir, "docs")
        os.makedirs(test_docs_dir)
        
        # Test configurations for different embedding models
        embedding_configs = [
            {"model": "huggingface", "model_id": "sentence-transformers/all-MiniLM-L6-v2"},
            {"model": "openai"},
            {"model": "voyage", "model_id": "voyage-3-lite"},
            {"model": "jina", "model_id": "jina-embeddings-v2-base-en"}
        ]
        
        for embedding_config in embedding_configs:
            config = {
                "general": {"llm_name": "gpt-4o-mini"},
                "embeddings": embedding_config,
                "reranker": {"reranker": "huggingface"},
                "qdrant": {"host": "localhost", "port": 6333}
            }
            
            with patch("symbiont_cli.main.QdrantClient") as mock_client_class, \
                 patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_hf, \
                 patch("symbiont_cli.main.OpenAIEmbeddings") as mock_openai, \
                 patch("symbiont_cli.main.VoyageAIEmbeddings") as mock_voyage, \
                 patch("symbiont_cli.main.JinaEmbeddings") as mock_jina, \
                 patch("symbiont_cli.main.ChatOpenAI") as mock_llm, \
                 patch.dict(os.environ, {"EMBEDDINGS_MODEL_API_KEY": "test_key"}):
                
                mock_client = mock_client_class.return_value
                mock_client.collection_exists.return_value = True
                
                # Create config file
                import toml
                config_path = os.path.join(temp_dir, "config.toml")
                with open(config_path, "w") as f:
                    toml.dump(config, f)
                
                # Change to temp directory
                original_cwd = os.getcwd()
                try:
                    os.chdir(temp_dir)
                    
                    cli = SymbiontCLI(
                        docs_directory=test_docs_dir,
                        collection_name="test_embeddings",
                        k_value=5,
                        llm_response="no",
                        output_directory="search_results",
                        q_list=None
                    )
                    
                    # Verify correct embedding model was initialized
                    if embedding_config["model"] == "huggingface":
                        mock_hf.assert_called_once()
                    elif embedding_config["model"] == "openai":
                        mock_openai.assert_called_once()
                    elif embedding_config["model"] == "voyage":
                        mock_voyage.assert_called_once()
                    elif embedding_config["model"] == "jina":
                        mock_jina.assert_called_once()
                    
                finally:
                    os.chdir(original_cwd)

    def test_reranker_configuration(self, temp_dir, mock_openai_key, test_docs_dir):
        """Test different reranker configurations"""
        reranker_configs = [
            {"reranker": "huggingface"},
            {"reranker": "cohere"}
        ]
        
        for reranker_config in reranker_configs:
            config = {
                "general": {"llm_name": "gpt-4o-mini"},
                "embeddings": {"model": "huggingface"},
                "reranker": reranker_config,
                "qdrant": {"host": "localhost", "port": 6333}
            }
            
            with patch("symbiont_cli.main.QdrantClient") as mock_client_class, \
                 patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_embeddings, \
                 patch("symbiont_cli.main.ChatOpenAI") as mock_llm, \
                 patch("symbiont_cli.main.HuggingFaceCrossEncoder") as mock_hf_reranker, \
                 patch("symbiont_cli.main.CohereRerank") as mock_cohere_reranker:
                
                mock_client = mock_client_class.return_value
                mock_client.collection_exists.return_value = True
                
                # Create config file
                import toml
                config_path = os.path.join(temp_dir, "config.toml")
                with open(config_path, "w") as f:
                    toml.dump(config, f)
                
                # Change to temp directory
                original_cwd = os.getcwd()
                try:
                    os.chdir(temp_dir)
                    
                    cli = SymbiontCLI(
                        docs_directory=test_docs_dir,
                        collection_name="test_reranker",
                        k_value=5,
                        llm_response="no",
                        output_directory="search_results",
                        q_list=None
                    )
                    
                    # Verify correct reranker was initialized
                    if reranker_config["reranker"] == "huggingface":
                        mock_hf_reranker.assert_called_once()
                    elif reranker_config["reranker"] == "cohere":
                        mock_cohere_reranker.assert_called_once()
                
                finally:
                    os.chdir(original_cwd)

    def test_batch_question_processing(self, temp_dir, mock_openai_key, sample_config, test_docs_dir):
        """Test processing multiple questions from file"""
        collection_name = "test_batch_collection"
        
        # Create test questions file
        questions_file = os.path.join(temp_dir, "test_questions.txt")
        test_questions = [
            "What is machine learning?",
            "How does AI work?",
            "What are neural networks?"
        ]
        with open(questions_file, "w") as f:
            for question in test_questions:
                f.write(f"{question}\n")
        
        with patch("symbiont_cli.main.QdrantClient") as mock_client_class, \
             patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_embeddings, \
             patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm, \
             patch("symbiont_cli.main.init_reranker") as mock_reranker, \
             patch("symbiont_cli.main.ContextualCompressionRetriever") as mock_compression, \
             patch("time.sleep"):  # Speed up the sleep between questions
            
            # Configure mocks
            mock_client = mock_client_class.return_value
            mock_client.collection_exists.return_value = False
            
            mock_loader_instance = mock_loader.return_value
            mock_loader_instance.load.return_value = [
                MagicMock(page_content="AI content", metadata={"source": "ai.pdf", "title": "AI", "page": 1})
            ]
            
            mock_compression_instance = mock_compression.return_value
            mock_compression_instance.invoke.return_value = [
                MagicMock(
                    page_content="AI answer content",
                    metadata={"source": "ai.pdf", "title": "AI", "page": 1, "relevance_score": 0.9}
                )
            ]
            
            mock_qa_chain = MagicMock()
            mock_qa_chain.run.return_value = "This is an AI response"
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            # Create logs directory
            logs_dir = os.path.join(temp_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                cli = SymbiontCLI(
                    docs_directory=test_docs_dir,
                    collection_name=collection_name,
                    k_value=5,
                    llm_response="yes",
                    output_directory="search_results",
                    q_list=questions_file
                )
                
                # Mock the QA chain
                cli.qa_stuff = mock_qa_chain
                
                # Process batch questions
                cli.generate_qa_list(questions_file)
                
                # Verify all questions were processed
                assert mock_qa_chain.run.call_count == len(test_questions)
                
                # Verify log file contains all questions
                log_file = os.path.join(logs_dir, f"{collection_name}.txt")
                assert os.path.exists(log_file)
                
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    for question in test_questions:
                        assert question.strip() in log_content
                
            finally:
                os.chdir(original_cwd)

    def test_error_handling_invalid_directory(self, temp_dir, mock_openai_key, sample_config):
        """Test error handling for invalid documents directory"""
        invalid_dir = "/nonexistent/directory"
        
        # Create config file
        import toml
        config_path = os.path.join(temp_dir, "config.toml")
        with open(config_path, "w") as f:
            toml.dump(sample_config, f)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            with pytest.raises(ValueError, match="does not exist"):
                SymbiontCLI(
                    docs_directory=invalid_dir,
                    collection_name="test_collection",
                    k_value=5,
                    llm_response="yes",
                    output_directory="search_results",
                    q_list=None
                )
        
        finally:
            os.chdir(original_cwd)

    def test_error_handling_missing_api_key(self, temp_dir, sample_config, test_docs_dir):
        """Test error handling for missing OpenAI API key"""
        # Create config file
        import toml
        config_path = os.path.join(temp_dir, "config.toml")
        with open(config_path, "w") as f:
            toml.dump(sample_config, f)
        
        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    SymbiontCLI(
                        docs_directory=test_docs_dir,
                        collection_name="test_collection",
                        k_value=5,
                        llm_response="yes",
                        output_directory="search_results",
                        q_list=None
                    )
            
            finally:
                os.chdir(original_cwd)

    def test_llm_response_disabled(self, temp_dir, mock_openai_key, sample_config, test_docs_dir):
        """Test operation with LLM response disabled"""
        collection_name = "test_no_llm_collection"
        test_query = "What is the content?"
        
        with patch("symbiont_cli.main.QdrantClient") as mock_client_class, \
             patch("symbiont_cli.main.HuggingFaceEmbeddings") as mock_embeddings, \
             patch("symbiont_cli.main.DirectoryLoader") as mock_loader, \
             patch("symbiont_cli.main.ChatOpenAI") as mock_llm, \
             patch("symbiont_cli.main.init_reranker") as mock_reranker, \
             patch("symbiont_cli.main.ContextualCompressionRetriever") as mock_compression:
            
            # Configure mocks
            mock_client = mock_client_class.return_value
            mock_client.collection_exists.return_value = False
            
            mock_loader_instance = mock_loader.return_value
            mock_loader_instance.load.return_value = [
                MagicMock(page_content="Content", metadata={"source": "test.pdf", "title": "Test", "page": 1})
            ]
            
            mock_compression_instance = mock_compression.return_value
            mock_compression_instance.invoke.return_value = [
                MagicMock(
                    page_content="Test content",
                    metadata={"source": "test.pdf", "title": "Test", "page": 1, "relevance_score": 0.8}
                )
            ]
            
            # Create config file
            import toml
            config_path = os.path.join(temp_dir, "config.toml")
            with open(config_path, "w") as f:
                toml.dump(sample_config, f)
            
            # Create logs directory
            logs_dir = os.path.join(temp_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                cli = SymbiontCLI(
                    docs_directory=test_docs_dir,
                    collection_name=collection_name,
                    k_value=5,
                    llm_response="no",  # Disable LLM response
                    output_directory="search_results",
                    q_list=None
                )
                
                # Perform search without LLM
                cli.perform_search_and_qa(test_query)
                
                # Verify search was performed but LLM was not called
                mock_compression_instance.invoke.assert_called_once_with(test_query)
                
                # Verify log file was created with search results only
                log_file = os.path.join(logs_dir, f"{collection_name}.txt")
                assert os.path.exists(log_file)
                
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    assert test_query in log_content
                    assert "LLM response is turned off" in log_content
                
            finally:
                os.chdir(original_cwd)