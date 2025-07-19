"""
End-to-end tests for CLI commands
"""
import pytest
import os
import json
import toml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from symbiont_cli.cli import app


class TestCLICommands:
    """Test all CLI commands end-to-end"""

    def test_init_command(self, runner, isolated_config):
        """Test basic init command creates config file"""
        result = runner.invoke(app, ["init"])
        
        assert result.exit_code == 0
        assert "Configuration file 'config.toml' created" in result.stdout
        assert Path("config.toml").exists()
        
        # Verify config content
        with open("config.toml") as f:
            config = toml.load(f)
        
        assert config["general"]["llm_name"] == "gpt-4o-mini"
        assert config["embeddings"]["model"] == "huggingface"
        assert config["qdrant"]["host"] == "localhost"

    def test_setup_command_with_defaults(self, runner, isolated_config, monkeypatch):
        """Test setup command with default selections"""
        # Mock user inputs for interactive setup
        inputs = ["1\n", "1\n", "1\n", "\n", "\n", "n\n"]  # All defaults + no API key setup
        
        result = runner.invoke(app, ["setup"], input="".join(inputs))
        
        # Should succeed even if Qdrant connection fails
        assert result.exit_code == 0
        assert "Setup completed" in result.stdout
        assert Path("config.toml").exists()

    def test_setup_command_overwrite_existing(self, runner, isolated_config, test_config_file):
        """Test setup command with existing config file"""
        # Copy test config to current directory
        with open("config.toml", "w") as f:
            with open(test_config_file) as src:
                f.write(src.read())
        
        inputs = ["y\n", "1\n", "1\n", "1\n", "\n", "\n", "n\n"]  # Overwrite + defaults
        result = runner.invoke(app, ["setup"], input="".join(inputs))
        
        assert result.exit_code == 0

    def test_status_command_no_config(self, runner, isolated_config):
        """Test status command when no config exists"""
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "Config file missing" in result.stdout

    def test_status_command_with_config(self, runner, isolated_config, sample_config):
        """Test status command with config file"""
        # Create config file
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "Config file found" in result.stdout
        assert "OpenAI API key found" in result.stdout

    @pytest.mark.parametrize("input_format,file_content", [
        ("text", "What is this?\nHow does it work?\n"),
        ("json", '{"questions": ["What is this?", "How does it work?"]}'),
        ("csv", "What is this?\nHow does it work?\n")
    ])
    def test_ask_command_file_input(self, runner, isolated_config, mock_expensive_operations, 
                                   sample_config, input_format, file_content, temp_dir):
        """Test ask command with different file input formats"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Create test file
        test_file = os.path.join(temp_dir, f"questions.{input_format}")
        with open(test_file, "w") as f:
            f.write(file_content)
        
        # Create test docs directory
        docs_dir = os.path.join(temp_dir, "docs")
        os.makedirs(docs_dir)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}), \
             patch("symbiont_cli.main.QdrantClient") as mock_client:
            
            mock_client.return_value.collection_exists.return_value = False
            mock_client.return_value.create_collection.return_value = None
            
            result = runner.invoke(app, [
                "ask", 
                "--file", test_file,
                "--format", input_format,
                "--docs", docs_dir,
                "--batch"
            ])
        
        assert result.exit_code == 0

    def test_ask_command_single_question(self, runner, isolated_config, mock_expensive_operations,
                                        sample_config, temp_dir):
        """Test ask command with single question"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Create test docs directory  
        docs_dir = os.path.join(temp_dir, "docs")
        os.makedirs(docs_dir)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}), \
             patch("symbiont_cli.main.QdrantClient") as mock_client:
            
            mock_client.return_value.collection_exists.return_value = False
            mock_client.return_value.create_collection.return_value = None
            
            result = runner.invoke(app, [
                "ask",
                "What is machine learning?",
                "--docs", docs_dir
            ])
        
        assert result.exit_code == 0

    def test_ask_command_quick_mode_no_session(self, runner, isolated_config, sample_config):
        """Test ask command quick mode with no previous session"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        result = runner.invoke(app, ["ask", "--quick", "test question"])
        
        assert result.exit_code == 1
        assert "No previous session found" in result.stdout

    def test_chat_command(self, runner, isolated_config, mock_expensive_operations,
                         sample_config, temp_dir):
        """Test chat command basic functionality"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Create test docs directory
        docs_dir = os.path.join(temp_dir, "docs") 
        os.makedirs(docs_dir)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}), \
             patch("symbiont_cli.main.QdrantClient") as mock_client, \
             patch("symbiont_cli.main.SymbiontCLI.query_loop") as mock_loop:
            
            mock_client.return_value.collection_exists.return_value = False
            mock_client.return_value.create_collection.return_value = None
            mock_loop.return_value = None
            
            result = runner.invoke(app, ["chat", docs_dir, "test_collection"])
        
        assert result.exit_code == 0

    def test_chat_command_auto_collection_name(self, runner, isolated_config, 
                                              mock_expensive_operations, sample_config, temp_dir):
        """Test chat command with auto-inferred collection name"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Create test docs directory with specific name
        docs_dir = os.path.join(temp_dir, "research-papers")
        os.makedirs(docs_dir)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}), \
             patch("symbiont_cli.main.QdrantClient") as mock_client, \
             patch("symbiont_cli.main.SymbiontCLI.query_loop") as mock_loop:
            
            mock_client.return_value.collection_exists.return_value = False  
            mock_client.return_value.create_collection.return_value = None
            mock_loop.return_value = None
            
            result = runner.invoke(app, ["chat", docs_dir])
        
        assert result.exit_code == 0
        assert "research-papers" in result.stdout

    def test_collections_command_empty(self, runner, isolated_config, sample_config):
        """Test collections command with no collections"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        with patch("symbiont_cli.cli.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value.collections = []
            
            result = runner.invoke(app, ["collections"])
        
        assert result.exit_code == 0
        assert "No collections found" in result.stdout

    def test_collections_command_with_data(self, runner, isolated_config, sample_config):
        """Test collections command with existing collections"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Mock collection data
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        
        with patch("symbiont_cli.cli.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value.collections = [mock_collection]
            
            result = runner.invoke(app, ["collections"])
        
        assert result.exit_code == 0
        assert "test_collection" in result.stdout

    def test_collections_command_detailed(self, runner, isolated_config, sample_config):
        """Test collections command with detailed view"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Mock collection data
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 100
        
        with patch("symbiont_cli.cli.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value.collections = [mock_collection]
            mock_client.return_value.get_collection.return_value = mock_info
            
            result = runner.invoke(app, ["collections", "--detailed"])
        
        assert result.exit_code == 0
        assert "test_collection" in result.stdout

    def test_collection_info_command(self, runner, isolated_config, sample_config):
        """Test collection-info command"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Mock collection data
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 100
        mock_info.config.params.vectors.size = 384
        mock_info.config.params.vectors.distance = "Cosine"
        
        with patch("symbiont_cli.cli.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value.collections = [mock_collection]
            mock_client.return_value.get_collection.return_value = mock_info
            mock_client.return_value.scroll.return_value = ([], None)
            
            result = runner.invoke(app, ["collection-info", "test_collection"])
        
        assert result.exit_code == 0
        assert "test_collection" in result.stdout

    def test_collection_info_not_found(self, runner, isolated_config, sample_config):
        """Test collection-info command for non-existent collection"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        with patch("symbiont_cli.cli.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value.collections = []
            
            result = runner.invoke(app, ["collection-info", "nonexistent"])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_remove_collection_command(self, runner, isolated_config, sample_config):
        """Test remove-collection command"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Mock collection data
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        
        with patch("symbiont_cli.cli.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value.collections = [mock_collection]
            mock_client.return_value.delete_collection.return_value = None
            
            result = runner.invoke(app, ["remove-collection", "test_collection"], input="y\n")
        
        assert result.exit_code == 0
        assert "deleted successfully" in result.stdout

    def test_remove_collection_cancel(self, runner, isolated_config, sample_config):
        """Test remove-collection command with cancellation"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Mock collection data
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        
        with patch("symbiont_cli.cli.QdrantClient") as mock_client:
            mock_client.return_value.get_collections.return_value.collections = [mock_collection]
            
            result = runner.invoke(app, ["remove-collection", "test_collection"], input="n\n")
        
        assert result.exit_code == 0
        assert "cancelled" in result.stdout

    def test_profiles_command_empty(self, runner, isolated_config):
        """Test profiles command with no profiles"""
        result = runner.invoke(app, ["profiles"])
        
        assert result.exit_code == 0
        assert "No profiles found" in result.stdout

    def test_profiles_create_command(self, runner, isolated_config, sample_config):
        """Test profiles create command"""
        # Setup base config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        result = runner.invoke(app, ["profiles", "--create", "test_profile"])
        
        assert result.exit_code == 0
        assert "Profile 'test_profile' created" in result.stdout
        assert Path("config_test_profile.toml").exists()

    def test_profiles_use_command(self, runner, isolated_config, sample_config):
        """Test profiles use command"""
        # Create a profile file
        with open("config_test_profile.toml", "w") as f:
            toml.dump(sample_config, f)
        
        result = runner.invoke(app, ["profiles", "--use", "test_profile"])
        
        assert result.exit_code == 0
        assert "Switched to profile 'test_profile'" in result.stdout

    def test_invalid_docs_directory(self, runner, isolated_config, sample_config):
        """Test commands with invalid docs directory"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            result = runner.invoke(app, [
                "ask", 
                "test question",
                "--docs", "/nonexistent/directory"
            ])
        
        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_missing_api_key(self, runner, isolated_config, sample_config, temp_dir):
        """Test commands without OpenAI API key"""
        # Setup config
        with open("config.toml", "w") as f:
            toml.dump(sample_config, f)
        
        # Create test docs directory
        docs_dir = os.path.join(temp_dir, "docs")
        os.makedirs(docs_dir)
        
        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(app, [
                "ask",
                "test question", 
                "--docs", docs_dir
            ])
        
        # Should fail due to missing API key
        assert result.exit_code == 1