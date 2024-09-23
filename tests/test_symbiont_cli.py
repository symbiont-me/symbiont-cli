import pytest
from unittest.mock import patch, MagicMock
from symbiont_cli.qdrant import SymbiontCLI


@pytest.fixture
def mock_dependencies():
    with patch("symbiont_cli.qdrant.QdrantClient") as MockQdrantClient, patch(
        "symbiont_cli.qdrant.DirectoryLoader"
    ) as MockDirectoryLoader, patch(
        "symbiont_cli.qdrant.OpenAIEmbeddings"
    ) as MockOpenAIEmbeddings, patch(
        "symbiont_cli.qdrant.ChatOpenAI"
    ) as MockChatOpenAI:
        mock_client = MockQdrantClient.return_value
        mock_loader = MockDirectoryLoader.return_value
        mock_loader.load.return_value = [
            MagicMock(page_content="Test content", metadata={"title": "Test"})
        ]
        mock_embeddings = MockOpenAIEmbeddings.return_value
        mock_llm = MockChatOpenAI.return_value

        yield mock_client, mock_loader, mock_embeddings, mock_llm


def test_end_to_end_workflow(mock_dependencies):
    mock_client, mock_loader, mock_embeddings, mock_llm = mock_dependencies

    cli = SymbiontCLI()

    cli.args = MagicMock(
        loader_directory="test_dir",
        collection_name="test_collection",
        k_value=3,
        llm_response="yes",
    )

    cli.perform_search_and_qa("test query")

    mock_loader.load.assert_called_once()
    mock_client.create_collection.assert_called_once()
    mock_llm.run.assert_called_once()
