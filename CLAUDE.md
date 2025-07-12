# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
- **Install dependencies**: `uv sync` (uses uv package manager, migrated from Poetry)
- **Activate virtual environment**: `source .venv/bin/activate` or use `uv run` prefix
- **Initialize config**: `uv run symbiont-cli init` (creates config.toml)

### Running the Application
- **Interactive chat**: `uv run symbiont-cli chat -d /path/to/docs -c collection_name`
- **Batch processing**: `uv run symbiont-cli chat -d /path/to/docs -c collection_name -q questions.txt`
- **List collections**: `uv run symbiont-cli list_collections`
- **Remove collection**: `uv run symbiont-cli remove_collection collection_name`

### Testing
- **Run tests**: `uv run pytest`
- **Run specific test**: `uv run pytest tests/test_symbiont_cli.py`

### Development Scripts
- **Reset vectors**: `uv run python symbiont_cli/reset_vectors.py`
- **List collections**: `uv run python symbiont_cli/list_collections.py`
- **Remove collection**: `uv run python symbiont_cli/remove_collection.py`

## Architecture Overview

### Core Components

**Main Entry Points:**
- `symbiont_cli/cli.py` - Typer-based CLI interface with commands for init, chat, list_collections, remove_collection
- `symbiont_cli/main.py` - Core SymbiontCLI class containing all business logic

**Key Architecture:**
- **Document Processing Pipeline**: PDF documents → LangChain loaders (PyMuPDFLoader) → Text chunks → Vector embeddings → Qdrant storage
- **Embedding Models**: Configurable via config.toml (OpenAI, HuggingFace, Voyage, Jina)
- **Reranking**: Cross-encoder reranking using HuggingFace or Cohere for improved retrieval
- **LLM Integration**: OpenAI models (configurable) for question-answering
- **Vector Storage**: Qdrant vector database for similarity search

### Configuration System

**config.toml structure:**
```toml
[general]
llm_name = "gpt-4o-mini"

[embeddings]
model = "huggingface"  # or "openai", "voyage", "jina"

[reranker]
reranker = "huggingface"  # or "cohere"

[qdrant]
host = "localhost"
port = 6333
```

### Data Flow
1. **Document Loading**: PDF files from specified directory are loaded using DirectoryLoader with PyMuPDFLoader
2. **Vector Storage Setup**: Creates Qdrant collection if it doesn't exist, embeds documents with progress tracking
3. **Query Processing**: User queries → similarity search with k=50 → contextual compression reranking → top results
4. **Response Generation**: Retrieved context + user query → LLM prompt → generated response
5. **Logging**: All queries, responses, and document metadata logged to collection-specific files

### Dependencies and Environment

**Key Dependencies:**
- **LangChain ecosystem**: langchain, langchain-community, langchain-openai, langchain-qdrant
- **Vector DB**: qdrant-client for vector storage
- **Document processing**: pymupdf, pypdf, pdfplumber, unstructured
- **ML libraries**: sentence-transformers, transformers, torch
- **CLI**: typer for command-line interface
- **Configuration**: toml for config file parsing

**Environment Variables:**
- `OPENAI_API_KEY` - Required for OpenAI embeddings and LLM
- `EMBEDDINGS_MODEL_API_KEY` - For Voyage/Jina embeddings
- `QA_BASE_PROMPT` - Custom base prompt for QA (optional)

### File Structure
- `symbiont_cli/` - Main package directory
- `docs/` - Contains research documents organized by topic (genocide, guerilla-war, palestine)
- `qdrant_storage/` - Qdrant database files and collections
- `logs/` - Query and response logs
- `tests/` - Unit tests with mock dependencies
- `config.toml` - Configuration file

### External Dependencies
- **Qdrant server** must be running on localhost:6333 before using the application
- **Internet connection** required for OpenAI API calls and some embedding models