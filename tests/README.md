# SymbiontCLI E2E Test Suite

This directory contains comprehensive end-to-end tests for SymbiontCLI that avoid excessive mocking and test real functionality.

## Test Categories

### 1. Unit Tests (test_e2e_cli.py)
- Tests CLI commands with minimal mocking
- Focuses on command parsing, validation, and basic flows
- Uses mocks only for expensive operations (LLM calls, embeddings)

### 2. Document Processing Tests (test_e2e_document_processing.py)
- Tests the complete document processing pipeline
- Vector store creation and population
- Search and QA functionality
- Different embedding model configurations
- Batch question processing

### 3. Integration Tests (test_e2e_integration.py)
- **@pytest.mark.integration**: Tests with real Qdrant database
- **@pytest.mark.docker**: Tests with Qdrant running in Docker
- **@pytest.mark.openai**: Tests with real OpenAI API

## Running Tests

### Install Test Dependencies
```bash
uv sync --extra test
```

### Run All Tests (Fast)
```bash
uv run pytest tests/ -v
```

### Run Only Unit Tests
```bash
uv run pytest tests/test_e2e_cli.py tests/test_e2e_document_processing.py -v
```

### Run Integration Tests (Requires Qdrant)
```bash
# Start Qdrant first
docker run -p 6333:6333 qdrant/qdrant

# Run integration tests
uv run pytest tests/ -m integration -v
```

### Run Docker Tests (Requires Docker)
```bash
uv run pytest tests/ -m docker -v
```

### Run OpenAI Tests (Requires Real API Key)
```bash
export OPENAI_API_KEY="your-real-api-key"
uv run pytest tests/ -m openai -v
```

### Run All Tests Including External Dependencies
```bash
# Requires: Qdrant running, Docker available, OpenAI API key set
uv run pytest tests/ -v
```

## Test Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI integration tests
- `EMBEDDINGS_MODEL_API_KEY`: For Voyage/Jina embedding tests

### External Services Required

#### For Integration Tests
- **Qdrant**: Running on localhost:6333
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

#### For Docker Tests
- **Docker**: Available and running
  - Tests will automatically start/stop Qdrant containers

#### For OpenAI Tests
- **Real OpenAI API Key**: Set in environment
- **Internet Connection**: For API calls

## Test Structure

### Fixtures (conftest.py)
- `runner`: CLI test runner
- `temp_dir`: Temporary directory for test files
- `sample_config`: Test configuration
- `test_docs_dir`: Directory with sample documents
- `qdrant_client`: Real Qdrant client (integration tests)
- `cleanup_collections`: Automatic cleanup of test collections
- `mock_expensive_operations`: Mock costly operations for speed

### Test Patterns

#### Isolated Configuration
Tests run in temporary directories with isolated config files to avoid affecting the main development environment.

#### Minimal Mocking
- Only mock expensive/external operations when necessary
- Use real implementations for core logic
- Mock: LLM calls, heavy embedding operations, file I/O where appropriate
- Real: CLI parsing, configuration loading, vector operations, Qdrant interactions

#### External Service Testing
- Integration tests verify real database operations
- Docker tests ensure containerized deployment works
- OpenAI tests validate actual API integration

## Test Coverage

### CLI Commands Tested
- ✅ `init` - Configuration file creation
- ✅ `setup` - Interactive setup wizard
- ✅ `ask` - Question answering (single, file, batch modes)
- ✅ `chat` - Interactive chat mode
- ✅ `collections` - Collection listing and management
- ✅ `collection-info` - Collection details
- ✅ `remove-collection` - Collection deletion
- ✅ `status` - System health check
- ✅ `profiles` - Configuration profile management

### Document Processing Pipeline
- ✅ Vector store creation and population
- ✅ Document loading and embedding
- ✅ Similarity search with reranking
- ✅ LLM response generation
- ✅ Batch question processing
- ✅ Different embedding models (HuggingFace, OpenAI, Voyage, Jina)
- ✅ Different reranking models (HuggingFace, Cohere)

### Error Handling
- ✅ Invalid directories
- ✅ Missing API keys
- ✅ Non-existent collections
- ✅ Qdrant connection failures
- ✅ File parsing errors

### Configuration Management
- ✅ Config file creation and loading
- ✅ Profile switching
- ✅ Last session persistence
- ✅ Environment variable handling

## Test Execution Levels

### Level 1: Fast Unit Tests (No External Dependencies)
```bash
uv run pytest tests/test_e2e_cli.py tests/test_e2e_document_processing.py
```
- ~30 seconds execution time
- No external services required
- Covers all CLI commands and document processing logic

### Level 2: Integration Tests (Requires Qdrant)
```bash
uv run pytest tests/ -m integration
```
- ~2-5 minutes execution time
- Requires Qdrant running on localhost:6333
- Tests real vector database operations

### Level 3: Full E2E Tests (All External Dependencies)
```bash
uv run pytest tests/
```
- ~5-10 minutes execution time
- Requires Qdrant, Docker, and OpenAI API key
- Complete end-to-end validation

## CI/CD Integration

### GitHub Actions Example
```yaml
name: E2E Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run unit tests
      run: |
        uv sync --extra test
        uv run pytest tests/test_e2e_cli.py tests/test_e2e_document_processing.py

  integration-tests:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
        - 6333:6333
    steps:
    - uses: actions/checkout@v3
    - name: Run integration tests
      run: |
        uv sync --extra test
        uv run pytest tests/ -m integration
```

## Best Practices

### When Adding New Tests

1. **Start with unit tests**: Test new functionality with minimal mocking
2. **Add integration tests**: For features that interact with Qdrant
3. **Consider Docker tests**: For deployment-related functionality
4. **Add OpenAI tests sparingly**: Only for critical LLM integration features

### Test Organization

- Keep tests focused and independent
- Use descriptive test names
- Group related tests in classes
- Use appropriate markers for test categorization

### Mocking Guidelines

- Mock only what's necessary for test speed/reliability
- Use real implementations when possible
- Mock external API calls and expensive operations
- Don't mock the code under test