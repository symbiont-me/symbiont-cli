# 🧠 SymbiontCLI

**Intelligent document processing and Q&A tool with advanced RAG capabilities**

SymbiontCLI is a powerful command-line interface for processing documents and performing question-answering tasks using state-of-the-art language models, embeddings, and vector search.

## ✨ Features

- 🚀 **Interactive Setup Wizard** - Guided configuration for first-time users
- 📄 **Multi-format Document Processing** - PDF support with advanced chunking
- 🎯 **Flexible Question Input** - Single questions, files (TXT, JSON, CSV, YAML), batch processing
- 🔍 **Advanced RAG Pipeline** - Similarity search + reranking for better results
- 🗄️ **Vector Database Management** - Qdrant integration with collection management
- 🤖 **Configurable LLMs** - OpenAI models with customizable prompts
- 📊 **Status Monitoring** - Health checks and system diagnostics
- 👤 **Profile Management** - Multiple configuration profiles
- ⚡ **Quick Mode** - Resume from last session instantly
- 📄 **Document Export** - Save results to Word (.docx) and PDF formats

## 🛠️ Prerequisites

- **uv** package manager (replaces Poetry)
- **Qdrant server** running on localhost:6333
- **OpenAI API key** (for LLM and embeddings)

## 📥 Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the setup wizard:**
   ```bash
   uv run symbiont-cli setup
   ```

3. **Start Qdrant server:**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Check system status:**
   ```bash
   uv run symbiont-cli status
   ```

5. **Ask your first question:**
   ```bash
   uv run symbiont-cli ask -d docs/folder "What is the main topic?"
   ```

## 🎯 Core Commands

### **Question & Answer**
```bash
# Single question
uv run symbiont-cli ask "What is machine learning?"

# Use specific collection
uv run symbiont-cli ask -c research "methodology"

# Auto-create collection from folder
uv run symbiont-cli ask -d docs/papers "findings"

# Quick mode (uses last collection)
uv run symbiont-cli ask --quick "follow-up question"

# From file inputs
uv run symbiont-cli ask --file questions.txt
uv run symbiont-cli ask --file data.json --format json
uv run symbiont-cli ask --file questions.csv --format csv

# Save results to documents
uv run symbiont-cli ask "research question" --output report.docx
uv run symbiont-cli ask "analysis query" --output results.pdf
```

### **Interactive Chat**
```bash
# Extended conversation mode
uv run symbiont-cli chat docs/research economics

# Use last collection
uv run symbiont-cli chat --quick

# Batch processing
uv run symbiont-cli chat docs/papers ai --questions questions.txt

# Save chat session to document
uv run symbiont-cli chat docs/research economics --output session.docx
uv run symbiont-cli chat docs/papers --output analysis.pdf
```

### **Collection Management**
```bash
# List all collections
uv run symbiont-cli collections

# Detailed view with metadata
uv run symbiont-cli collections --detailed

# Search collections
uv run symbiont-cli collections --search "economics"

# Collection information
uv run symbiont-cli collection-info my_collection

# Remove collection
uv run symbiont-cli remove-collection old_collection
```

### **Configuration**
```bash
# Interactive setup wizard
uv run symbiont-cli setup

# Create basic config
uv run symbiont-cli init

# Profile management
uv run symbiont-cli profiles
uv run symbiont-cli profiles --create work
uv run symbiont-cli profiles --use work

# System health check
uv run symbiont-cli status
```

## 📄 Document Export Feature

SymbiontCLI can export your query results and AI responses to professional documents for research and sharing purposes.

### **Supported Formats**
- **Word Documents (.docx)** - Professional reports with formatted sections
- **PDF Documents (.pdf)** - Clean, printable documents ideal for sharing

### **Usage Examples**

**Single Question with Export:**
```bash
# Export to Word document
uv run symbiont-cli ask "What are the key findings?" -o research-findings.docx

# Export to PDF
uv run symbiont-cli ask "Summarize the methodology" -o methodology-summary.pdf

# Use with specific collection
uv run symbiont-cli ask -c economics "What are the trends?" -o economic-trends.docx
```

**Chat Session Export:**
```bash
# Save entire chat session to Word
uv run symbiont-cli chat docs/research papers -o full-session.docx

# Export interactive session to PDF
uv run symbiont-cli chat docs/analysis --output analysis-report.pdf
```

**Batch Processing with Export:**
```bash
# Process multiple questions and save to document
uv run symbiont-cli ask --file questions.txt --batch -o batch-results.docx

# JSON input with PDF export
uv run symbiont-cli ask --file queries.json --format json -o results.pdf
```

### **Document Structure**

Generated documents include:

1. **Query Information**
   - Original question(s)
   - Collection used
   - Timestamp
   - Number of results found

2. **AI Generated Summary**
   - LLM response (if available)
   - Processing statistics (tokens used, cost)

3. **Configuration Details**
   - Models used (LLM, embeddings, reranker)
   - Settings and parameters

4. **Retrieved Documents**
   - Source document metadata
   - Relevance scores
   - Full content of relevant passages
   - Document page numbers and titles

### **Tips for Research Use**

**For Academic Research:**
```bash
# Create comprehensive research reports
uv run symbiont-cli ask "What does the literature say about X?" -o literature-review.docx

# Export methodology analysis
uv run symbiont-cli ask -c papers "Compare methodologies" -o methodology-comparison.pdf
```

**For Business Analysis:**
```bash
# Generate executive summaries
uv run symbiont-cli ask "Key business insights" -o executive-summary.docx

# Create detailed analysis reports
uv run symbiont-cli chat docs/reports analysis -o business-analysis.pdf
```

**For Documentation:**
```bash
# Document findings with sources
uv run symbiont-cli ask --file research-questions.txt -o documented-findings.docx

# Create reference materials
uv run symbiont-cli ask "Best practices overview" -o best-practices-guide.pdf
```

## ⚙️ Configuration

SymbiontCLI uses a `config.toml` file with the following structure:

```toml
[general]
llm_name = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"

[embeddings]
model = "huggingface"  # or "openai", "voyage", "jina"

[reranker]
reranker = "huggingface"  # or "cohere"

[qdrant]
host = "localhost"
port = 6333
```

### Environment Variables
```bash
export OPENAI_API_KEY="your_api_key_here"
export EMBEDDINGS_MODEL_API_KEY="for_voyage_jina"  # optional
export QA_BASE_PROMPT="custom_prompt"  # optional
```

## 🏗️ Architecture

**Document Processing Pipeline:**
1. PDF Loading → Text Chunking → Vector Embeddings → Qdrant Storage
2. Query Processing → Similarity Search → Reranking → LLM Generation

**Key Components:**
- **Document Loaders**: PyMuPDFLoader for robust PDF processing
- **Embeddings**: OpenAI, HuggingFace, Voyage, or Jina models
- **Vector Store**: Qdrant for similarity search
- **Reranking**: Cross-encoder models for improved retrieval
- **LLM Integration**: OpenAI models for response generation

## 📁 Project Structure

```
symbiont_cli/
├── cli.py              # Main CLI interface
├── main.py             # Core SymbiontCLI class
├── reset_vectors.py    # Utility scripts
├── list_collections.py
└── remove_collection.py

docs/                   # Document storage
qdrant_storage/         # Vector database
logs/                   # Query/response logs
config.toml             # Configuration
```

## 🔧 Development

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_symbiont_cli.py
```

### Development Scripts
```bash
# Reset vector database
uv run python symbiont_cli/reset_vectors.py

# List collections directly
uv run python symbiont_cli/list_collections.py
```

## 🐛 Troubleshooting

**Common Issues:**

1. **Qdrant Connection Failed**
   ```bash
   # Start Qdrant server
   docker run -p 6333:6333 qdrant/qdrant
   
   # Check status
   uv run symbiont-cli status
   ```

2. **OpenAI API Key Missing**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

3. **Collection Not Found**
   ```bash
   # List available collections
   uv run symbiont-cli collections
   
   # Create new collection
   uv run symbiont-cli ask -d docs/folder "test question"
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `uv run pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.