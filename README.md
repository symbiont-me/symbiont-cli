# 🌟 Symbiont CLI 🌟

Symbiont CLI is a powerful command-line interface tool designed for processing documents 📄 and performing question-answering tasks 🧠 using various language models and vector stores.
It is an extension of the Symbiont app: https://github.com/symbiont-me/symbiont

## 🚀 Features

- 📥 Load and process PDF documents from a specified directory
- 🛠️ Create and manage vector embeddings using OpenAI or HuggingFace models
- 🗄️ Store embeddings in a Qdrant vector database
- 🔍 Perform similarity searches on stored documents
- 🤖 Generate answers to questions using a language model (optional)

## 🛠️ Prerequisites

- Poetry (https://python-poetry.org/docs/)
- Qdrant server running locally on port 6333 (https://qdrant.tech/documentation/quickstart/)

## 📥 Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:symbiont-me/symbiont-cli.git
   cd symbiont-cli
   ```

2. Install the required dependencies:
   ```bash
   poetry install
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 📖 Usage

Run the script with the following command:

```bash
poetry run python symbiont_cli.py --docs_directory /path/to/documents --collection_name your_collection_name [options]
```

### Required Arguments:

- `--docs_directory`: Path to the directory containing PDF documents to process
- `--collection_name`: Name of the Qdrant collection to use

### Optional Arguments:

- `--k_value`: Number of documents to retrieve (default: 3)
- `--llm_response`: Whether to use the language model for responses (default: "yes")
- `--output_directory`: Directory to save search results (default: "search_results")

### 📄 Example:

```bash
python symbiont_cli.py --docs_directory ./documents --collection_name my_collection --k_value 5 --llm_response yes
```

Once the script is running, you can enter queries at the prompt. Type 'exit' to stop the program.

## ⚠️ Notes

- Make sure you have a Qdrant server running locally on port 6333 before starting the script.
- The script will create a new collection in Qdrant if it doesn't exist and load documents from the specified directory.
- If the collection already exists, it will simply search the collection.
- If the OpenAI API key is not set, the script will fall back to using HuggingFace embeddings.

## 🐛 Troubleshooting

If you encounter any issues, check the following:

1. Ensure all dependencies are installed correctly.
2. Verify that your OpenAI API key is set in the `.env` file.
3. Make sure the Qdrant server is running and accessible.

For any other problems, please refer to the error messages in the console output.

## 💖 Contributions

Contributions are welcome! If you’d like to contribute to SymbiontCLI, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Open a pull request with a detailed description of your changes.

Thank you for your interest in improving SymbiontCLI! 🙌

