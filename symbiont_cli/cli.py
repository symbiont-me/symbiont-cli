import typer
from symbiont_cli.main import SymbiontCLI
import os

app = typer.Typer()

@app.command()
def init():
    """
    Initialize the configuration file.
    """
    config_content = """[general]
llm_name = "gpt-4o-mini"

[embeddings]
model = "huggingface"
# model = "openai"
# model = "voyage"
# model = "jina"

[reranker]
# reranker = "cohere"
reranker = "huggingface"

[qdrant]
host = "localhost"
port = 6333
"""
    with open("config.toml", "w") as f:
        f.write(config_content)
    print("Configuration file 'config.toml' created.")

@app.command()
def chat(
    docs_directory: str = typer.Option(..., "--docs_directory", "-d", help="Directory to load documents from"),
    collection_name: str = typer.Option(..., "--collection_name", "-c", help="Name of the Qdrant collection"),
    k_value: int = typer.Option(3, "--k_value", "-k", help="Number of documents to retrieve"),
    llm_response: str = typer.Option("yes", "--llm_response", "-l", help="If you want to use the LLM for responses or plain similarity search"),
    output_directory: str = typer.Option("search_results", "--output_directory", "-o", help="Directory to save search results"),
    q_list: str = typer.Option(None, "--q_list", "-q", help="File containing list of questions to ask"),
):
    """
    Chat with your documents.
    """
    cli = SymbiontCLI(
        docs_directory=docs_directory,
        collection_name=collection_name,
        k_value=k_value,
        llm_response=llm_response,
        output_directory=output_directory,
        q_list=q_list,
    )
    cli.run()

@app.command()
def list_collections():
    """
    List all collections in the Qdrant database.
    """
    # This will be implemented later
    print("Listing collections...")

@app.command()
def remove_collection(collection_name: str):
    """
    Remove a collection from the Qdrant database.
    """
    # This will be implemented later
    print(f"Removing collection: {collection_name}")

if __name__ == "__main__":
    app()
