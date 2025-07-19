import typer
import os
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from qdrant_client import QdrantClient
import toml
import json

app = typer.Typer(
    help="🧠 SymbiontCLI - Intelligent document processing and Q&A tool",
    add_completion=False
)
console = Console()

@app.command()
def setup():
    """
    🚀 Interactive setup wizard for first-time configuration.
    """
    console.print("[bold blue]🧠 Welcome to SymbiontCLI Setup Wizard![/bold blue]")
    console.print("Let's configure your system step by step.\n")
    
    # Check if config already exists
    if Path("config.toml").exists():
        if not Confirm.ask("Config file already exists. Overwrite?"):
            console.print("[yellow]Setup cancelled.[/yellow]")
            raise typer.Exit(0)
    
    config = {}
    
    # General settings
    console.print("[bold green]📋 General Settings[/bold green]")
    llm_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
    console.print("Available LLM models:")
    for i, model in enumerate(llm_models, 1):
        console.print(f"  {i}. {model}")
    
    while True:
        try:
            choice = typer.prompt("Choose LLM model (1-4)", default="1")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(llm_models):
                config["llm_name"] = llm_models[choice_idx]
                break
        except (ValueError, IndexError):
            console.print("[red]Invalid choice. Please enter 1-4.[/red]")
    
    # Embeddings settings
    console.print("\n[bold green]🔍 Embeddings Settings[/bold green]")
    embedding_models = ["huggingface", "openai", "voyage", "jina"]
    console.print("Available embedding models:")
    for i, model in enumerate(embedding_models, 1):
        console.print(f"  {i}. {model}")
    
    while True:
        try:
            choice = typer.prompt("Choose embedding model (1-4)", default="1")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(embedding_models):
                config["embedding_model"] = embedding_models[choice_idx]
                break
        except (ValueError, IndexError):
            console.print("[red]Invalid choice. Please enter 1-4.[/red]")
    
    # Reranker settings
    console.print("\n[bold green]🎯 Reranker Settings[/bold green]")
    rerankers = ["huggingface", "cohere"]
    console.print("Available rerankers:")
    for i, reranker in enumerate(rerankers, 1):
        console.print(f"  {i}. {reranker}")
    
    while True:
        try:
            choice = typer.prompt("Choose reranker (1-2)", default="1")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(rerankers):
                config["reranker"] = rerankers[choice_idx]
                break
        except (ValueError, IndexError):
            console.print("[red]Invalid choice. Please enter 1-2.[/red]")
    
    # Qdrant settings
    console.print("\n[bold green]🗄️  Qdrant Database Settings[/bold green]")
    qdrant_host = typer.prompt("Qdrant host", default="localhost")
    qdrant_port = typer.prompt("Qdrant port", default="6333")
    
    # Test Qdrant connection
    try:
        test_client = QdrantClient(host=qdrant_host, port=int(qdrant_port))
        test_client.get_collections()
        console.print("[green]✅ Qdrant connection successful![/green]")
    except Exception:
        console.print("[yellow]⚠️  Warning: Could not connect to Qdrant. Please ensure it's running.[/yellow]")
    
    # Environment variables check
    console.print("\n[bold green]🔑 Environment Variables[/bold green]")
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]❌ OPENAI_API_KEY not found in environment.[/red]")
        if Confirm.ask("Would you like to set it now?"):
            typer.prompt("Enter your OpenAI API key", hide_input=True)
            console.print("[yellow]Note: Add 'export OPENAI_API_KEY=<your_key>' to your shell profile.[/yellow]")
    else:
        console.print("[green]✅ OPENAI_API_KEY found.[/green]")
    
    # Write config file
    config_content = f"""[general]
llm_name = "{config['llm_name']}"

[embeddings]
model = "{config['embedding_model']}"

[reranker]
reranker = "{config['reranker']}"

[qdrant]
host = "{qdrant_host}"
port = {qdrant_port}
"""
    
    with open("config.toml", "w") as f:
        f.write(config_content)
    
    console.print("\n[bold green]🎉 Setup completed! Config saved to config.toml[/bold green]")
    console.print("\n[bold blue]Next steps:[/bold blue]")
    console.print("1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    console.print("2. Try: symbiont ask -d <docs_folder> \"your question\"")
    console.print("3. Check status: symbiont status")

@app.command()
def init():
    """
    📝 Initialize a basic configuration file (use 'setup' for interactive wizard).
    """
    config_content = """[general]
llm_name = "gpt-4o-mini"

[embeddings]
model = "huggingface"

[reranker]
reranker = "huggingface"

[qdrant]
host = "localhost"
port = 6333
"""
    with open("config.toml", "w") as f:
        f.write(config_content)
    console.print("[green]Configuration file 'config.toml' created.[/green]")
    console.print("[blue]💡 Tip: Use 'symbiont setup' for interactive configuration.[/blue]")

@app.command()
def ask(
    question: Optional[str] = typer.Argument(None, help="Question to ask (interactive if not provided)"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection to search"),
    docs: Optional[str] = typer.Option(None, "--docs", "-d", help="Documents directory"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Use last collection"),
    k: int = typer.Option(5, "--results", "-k", help="Number of results"),
    input_file: Optional[str] = typer.Option(None, "--file", "-f", help="Read questions from file"),
    input_format: str = typer.Option("text", "--format", help="Input format: text, json, csv, yaml"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Process multiple questions without interaction"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save output to file (supports .docx, .pdf)"),
):
    """
    🔍 Ask questions about your documents with flexible input formats.
    
    Examples:
        symbiont ask "What is machine learning?"                    # Single question
        symbiont ask --file questions.txt                          # Text file (one per line)
        symbiont ask --file data.json --format json                # JSON format
        symbiont ask --file questions.csv --format csv             # CSV format
        symbiont ask -c economics "GDP trends"                     # Use specific collection
        symbiont ask -d docs/papers "methodology"                  # Auto-create collection
        symbiont ask "research question" -o report.docx            # Save to Word document
        symbiont ask "analysis query" -o results.pdf               # Save to PDF document
    """
    
    # Handle input from file
    questions_list = []
    if input_file:
        if not Path(input_file).exists():
            console.print(f"[red]Input file '{input_file}' not found.[/red]")
            raise typer.Exit(1)
        
        questions_list = parse_input_file(input_file, input_format)
        if not questions_list:
            console.print(f"[red]No questions found in '{input_file}'.[/red]")
            raise typer.Exit(1)
        
        console.print(f"[blue]Loaded {len(questions_list)} questions from {input_file}[/blue]")
    elif question:
        questions_list = [question]
    
    # Handle collection setup
    if quick:
        last_config = load_last_config()
        if not last_config:
            console.print("[red]No previous session found.[/red]")
            console.print("[yellow]Use: symbiont ask -d <docs_dir> -c <collection>[/yellow]")
            raise typer.Exit(1)
        docs = last_config.get("docs_directory")
        collection = last_config.get("collection_name")
        console.print(f"[blue]Using collection: {collection}[/blue]")
    elif docs and not collection:
        collection = Path(docs).name.lower().replace(" ", "-")
        console.print(f"[blue]Auto-detected collection: {collection}[/blue]")
    elif collection and not docs:
        # Collection specified but no docs - check if collection exists
        try:
            config = load_config()
            client = QdrantClient(
                host=config.get("qdrant", {}).get("host", "localhost"),
                port=config.get("qdrant", {}).get("port", 6333),
            )
            collections = [c.name for c in client.get_collections().collections]
            if collection not in collections:
                console.print(f"[red]Collection '{collection}' not found.[/red]")
                console.print(f"[yellow]Available collections: {', '.join(collections)}[/yellow]")
                raise typer.Exit(1)
            console.print(f"[blue]Using existing collection: {collection}[/blue]")
            # Set docs to None for existing collections
            docs = None
        except Exception as e:
            console.print(f"[red]Error checking collection: {e}[/red]")
            raise typer.Exit(1)
    elif not docs and not collection:
        last_config = load_last_config()
        if last_config:
            docs = last_config.get("docs_directory") 
            collection = last_config.get("collection_name")
            console.print(f"[blue]Using last collection: {collection}[/blue]")
        else:
            console.print("[red]No collection specified and no previous session found.[/red]")
            console.print("[yellow]Use: symbiont ask -d <docs_dir> -c <collection>[/yellow]")
            raise typer.Exit(1)
    
    # Validate inputs
    if docs and not Path(docs).exists():
        console.print(f"[red]Directory '{docs}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Interactive question input if none provided
    if not questions_list:
        try:
            question = typer.prompt("❓ Enter your question")
            questions_list = [question]
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)
    
    # Save config for future quick access
    if docs and collection:
        save_last_config({
            "docs_directory": docs,
            "collection_name": collection,
            "k_value": k
        })
    
    # Process questions
    try:
        from symbiont_cli.main import SymbiontCLI
        cli = SymbiontCLI(
            docs_directory=docs,
            collection_name=collection,
            k_value=k,
            llm_response="yes",
            output_directory="search_results",
            q_list=None,
            output_file=output,
        )
        
        if len(questions_list) == 1:
            # Single question
            cli.process_single_question(questions_list[0])
        else:
            # Multiple questions
            console.print(f"[blue]Processing {len(questions_list)} questions...[/blue]")
            for i, q in enumerate(questions_list, 1):
                if not batch:
                    console.print(f"\n[cyan]Question {i}/{len(questions_list)}:[/cyan] {q}")
                    if i < len(questions_list):
                        if not Confirm.ask("Continue to next question?", default=True):
                            break
                else:
                    console.print(f"\n[cyan]Question {i}:[/cyan] {q}")
                
                cli.process_single_question(q)
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

def parse_input_file(file_path, format_type):
    """Parse questions from input file based on format"""
    questions = []
    
    try:
        if format_type == "text":
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        
        elif format_type == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions = [str(q) for q in data]
                elif isinstance(data, dict):
                    questions = data.get("questions", [])
                else:
                    questions = [str(data)]
        
        elif format_type == "csv":
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():  # Use first column
                        questions.append(row[0].strip())
        
        elif format_type == "yaml":
            try:
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, list):
                        questions = [str(q) for q in data]
                    elif isinstance(data, dict):
                        questions = data.get("questions", [])
                    else:
                        questions = [str(data)]
            except ImportError:
                console.print("[red]PyYAML not installed. Install with: pip install pyyaml[/red]")
                raise typer.Exit(1)
        
        else:
            console.print(f"[red]Unsupported format: {format_type}[/red]")
            console.print("[yellow]Supported formats: text, json, csv, yaml[/yellow]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error parsing {file_path}: {e}[/red]")
        raise typer.Exit(1)
    
    return questions

@app.command()
def chat(
    docs_directory: Optional[str] = typer.Argument(None, help="Directory containing documents"),
    collection_name: Optional[str] = typer.Argument(None, help="Collection name (auto-inferred if not provided)"),
    k_value: int = typer.Option(5, "--k", "-k", help="Number of documents to retrieve"),
    llm_response: bool = typer.Option(True, "--llm/--no-llm", help="Use LLM for responses"),
    output_directory: str = typer.Option("search_results", "--output-dir", help="Output directory"),
    q_list: Optional[str] = typer.Option(None, "--questions", "-q", help="File with questions to process"),
    quick: bool = typer.Option(False, "--quick", help="Use last used collection"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save output to file (supports .docx, .pdf)"),
):
    """
    💬 Chat with your documents using AI.
    
    Examples:
        symbiont chat docs/papers economics     # Create/use 'economics' collection
        symbiont chat --quick                   # Use last collection
        symbiont chat docs/research             # Auto-name collection from folder
        symbiont chat docs/papers economics -o session.docx  # Save session to Word
        symbiont chat docs/research --output results.pdf     # Save session to PDF
    """
    # Handle quick mode - use last collection
    if quick:
        last_config = load_last_config()
        if not last_config:
            console.print("[red]No previous session found. Please specify docs and collection.[/red]")
            raise typer.Exit(1)
        docs_directory = last_config.get("docs_directory")
        collection_name = last_config.get("collection_name")
        console.print(f"[green]Using last session: {collection_name}[/green]")
    
    # Auto-infer collection name from directory if not provided
    if docs_directory and not collection_name:
        collection_name = Path(docs_directory).name.lower().replace(" ", "-")
        console.print(f"[blue]Auto-detected collection name: {collection_name}[/blue]")
    
    # Validate inputs
    if not docs_directory or not collection_name:
        console.print("[red]Error: Please provide both docs_directory and collection_name[/red]")
        console.print("[yellow]Usage: symbiont chat <docs_directory> <collection_name>[/yellow]")
        raise typer.Exit(1)
    
    if not Path(docs_directory).exists():
        console.print(f"[red]Error: Directory '{docs_directory}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Save current config for quick mode
    save_last_config({
        "docs_directory": docs_directory,
        "collection_name": collection_name,
        "k_value": k_value
    })
    
    try:
        from symbiont_cli.main import SymbiontCLI
        cli = SymbiontCLI(
            docs_directory=docs_directory,
            collection_name=collection_name,
            k_value=k_value,
            llm_response="yes" if llm_response else "no",
            output_directory=output_directory,
            q_list=q_list,
            output_file=output,
        )
        cli.run()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("collections")
def list_collections(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed collection info"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Filter collections by name"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export collection list to file"),
):
    """
    📚 List and manage collections in the Qdrant database.
    """
    try:
        config = load_config()
        client = QdrantClient(
            host=config.get("qdrant", {}).get("host", "localhost"),
            port=config.get("qdrant", {}).get("port", 6333),
        )
        
        collections = client.get_collections().collections
        
        if not collections:
            console.print("[yellow]No collections found.[/yellow]")
            return
        
        # Filter collections if search term provided
        if search:
            collections = [c for c in collections if search.lower() in c.name.lower()]
            if not collections:
                console.print(f"[yellow]No collections found matching '{search}'.[/yellow]")
                return
            console.print(f"[blue]Showing collections matching '{search}':[/blue]")
        
        # Get last used collection
        last_config = load_last_config()
        last_collection = last_config.get("collection_name") if last_config else None
        
        if detailed:
            table = Table(title="Collections" + (f" (filtered: {search})" if search else ""))
            table.add_column("Name", style="cyan")
            table.add_column("Vectors", style="green")
            table.add_column("Status", style="blue")
            table.add_column("Last Used", style="yellow")
            
            collection_data = []
            for collection in collections:
                try:
                    info = client.get_collection(collection.name)
                    status = "✅ Ready" if info.status == "green" else "⚠️  " + str(info.status)
                    is_last = "🎯 Current" if collection.name == last_collection else ""
                    table.add_row(
                        collection.name,
                        str(info.vectors_count or 0),
                        status,
                        is_last
                    )
                    
                    # Store data for export
                    if export:
                        collection_data.append({
                            "name": collection.name,
                            "vectors": info.vectors_count or 0,
                            "status": info.status,
                            "is_current": collection.name == last_collection
                        })
                except Exception:
                    table.add_row(collection.name, "Unknown", "❌ Error", "")
                    if export:
                        collection_data.append({
                            "name": collection.name,
                            "vectors": "Unknown",
                            "status": "Error",
                            "is_current": False
                        })
            
            console.print(table)
            
            # Export to file if requested
            if export:
                import json
                with open(export, 'w') as f:
                    json.dump(collection_data, f, indent=2)
                console.print(f"[green]Collection data exported to {export}[/green]")
        else:
            console.print("[bold blue]Available Collections:[/bold blue]")
            for collection in collections:
                marker = "🎯 " if collection.name == last_collection else "  • "
                console.print(f"{marker}{collection.name}")
                
    except Exception as e:
        console.print(f"[red]Error connecting to Qdrant: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running on localhost:6333[/yellow]")
        raise typer.Exit(1)

@app.command()
def collection_info(collection_name: str):
    """
    📋 Show detailed information about a specific collection.
    """
    try:
        config = load_config()
        client = QdrantClient(
            host=config.get("qdrant", {}).get("host", "localhost"),
            port=config.get("qdrant", {}).get("port", 6333),
        )
        
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            console.print(f"[red]Collection '{collection_name}' not found.[/red]")
            console.print(f"[yellow]Available collections: {', '.join(collections)}[/yellow]")
            raise typer.Exit(1)
        
        # Get collection info
        info = client.get_collection(collection_name)
        
        # Create info table
        table = Table(title=f"Collection: {collection_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", "✅ Ready" if info.status == "green" else "⚠️  " + str(info.status))
        table.add_row("Vector Count", str(info.vectors_count or 0))
        table.add_row("Vector Size", str(info.config.params.vectors.size))
        table.add_row("Distance Metric", str(info.config.params.vectors.distance))
        
        # Check if it's the current collection
        last_config = load_last_config()
        if last_config and last_config.get("collection_name") == collection_name:
            table.add_row("Current Collection", "🎯 Yes")
        
        console.print(table)
        
        # Show some sample documents if available
        try:
            sample_points = client.scroll(
                collection_name=collection_name,
                limit=3,
                with_payload=True
            )[0]
            
            if sample_points:
                console.print(f"\n[bold blue]Sample Documents (showing {len(sample_points)}):[/bold blue]")
                for i, point in enumerate(sample_points, 1):
                    if point.payload:
                        console.print(f"\n{i}. Document:")
                        for key, value in point.payload.items():
                            if key in ['source', 'title', 'page']:
                                console.print(f"   {key}: {value}")
        except Exception:
            pass  # Sample documents not available
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def remove_collection(collection_name: str):
    """
    🗑️  Remove a collection from the Qdrant database.
    """
    try:
        config = load_config()
        client = QdrantClient(
            host=config.get("qdrant", {}).get("host", "localhost"),
            port=config.get("qdrant", {}).get("port", 6333),
        )
        
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            console.print(f"[red]Collection '{collection_name}' not found.[/red]")
            console.print(f"[yellow]Available collections: {', '.join(collections)}[/yellow]")
            raise typer.Exit(1)
        
        # Confirm deletion
        if Confirm.ask(f"Are you sure you want to delete collection '{collection_name}'?"):
            client.delete_collection(collection_name)
            console.print(f"[green]Collection '{collection_name}' deleted successfully.[/green]")
        else:
            console.print("[yellow]Operation cancelled.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def status():
    """
    📊 Show system status and health check.
    """
    try:
        # Check config file
        if Path("config.toml").exists():
            console.print("[green]✅ Config file found[/green]")
            config = load_config()
        else:
            console.print("[red]❌ Config file missing[/red]")
            console.print("[yellow]Run 'symbiont init' to create config.toml[/yellow]")
            return
        
        # Check Qdrant connection
        try:
            client = QdrantClient(
                host=config.get("qdrant", {}).get("host", "localhost"),
                port=config.get("qdrant", {}).get("port", 6333),
            )
            collections = client.get_collections().collections
            console.print(f"[green]✅ Qdrant connected ({len(collections)} collections)[/green]")
        except Exception:
            console.print("[red]❌ Qdrant connection failed[/red]")
            console.print("[yellow]Make sure Qdrant is running on localhost:6333[/yellow]")
        
        # Check environment variables
        if os.getenv("OPENAI_API_KEY"):
            console.print("[green]✅ OpenAI API key found[/green]")
        else:
            console.print("[red]❌ OPENAI_API_KEY not set[/red]")
            
        # Show last session
        last_config = load_last_config()
        if last_config:
            console.print(f"[blue]📂 Last session: {last_config.get('collection_name')}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")

@app.command()
def profiles(
    create: Optional[str] = typer.Option(None, "--create", "-c", help="Create new profile"),
    use: Optional[str] = typer.Option(None, "--use", "-u", help="Switch to profile"),
    delete: Optional[str] = typer.Option(None, "--delete", "-d", help="Delete profile"),
    copy: Optional[str] = typer.Option(None, "--copy", help="Copy current config to new profile"),
):
    """
    👤 Manage configuration profiles for different setups.
    """
    
    if create:
        # Create new profile
        if Path(f"config_{create}.toml").exists():
            console.print(f"[red]Profile '{create}' already exists.[/red]")
            raise typer.Exit(1)
        
        # Copy from current config.toml or create default
        if Path("config.toml").exists():
            config = load_config()
        else:
            config = {
                "general": {"llm_name": "gpt-4o-mini"},
                "embeddings": {"model": "huggingface"},
                "reranker": {"reranker": "huggingface"},
                "qdrant": {"host": "localhost", "port": 6333}
            }
        
        save_profile_config(create, config)
        console.print(f"[green]Profile '{create}' created.[/green]")
        return
    
    if copy:
        # Copy current config to new profile
        if not Path("config.toml").exists():
            console.print("[red]No config.toml found to copy.[/red]")
            raise typer.Exit(1)
        
        if Path(f"config_{copy}.toml").exists():
            if not Confirm.ask(f"Profile '{copy}' exists. Overwrite?"):
                raise typer.Exit(0)
        
        config = load_config()
        save_profile_config(copy, config)
        console.print(f"[green]Current config copied to profile '{copy}'.[/green]")
        return
    
    if use:
        # Switch to profile
        profile_config = load_profile_config(use)
        if not profile_config:
            console.print(f"[red]Profile '{use}' not found.[/red]")
            available = list_profiles()
            if available:
                console.print(f"[yellow]Available profiles: {', '.join(available)}[/yellow]")
            raise typer.Exit(1)
        
        # Backup current config if it exists
        if Path("config.toml").exists():
            import shutil
            shutil.copy("config.toml", "config.toml.backup")
        
        # Write profile config as current config
        with open("config.toml", "w") as f:
            toml.dump(profile_config, f)
        
        console.print(f"[green]Switched to profile '{use}'.[/green]")
        return
    
    if delete:
        # Delete profile
        profile_path = Path(f"config_{delete}.toml")
        if not profile_path.exists():
            console.print(f"[red]Profile '{delete}' not found.[/red]")
            raise typer.Exit(1)
        
        if Confirm.ask(f"Delete profile '{delete}'?"):
            profile_path.unlink()
            console.print(f"[green]Profile '{delete}' deleted.[/green]")
        return
    
    # List profiles
    profiles_list = list_profiles()
    
    if not profiles_list:
        console.print("[yellow]No profiles found.[/yellow]")
        console.print("[blue]Create one with: symbiont profiles --create <name>[/blue]")
        return
    
    # Show current config info
    current_config = load_config() if Path("config.toml").exists() else None
    
    table = Table(title="Configuration Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("LLM", style="green")
    table.add_column("Embeddings", style="blue")
    table.add_column("Current", style="yellow")
    
    for profile in profiles_list:
        profile_config = load_profile_config(profile)
        if profile_config:
            llm = profile_config.get("general", {}).get("llm_name", "Unknown")
            embedding = profile_config.get("embeddings", {}).get("model", "Unknown")
            
            # Check if this profile matches current config
            is_current = ""
            if current_config:
                if (current_config.get("general", {}).get("llm_name") == llm and
                    current_config.get("embeddings", {}).get("model") == embedding):
                    is_current = "🎯 Active"
            
            table.add_row(profile, llm, embedding, is_current)
    
    console.print(table)
    
    console.print("\n[bold blue]Profile Commands:[/bold blue]")
    console.print("  --create <name>    Create new profile")
    console.print("  --use <name>       Switch to profile")
    console.print("  --copy <name>      Copy current config to new profile")
    console.print("  --delete <name>    Delete profile")

def load_config():
    """Load configuration from config.toml"""
    try:
        with open("config.toml", "r") as f:
            return toml.load(f)
    except FileNotFoundError:
        console.print("[red]Config file not found. Run 'symbiont init' first.[/red]")
        raise typer.Exit(1)

def load_last_config():
    """Load last session configuration"""
    try:
        with open(".symbiont_last.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_last_config(config_data):
    """Save last session configuration"""
    with open(".symbiont_last.json", "w") as f:
        json.dump(config_data, f)

def load_profile_config(profile_name):
    """Load configuration from a specific profile"""
    profile_path = f"config_{profile_name}.toml"
    try:
        with open(profile_path, "r") as f:
            return toml.load(f)
    except FileNotFoundError:
        return None

def save_profile_config(profile_name, config_data):
    """Save configuration to a specific profile"""
    profile_path = f"config_{profile_name}.toml"
    with open(profile_path, "w") as f:
        toml.dump(config_data, f)

def list_profiles():
    """List all available configuration profiles"""
    profiles = []
    for file in Path(".").glob("config_*.toml"):
        profile_name = file.stem.replace("config_", "")
        profiles.append(profile_name)
    return profiles

if __name__ == "__main__":
    app()
