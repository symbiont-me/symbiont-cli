import os
import logging
import toml
from uuid import uuid4
from langchain_community import vectorstores
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from colorama import Fore, Style, init
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_community.callbacks.manager import get_openai_callback
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain.retrievers import ContextualCompressionRetriever
try:
    from langchain_cohere import CohereRerank
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    CohereRerank = None
from tqdm import tqdm
import time
from .document_generator import DocumentGenerator

load_dotenv()
# Initialize colorama
init(autoreset=True)


def filter_metadata(metadata: dict) -> dict:
    """Filter out unnecessary metadata fields for cleaner output"""
    # Fields to exclude from metadata display
    excluded_fields = {
        'producer', 'creator', 'author', 'subject', 'keywords',
        'creation_date', 'modification_date', 'trapped', 'encrypted'
    }
    
    # Only keep relevant fields
    relevant_fields = {
        'source', 'title', 'page', 'relevance_score', 'file_path'
    }
    
    filtered = {}
    for key, value in metadata.items():
        # Convert key to lowercase for case-insensitive comparison
        key_lower = key.lower()
        
        # Include if it's a relevant field and not in excluded list
        if key_lower in relevant_fields or (key_lower not in excluded_fields and key in relevant_fields):
            filtered[key] = value
            
    return filtered


class ColorHandler(logging.StreamHandler):
    def emit(self, record):
        color = Fore.WHITE
        if record.levelno == logging.INFO:
            color = Fore.GREEN
        elif record.levelno == logging.WARNING:
            color = Fore.YELLOW
        elif record.levelno == logging.ERROR:
            color = Fore.RED
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        super().emit(record)


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = ColorHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()


def init_reranker(config):
    reranker_choice = config.get("reranker", {}).get("reranker", "huggingface")
    if reranker_choice == "cohere":
        if COHERE_AVAILABLE:
            logger.info("Using Cohere Reranker")
            return CohereRerank(model="rerank-english-v3.0", top_n=10)
        else:
            logger.warning("Cohere reranker requested but not available. Falling back to HuggingFace.")
            reranker_choice = "huggingface"
    
    if reranker_choice == "huggingface":
        logger.info("Using HuggingFace CrossEncoder Reranker")
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        return CrossEncoderReranker(model=model, top_n=10)


class SymbiontCLI:
    def __init__(
        self,
        docs_directory: str,
        collection_name: str,
        k_value: int,
        llm_response: str,
        output_directory: str,
        q_list: str | None,
        output_file: str | None = None,
    ):
        load_dotenv()
        self.config = self.load_config()
        self.docs_directory = docs_directory
        self.collection_name = collection_name
        self.k_value = k_value
        self.llm_response = llm_response
        self.output_directory = output_directory
        self.q_list = q_list
        self.output_file = output_file

        logger.info("Initializing SymbiontCLI...")
        if self.docs_directory and not os.path.isdir(self.docs_directory):
            logger.error(f"Directory not found: {self.docs_directory}")
            raise ValueError(f"Directory {self.docs_directory} does not exist")

        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.llm_name = self.config.get("general", {}).get("llm_name", "gpt-4o-mini")
        logger.info(f"Using LLM: {self.llm_name}")
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        self.embeddings = self.initialize_embeddings()
        try:
            self.client = QdrantClient(
                host=self.config.get("qdrant", {}).get("host", "localhost"),
                port=self.config.get("qdrant", {}).get("port", 6333),
            )
            logger.info("Successfully connected to Qdrant.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        self.context = ""
        self.vector_store = self.setup_vector_store()
        self.llm = self.initialize_llm()
        self.qa_stuff = self.setup_qa()
        self.compressor = init_reranker(self.config)
        self.document_generator = DocumentGenerator() if self.output_file else None
        logger.info("SymbiontCLI initialized successfully.")

    def load_config(self):
        if os.path.exists("config.toml"):
            logger.info("Loading configuration from config.toml")
            with open("config.toml", "r") as f:
                return toml.load(f)
        logger.warning("config.toml not found, using default settings.")
        return {}

    def __remove_next_line(self, text):
        return text.replace("\n", " ")

    def initialize_embeddings(self):
        embedding_model = self.config.get("embeddings", {}).get("model", "huggingface")
        logger.info(f"Initializing embedding model: {embedding_model}")
        if embedding_model == "openai":
            if "OPENAI_API_KEY" in os.environ:
                return OpenAIEmbeddings()
            else:
                logger.error("OPENAI_API_KEY not found for OpenAI embeddings.")
                raise ValueError("Please set the OPENAI_API_KEY environment variable")
        if embedding_model == "huggingface":
            model_id = self.config.get("embeddings", {}).get(
                "model_id", "sentence-transformers/all-MiniLM-L6-v2"
            )
            model_kwargs = {"device": "cpu"}
            return HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs)
        if embedding_model == "voyage":
            return VoyageAIEmbeddings(
                voyage_api_key=os.environ.get("EMBEDDINGS_MODEL_API_KEY"),
                model=self.config.get("embeddings", {}).get("model_id", "voyage-3-lite"),
                batch_size=8,
            )  # type: ignore
        if embedding_model == "jina":
            api_key = os.environ.get("EMBEDDINGS_MODEL_API_KEY")
            return JinaEmbeddings(
                jina_api_key=api_key,
                model_name=self.config.get("embeddings", {}).get(
                    "model_id", "jina-embeddings-v2-base-en"
                ),
                session="default",
            )
        return HuggingFaceEmbeddings(
            model_name=self.config.get("embeddings", {}).get(
                "model_id", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    def setup_vector_store(self):
        logger.info(f"Setting up vector store for collection: {self.collection_name}")
        if not self.client.collection_exists(collection_name=self.collection_name):
            if not self.docs_directory:
                logger.error(f"Collection '{self.collection_name}' does not exist and no documents directory provided.")
                raise ValueError(f"Collection {self.collection_name} does not exist. Please provide a documents directory to create it.")
            
            try:
                logger.info(f"Collection '{self.collection_name}' does not exist. Creating new collection...")
                vector_size = self.get_vector_size()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info("Loading documents...")
                loader = DirectoryLoader(
                    self.docs_directory,
                    # glob="**/*.pdf",
                    show_progress=True,
                    loader_cls=PyMuPDFLoader,
                    silent_errors=True,
                    use_multithreading=True,
                )

                documents = loader.load()
                logger.info(f"Found {len(documents)} documents to embed.")

                uuids = [str(uuid4()) for _ in range(len(documents))]

                vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings,
                )
                logger.info("Embedding documents and adding to vector store...")
                with tqdm(total=len(documents), desc="Adding documents") as pbar:
                    for doc, uuid in zip(documents, uuids):
                        try:
                            if isinstance(self.embeddings, JinaEmbeddings):
                                time.sleep(0.01)
                            vector_store.add_documents(documents=[doc], ids=[uuid])
                        except Exception as e:
                            logger.error(f"Error adding document with ID {uuid}: {e}")
                        pbar.update(1)
                logger.info("Successfully created and populated collection.")

            except Exception as e:
                logger.error(f"Error creating collection: {e}")
                raise
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def get_vector_size(self):
        if isinstance(self.embeddings, OpenAIEmbeddings):
            return 1536
        if isinstance(self.embeddings, HuggingFaceEmbeddings):
            # This is a common size, but might need to be more dynamic
            return 384
        if isinstance(self.embeddings, VoyageAIEmbeddings):
            return 1024
        if isinstance(self.embeddings, JinaEmbeddings):
            return 768
        logger.error("Could not determine vector size for the selected embedding model.")
        raise ValueError("Unknown vector size for the selected embedding model.")

    def initialize_llm(self):
        logger.info(f"Initializing LLM: {self.llm_name}")
        return ChatOpenAI(
            model=self.llm_name,
            temperature=0.9,
            api_key=SecretStr(self.api_key),
        )

    def setup_qa(self):
        logger.info("Setting up Question-Answering chain...")
        default_base_prompt = (
            "As an expert, use the following context to answer the question. "
        )
        "Given the following context and question, provide an answer. "
        "Be concise and brief. If the CONTEXT does not provide information. "
        "Answer: 'I don\'t have enough information':"
        base_prompt = os.environ.get("QA_BASE_PROMPT", default_base_prompt)
        custom_prompt = PromptTemplate(
            template=(
                f"{base_prompt}\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

        if vectorstores is None:
            raise ValueError("VectorStores not found")

        logger.critical(self.llm)
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            verbose=True,
            chain_type_kwargs={"prompt": custom_prompt},
        )

    def log_search_results_to_file(
        self, results, query, response="LLM response is turned off"
    ):
        # TODO fix file path issues
        logs_directory = os.path.join("logs")
        if not os.path.isdir(logs_directory):
            os.makedirs(logs_directory)

        file_name = f"{self.collection_name}.txt"
        with open(os.path.join(logs_directory, file_name), "a") as f:
            f.write(f"Query: {query}\n")
            f.write(f"LLM Response: \n {response}\n")

            f.write("\n" + "=" * 40 + "\n")
            for doc in results:
                f.write("Document Metadata:\n")
                filtered_metadata = filter_metadata(doc.metadata)
                for key, value in filtered_metadata.items():
                    f.write(f"{key}: {value}, ")
                f.write("\n")
                # f.write("\n" + self.__remove_next_line(doc.page_content) + "\n")
                f.write("\n" + "=" * 40 + "\n")

    def print_search_results(self, results):
        for doc in results:
            self.context += self.__remove_next_line(doc.page_content) + " "
            logger.info("Document Metadata:")
            filtered_metadata = filter_metadata(doc.metadata)
            for key, value in filtered_metadata.items():
                logger.info(f"  {key}: {value}")
            logger.info("Page Content:")
            logger.info("\n" + self.__remove_next_line(doc.page_content))
            logger.info("\n" + "=" * 40 + "\n")

    def perform_search_and_qa(self, query):
        try:
            logger.info(f"Performing search for query: '{query}'")
            # results = self.vector_store.similarity_search(query, k=self.k_value)

            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 50}),
            )
            results = compression_retriever.invoke(query)
            self.print_search_results(results[::-1])

            response = None
            processing_info = {}
            
            if self.llm_response.lower() == "no":
                self.log_search_results_to_file(results, query)
            else:
                logger.info("Generating response from LLM...")
                with get_openai_callback() as cb:
                    response = self.qa_stuff.run({"context": self.context, "query": query})
                    logger.critical("\n" + str(cb))
                    logger.info("\n" + response)
                    
                    # Store processing info for document generation
                    processing_info = {
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_cost": cb.total_cost if hasattr(cb, 'total_cost') else None
                    }
                
                self.log_search_results_to_file(results, query, response)
            
            # Generate document if output file is specified
            if self.output_file and self.document_generator:
                logger.info(f"Generating document: {self.output_file}")
                self.document_generator.set_data(
                    query=query,
                    collection_name=self.collection_name,
                    search_results=results,
                    llm_response=response,
                    config_info=self.config,
                    processing_info=processing_info
                )
                self.document_generator.generate_document(self.output_file)
                logger.info(f"Document saved: {self.output_file}")
                
        except Exception as e:
            logger.error(f"Error during search and QA: {e}")

    def generate_qa_list(self, questions):
        with open(questions, "r") as f:
            questions = f.readlines()

        for q in questions:
            time.sleep(5)
            self.perform_search_and_qa(q)

    def query_loop(self):
        while True:
            query = input(f"{Fore.CYAN}Enter your query (or type 'exit' to stop): {Style.RESET_ALL}")
            if query.lower() == "exit":
                break
            self.perform_search_and_qa(query)

    def process_single_question(self, question):
        """Process a single question and return the response"""
        try:
            logger.info(f"Processing question: '{question}'")
            self.perform_search_and_qa(question)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    def run(self):
        try:
            if self.q_list:
                logger.info(f"Running questions from file: {self.q_list}")
                self.generate_qa_list(self.q_list)
            else:
                logger.info("Starting interactive query loop...")
                self.query_loop()

        except KeyboardInterrupt:
            logger.info("Exiting gracefully...")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
