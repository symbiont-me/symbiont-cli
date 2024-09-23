import os
import argparse
import logging
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from colorama import Fore, Style, init
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr

# Initialize colorama
init(autoreset=True)


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


class SymbiontCLI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.llm_name = os.environ.get("LLM_NAME", "gpt-3.5-turbo")
        if not self.api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        self.embeddings = self.initialize_embeddings()
        self.client = QdrantClient("localhost:6333")
        self.context = ""
        self.args = self.parse_arguments()
        self.vector_store = self.setup_vector_store()
        self.llm = self.initialize_llm()
        self.qa_stuff = self.setup_qa()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(
            description="Process documents and store embeddings."
        )
        parser.add_argument(
            "--docs_directory",
            type=str,
            required=True,
            help="Directory to load documents from",
        )
        parser.add_argument(
            "--collection_name",
            type=str,
            required=True,
            help="Name of the Qdrant collection",
        )
        parser.add_argument(
            "--k_value",
            type=int,
            default=3,
            help="Number of documents to retrieve",
        )
        parser.add_argument(
            "--llm_response",
            type=str,
            default="yes",
            help="If you want to use the LLM for responses or plain similarity search",
        )
        parser.add_argument(
            "--output_directory",
            type=str,
            default="search_results",
            help="Directory to save search results",
        )

        args = parser.parse_args()
        if not os.path.isdir(args.docs_directory):
            raise ValueError(f"Directory {args.docs_directory} does not exist")
        return args

    def initialize_embeddings(self):
        if "OPENAI_API_KEY" in os.environ:
            return OpenAIEmbeddings()
        else:
            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {"device": "cpu"}
            return HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs)

    def setup_vector_store(self):
        if not self.client.collection_exists(collection_name=self.args.collection_name):
            logger.info("Creating collection...")
            vector_size = 1536 if isinstance(self.embeddings, OpenAIEmbeddings) else 384
            self.client.create_collection(
                collection_name=self.args.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            loader = DirectoryLoader(
                self.args.docs_directory,
                glob="**/*.pdf",
                show_progress=True,
                loader_cls=PyMuPDFLoader,
            )
            documents = loader.load()
            uuids = [str(uuid4()) for _ in range(len(documents))]
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.args.collection_name,
                embedding=self.embeddings,
            )
            vector_store.add_documents(documents=documents, ids=uuids)
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.args.collection_name,
            embedding=self.embeddings,
        )

    def initialize_llm(self):
        return ChatOpenAI(
            model=self.llm_name,
            temperature=0.9,
            api_key=SecretStr(self.api_key),
        )

    def setup_qa(self):
        default_base_prompt = (
            "As an expert, use the following context to answer the question. "
        )
        "Given the following context and question, provide an answer. "
        "Be concise and brief. If the CONTEXT does not provide information. "
        "Answer: 'I don't have enough information':"
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
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            verbose=True,
            chain_type_kwargs={"prompt": custom_prompt},
        )

    def __remove_next_line(self, text):
        return text.replace("\n", " ")

    def print_search_results(self, results):
        for doc in results:
            self.context += self.__remove_next_line(doc.page_content) + " "
            logger.info("Document Metadata:")
            for key, value in doc.metadata.items():
                logger.info(f"  {key}: {value}")
            logger.info("\nPage Content:")
            logger.info(self.__remove_next_line(doc.page_content))
            logger.info("\n" + "=" * 40 + "\n")

    def perform_search_and_qa(self, query):
        try:
            results = self.vector_store.similarity_search(query, k=self.args.k_value)
            self.print_search_results(results)
            if self.args.llm_response.lower() == "no":
                return
            response = self.qa_stuff.run({"context": self.context, "query": query})
            logger.info(response)
        except Exception as e:
            logger.error(f"Error during search and QA: {e}")

    def run(self):
        try:
            while True:
                query = input("Enter your query (or type 'exit' to stop): ")
                if query.lower() == "exit":
                    break
                self.perform_search_and_qa(query)
        except KeyboardInterrupt:
            logger.info("Exiting gracefully...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    try:
        cli = SymbiontCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Failed to start SymbiontCLI: {e}")
