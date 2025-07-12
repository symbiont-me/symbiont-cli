import os
import argparse
import logging
from uuid import uuid4
from langchain_community import vectorstores
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
from langchain.callbacks import get_openai_callback
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from tqdm import tqdm
import time

load_dotenv()
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

load_dotenv()


def init_reranker():
    if os.environ.get("RERANKER") == "cohere":
        logger.info("Using Cohere Reranker")
        return CohereRerank(model="rerank-english-v3.0", top_n=10)
    elif os.environ.get("RERANKER") == "huggingface":
        logger.info("Using HuggingFace CrossEncoder Reranker")
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        return CrossEncoderReranker(model=model, top_n=10)


compressor = init_reranker()


class SymbiontCLI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.llm_name = os.environ.get("LLM_NAME", "gpt-4o-mini")
        logger.critical(f"Using LLM: {self.llm_name}")
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
        parser.add_argument(
            "--q_list",
            type=str,
            default=None,
            help="File containing list of questions to ask",
        )

        args = parser.parse_args()
        if not os.path.isdir(args.docs_directory):
            raise ValueError(f"Directory {args.docs_directory} does not exist")
        return args

    def __remove_next_line(self, text):
        return text.replace("\n", " ")

    def initialize_embeddings(self):
        if os.environ.get("EMBEDDINGS_MODEL", "").lower() == "openai":
            if "OPENAI_API_KEY" in os.environ:
                return OpenAIEmbeddings()
            else:
                raise ValueError("Please set the OPENAI_API_KEY environment variable")
        if "huggingface" in os.environ.get("EMBEDDINGS_MODEL", "").lower():
            model_id = os.environ.get("EMBEDDINGS_MODEL")
            # model_id = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {"device": "cpu"}
            return HuggingFaceEmbeddings(model_name=model_id, model_kwargs=model_kwargs)
        if "voyage" in os.environ.get("EMBEDDINGS_MODEL", "").lower():
            return VoyageAIEmbeddings(
                voyage_api_key=os.environ.get("EMBEDDINGS_MODEL_API_KEY"),
                model=os.environ.get("EMBEDDINGS_MODEL", "voyage-3-lite"),
                batch_size=8,
            )  # type: ignore
        if "jina" in os.environ.get("EMBEDDINGS_MODEL", "").lower():
            api_key = os.environ.get("EMBEDDINGS_MODEL_API_KEY")
            return JinaEmbeddings(
                jina_api_key=api_key,
                model_name="jina-embeddings-v2-base-en",
                session="default",
            )
        return HuggingFaceEmbeddings(
            model_name=os.environ.get(
                "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            # model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def setup_vector_store(self):
        if not self.client.collection_exists(collection_name=self.args.collection_name):
            try:
                logger.info("Creating collection...")
                vector_size = os.getenv("VECTOR_SIZE")
                if isinstance(self.embeddings, JinaEmbeddings):
                    vector_size = 768
                elif isinstance(self.embeddings, OpenAIEmbeddings):
                    vector_size = 1536
                # elif isinstance(self.embeddings, VoyageAIEmbeddings):
                #     vector_size = 1024
                if vector_size is None:
                    raise ValueError("Unknown vector size")
                self.client.create_collection(
                    collection_name=self.args.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                loader = DirectoryLoader(
                    self.args.docs_directory,
                    # glob="**/*.pdf",
                    show_progress=True,
                    loader_cls=PyMuPDFLoader,
                    silent_errors=True,
                    use_multithreading=True,
                )

                documents = loader.load()

                uuids = [str(uuid4()) for _ in range(len(documents))]

                vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.args.collection_name,
                    embedding=self.embeddings,
                )
                with tqdm(total=len(documents), desc="Adding documents") as pbar:
                    for doc, uuid in zip(documents, uuids):
                        try:
                            if isinstance(self.embeddings, JinaEmbeddings):
                                time.sleep(0.01)
                            vector_store.add_documents(documents=[doc], ids=[uuid])
                        except Exception as e:
                            logger.error(f"Error adding document with ID {uuid}: {e}")
                        pbar.update(1)

            except Exception as e:
                logger.error(f"Error adding documents: {e}")

        return QdrantVectorStore(
            client=self.client,
            collection_name=self.args.collection_name,
            embedding=self.embeddings,
        )

    def initialize_llm(self):
        return ChatOpenAI(
            model="gpt-4o",
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

        file_name = f"{self.args.collection_name}.txt"
        with open(os.path.join(logs_directory, file_name), "a") as f:
            f.write(f"Query: {query}\n")
            f.write(f"LLM Response: \n {response}\n")

            f.write("\n" + "=" * 40 + "\n")
            for doc in results:
                f.write("Document Metadata:\n")
                f.write(f"Source: {doc.metadata['source']}, ")
                f.write(f"Title: {doc.metadata['title']}, ")
                f.write(f"Page: {doc.metadata['page']} ")
                f.write(f"Relevance: {doc.metadata.get('relevance_score', 'N/A')}\n")
                # f.write("\n" + self.__remove_next_line(doc.page_content) + "\n")
                f.write("\n" + "=" * 40 + "\n")

    def print_search_results(self, results):
        for doc in results:
            self.context += self.__remove_next_line(doc.page_content) + " "
            logger.info("Document Metadata:")
            for key, value in doc.metadata.items():
                logger.info(f"  {key}: {value}")
            logger.info("Page Content:")
            logger.info("\n" + self.__remove_next_line(doc.page_content))
            logger.info("\n" + "=" * 40 + "\n")

    def perform_search_and_qa(self, query):
        try:
            # results = self.vector_store.similarity_search(query, k=self.args.k_value)

            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 50}),
            )
            results = compression_retriever.invoke(query)
            self.print_search_results(results[::-1])

            if self.args.llm_response.lower() == "no":
                self.log_search_results_to_file(results, query)
                return
            with get_openai_callback() as cb:
                response = self.qa_stuff.run({"context": self.context, "query": query})
                logger.critical("\n" + str(cb))
                logger.info("\n" + response)
            self.log_search_results_to_file(results, query, response)
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
            query = input("Enter your query (or type 'exit' to stop): ")
            if query.lower() == "exit":
                break
            self.perform_search_and_qa(query)

    def run(self):
        try:
            if self.args.q_list:
                print("Running questions from file...")
                self.generate_qa_list(self.args.q_list)
            else:
                self.query_loop()

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
