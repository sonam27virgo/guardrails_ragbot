from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import configparser
import os


class DataPreProcessor:
    def __init__(self, config_path='config.env'):
        """Initialize the DataPreProcessor with configuration settings."""
        self.configs = configparser.ConfigParser()
        self.configs.read(config_path)

        try:
            self.filepath = Path(self.configs['DATASOURCE']['FILEPATH'])
            self.chunk_size = int(self.configs['CHUNKING']['CHUNK_SIZE'])
            self.chunk_overlap = int(self.configs['CHUNKING']['CHUNK_OVERLAP'])
        except KeyError as e:
            raise KeyError(f"Missing configuration key: {e}")

        self.verbose = False
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def load_data(self):
        """Load text documents from the specified directory."""
        documents = []
        if not self.filepath.exists() or not self.filepath.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.filepath}")

        for file in self.filepath.iterdir():
            if file.is_file():
                loader = TextLoader(str(file))
                try:
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file.name}: {e}")

        return documents

    def clean_text(self, text):
        """Basic text cleaning (Placeholder for future implementation)."""
        return text.strip()

    def chunk_documents(self, documents):
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def create_embeddings_and_insert_db(self, chunked_docs):
        """Generate embeddings and insert them into a Chroma vector database."""
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory = "vectordb")

        batch_size = 20
        total_docs = len(chunked_docs)

        for i in range(0, total_docs, batch_size):
            batch = chunked_docs[i:i + batch_size]
            self.vector_store.add_documents(documents=batch)

        if self.verbose:
            print('Embeddings inserted into Vector DB successfully.')

    def prepare_data(self):
        """Load, chunk, and insert data into the vector database."""
        if self.verbose:
            print("Reading and loading data...")

        documents = self.load_data()
        chunked_docs = self.chunk_documents(documents)
        self.create_embeddings_and_insert_db(chunked_docs)

    def load_vectordb(self):
        self.vector_store = Chroma(persist_directory = "vectordb", embedding_function=self.embeddings)
        return self.vector_store
