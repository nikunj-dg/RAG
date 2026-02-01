"""
LOAD THE REQUIRED LIBRARIES
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from datetime import datetime

import os
import faiss
import csv


"""
SETUP API KEYS
"""

# Load environment variables from .env file
load_dotenv('config.env')

# Get the API key
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if not os.environ.get("DOCS_PATH"):
  os.environ["DOCS_PATH"] = os.getenv("DOCS_PATH")

if not os.environ.get("VECTOR_STORE_PATH"):
  os.environ["VECTOR_STORE_PATH"] = os.getenv("VECTOR_STORE_PATH")

if not os.environ.get("DATABASE_PATH"):
  os.environ["DATABASE_PATH"] = os.getenv("DATABASE_PATH")


"""
SETUP RELEVANT VARIABLES AND MODELS
"""

# Setup the embeddings model 
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


"""
HELPER FUNCTIONS
"""

def format_file_size(size_bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f'{size:.2f} {units[unit_index]}'


def get_document_details(document):
    file_name = os.path.basename(document.metadata['source'])
    file_type = os.path.splitext(file_name)[1][1:]
    file_size = format_file_size(os.path.getsize(document.metadata['source']))

    return [file_name, file_type, file_size, datetime.now()]


def get_new_docs(documents):
    new_documents = []

    old_files = set()
    with open(f"{os.environ("DATABASE_PATH")}\file_details.csv", mode = 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            file = (row['file_name'], row['file_type'], row['file_size'])
            old_files.add(file)

    new_files = []
    for document in documents:
        details = get_document_details(document)
            
        if tuple(details[:3]) not in old_files:
            new_files.append(details) 
            new_documents.append(document)
    
    if new_files:
        with open(f"{os.environ("DATABASE_PATH")}\file_details.csv", mode = 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(new_files)
    
    return new_documents


"""
WRAPPER FUNCTIONS 
"""

def load_docs():
    documents = []
    file_types = ['*.txt', '*.pdf', '*.html']

    for file_type in file_types:
        loader = DirectoryLoader(os.environ("DOCS_PATH"), glob = file_type, show_progress = True)
        documents.extend(loader.load())
    
    documents = get_new_docs(documents)

    print(f"\nINFO - Documents Loaded, ({len(documents)} files)")

    return documents


def split_docs(docs, chunk_size=1000, chunk_overlap=300, add_start_index=True):
    if not docs:
        raise ValueError("There are no documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )

    docs_chunks = text_splitter.split_documents(docs)

    return docs_chunks


def load_vector_store():
    if os.path.exists(f"{os.environ("VECTOR_STORE_PATH")}/index.faiss"):
        try:
            vector_store = FAISS.load_local(
                folder_path=os.environ("VECTOR_STORE_PATH"),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
                )
            
            print("Loaded vector store")

            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")

            return None
    else:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        print("Created a new vector store")

        return vector_store


def embed_store(docs_chunks, vector_store):
    vector_store.add_documents(docs_chunks)
    vector_store.save_local(os.environ("VECTOR_STORE_PATH"))

    print("\nINFO - vector store saved")


"""
MAIN
"""

print("\nStep - Loading the documents")
documents = load_docs()


if documents:
  print("\nStep - Chunking the documents")
  docs_chunks = split_docs(documents, chunk_overlap=100)
else:
  print("No new files")


if documents:
  print("\nStep - Loading the vector store")
  vector_store = load_vector_store()


if documents:
  print("\nStep - Embeddding and storing the chunks")
  embed_store(docs_chunks, vector_store)

