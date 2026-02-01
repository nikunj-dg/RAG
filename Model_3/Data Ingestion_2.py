# Databricks notebook source
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from databricks_langchain import DatabricksEmbeddings

import os
import faiss

# COMMAND ----------

if 'config' not in locals(): config = {}

config['DOCS_DIR'] = '/Volumes/genai_databricks/rag_t3/datasets/pdf'
config['VECTOR_STORE_PATH'] = '/Volumes/genai_databricks/rag_t3/vector_store'

embeddings = DatabricksEmbeddings(endpoint = 'databricks-bge-large-en')

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC USE CATALOG genai_databricks;
# MAGIC USE SCHEMA rag_t3;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS rag_t3_files (
# MAGIC   id BIGINT GENERATED ALWAYS AS IDENTITY,
# MAGIC   file_name VARCHAR(128),
# MAGIC   file_type VARCHAR(32),
# MAGIC   file_size VARCHAR(64),
# MAGIC   file_path VARCHAR(256),
# MAGIC   timestamp TIMESTAMP
# MAGIC ) USING DELTA;

# COMMAND ----------

def format_file_size(size_bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f'{size:.2f} {units[unit_index]}'

# COMMAND ----------

def get_document_details(document):
    file_name = os.path.basename(document.metadata['source'])
    file_type = os.path.splitext(file_name)[1][1:]
    file_size = format_file_size(os.path.getsize(document.metadata['source']))

    return [file_name, file_type, file_size]

# COMMAND ----------

def get_new_docs(documents):
    new_documents = []

    for document in documents:
        details = get_document_details(document)
        query = f"""SELECT file_name FROM rag_t3_files WHERE file_name='{details[0]}' AND file_type='{details[1]}';"""
        result = spark.sql(query)

        if result.isEmpty():
            query = f"""
            INSERT INTO rag_t3_files (file_name, file_type, file_size, file_path, timestamp)
            VALUES ('{details[0]}', '{details[1]}', '{details[2]}', '{document.metadata['source']}', NOW());
            """

            result = spark.sql(query)
        
            new_documents.append(document)
    
    return new_documents

# COMMAND ----------

def load_docs():
    documents = []
    file_types = ['*.txt', '*.pdf', '*.html']

    for file_type in file_types:
        loader = DirectoryLoader(config['DOCS_DIR'], glob = file_type, show_progress = True)
        documents.extend(loader.load())
    
    documents = get_new_docs(documents)

    print(f"\nINFO - Documents Loaded, ({len(documents)} files)")

    return documents

# COMMAND ----------

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

# COMMAND ----------

def load_vector_store():
    if os.path.exists(f"{config['VECTOR_STORE_PATH']}/index.faiss"):
        try:
            vector_store = FAISS.load_local(
                folder_path=config['VECTOR_STORE_PATH'],
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
                  

# COMMAND ----------

def embed_store(docs_chunks, vector_store):
    vector_store.add_documents(docs_chunks)
    vector_store.save_local(config['VECTOR_STORE_PATH'])

    print("\nINFO - vector store saved")

# COMMAND ----------

# MAGIC %md
# MAGIC Main

# COMMAND ----------

print("\nStep - Loading the documents")
documents = load_docs()

# COMMAND ----------

if documents:
  print("\nStep - Chunking the documents")
  docs_chunks = split_docs(documents, chunk_overlap=100)
else:
  print("No new files")

# COMMAND ----------

if documents:
  print("\nStep - Loading the vector store")
  vector_store = load_vector_store()

# COMMAND ----------

if documents:
  print("\nStep - Embeddding and storing the chunks")
  embed_store(docs_chunks, vector_store)

# COMMAND ----------

# MAGIC %md
# MAGIC Scratch cell

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM rag_t3_files;