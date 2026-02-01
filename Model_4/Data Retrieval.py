# Databricks notebook source
from databricks_langchain import ChatDatabricks
from langchain_community.vectorstores import FAISS
from databricks_langchain import DatabricksEmbeddings

import os
import faiss
import json

# COMMAND ----------

if "config" not in locals(): config = {}

config['VECTOR_STORE_PATH'] = '/Volumes/genai_databricks/rag_t3/vector_store'

embeddings = DatabricksEmbeddings(endpoint='databricks-bge-large-en')

llm = ChatDatabricks(
    endpoint='databricks-dbrx-instruct',
    temperature=0.1,
)

# COMMAND ----------

def load_vector_store():
    try:
        vector_store = FAISS.load_local(
            folder_path=config['VECTOR_STORE_PATH'],
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")

        return None

# COMMAND ----------

def retrieve_data(query, vector_store):
    try:
        retrieved_data = vector_store.similarity_search(query)

        return retrieved_data
    except Exception as e:
        print(f"Error retrieving data: {e}")

        return []

# COMMAND ----------

def generate_answer(query, retrieved_data=None):
    messages = [
        (
            "system", 
            """Provide helpful responses to user queries, You can use retireved_data for reference. Only give true and to the point answers. If you don't know the answer, say so.""",
        ),
        (
            "human",
            f"User Query: {query}. \nRetrieved Data: {retrieved_data}",
        ),
    ]

    try:
        response = llm.invoke(messages)

        return response
    except Exception as e:
        return f"an error occured: {e}"

# COMMAND ----------

# MAGIC %md
# MAGIC Main

# COMMAND ----------

print("\nStep - Loading vector store")
vector_store = load_vector_store()

# COMMAND ----------

query = dbutils.widgets.get("user_query")
# query = "What are the components of an electric vehicle ?"

# COMMAND ----------

print("\nStep - Retrieving data")
retrieved_data = retrieve_data(query, vector_store)
print("Retrieved Data:", retrieved_data)

# COMMAND ----------

print("\nStep - Generating a response")
response = generate_answer(query, retrieved_data)
print("Response: ", response.content)

# COMMAND ----------

data = {
    "result": "scuccess",
    "response": response.content,
    "response_metadata": response.response_metadata,
    "response_id": response.id
}

dbutils.notebook.exit(json.dumps(data))