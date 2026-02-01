# RAG

A collection of of Q & A and RAG model implementations, to can help practice. 

## Model 1

An implementation meant to run locally. 

It reads the PDFs in a given directory, extracts data, processes it and stores it in a vector database. Then, based on the question, retrieves relevant data and passes it to the LLM along with the question for response generation. 

### Usage 

Download the code, make the following changes:

- Add API keys to the environment

data_ingestion.py
- Add the paths

data_retrieval_generation.py
- Add the paths
- Enter the user query

Run data_ingestion.py

Run data_retrieval_generation.py

## Model 2

Similar to Model 1, implemented for Databricks. 

Additionally, stores the processed file details in a table. 

### Usage 

Upload the .ipynb files to Databricks, make the following changes:

- Add API keys to the environment
- Add the documents to Databricks 

Data_Ingestion.ipynb
- Make necessary changes to the paths

Data Retrieval.ipynb
- Make necessary changes to the paths

Run Data_Ingestion.ipynb

Run Data Retrieval.ipynb

## Model 3

Similar to Model 2.

Additionally, a model is created using mlflow. 

### Usage 

Upload the files to Databricks, make the following changes:

- Add API keys to the environment
- Add the documents to Databricks
- Run the init_script.sh script 

Data Ingestion_2.py
- Make necessary changes to the paths

Data Retrieval_2.py
- Make necessary changes to the paths

Run Data Ingestion_2.py

Run Data Retrieval_2.py



Google Vertex LLM, Databricks LLM is used to generate responses by providing information, thereby implementing RAG model. 

The basic logic is separated into different scripts for understanding. 

Run data_retrieval_generation.py



