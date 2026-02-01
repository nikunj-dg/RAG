# RAG_1

A basic RAG model implementation by integrating LLM for a Q&A system. 

Google Vertex LLM is used to generate responses by providing information, thereby implementing RAG model. 

The basic logic is separated into different scripts for understanding. 

data_ingestion.py - Performs ETL on the knowledge base ( PDFs )

data_retrieval_generation.py - Retrieves relevant knowledge from the database and passes it to LLM to generate a response and then displays it. 

## Usage

Download the codebase 

Get the Google API key and add it to config.env 

Run data_ingestion.py

Run data_retrieval_generation.py



