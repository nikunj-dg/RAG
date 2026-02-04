# RAG Practice

A collection of Q&A and RAG model implementations built to understand and practice end-to-end Retrieval-Augmented Generation pipelines across:

- Local FAISS-based RAG
- Databricks RAG 
- Databricks + MLflow Model
- Databricks DBRX Instruct LLM

This project walks through:

- Document ingestion
- Metadata tracking (Delta Table)
- Chunking and embedding
- FAISS vector storage
- Similarity search retrieval
- Prompt augmentation
- LLM response generation
- MLflow integration (Model 3)
- Databricks production-style execution

## Model 1 - Local RAG 

- Runs locally
- Reads PDFs from directory
- Splits into chunks
- Stores vectors in FAISS
- Retrieves context
- Sends context + query to LLM

Make sure to:

- Add API keys
- Update document paths
- Enter your query

## Model 2 - Databricks RAG + Metadata Tracking

- Uses Databricks Embeddings endpoint
- Uses Unity Catalog Volumes
- Stores file metadata in Delta table
- Skips already processed files

## Model 3 - Databricks + MLflow

- Model logging via MLflow
- Model versioning
- Reproducibility
- Better production readiness

## Technologies Used

- Python
- LangChain
- FAISS
- Databricks
- Unity Catalog
- Delta Lake
- MLflow
- Databricks Embeddings
- DBRX Instruct LLM

## Future Improvements

- API wrapper (FastAPI)
- Streaming responses
- Deployment to serving endpoint
- Automated document monitoring


