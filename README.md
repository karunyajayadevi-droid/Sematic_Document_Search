
# Week 11 – Vector Embeddings with ChromaDB

## Project Title
Creating and Storing Vector Embeddings using ChromaDB

## Objective
- Understand vector databases in LLM applications.
- Generate embeddings using Sentence Transformers.
- Store embeddings in ChromaDB.
- Perform semantic similarity searches.
- Prepare a knowledge base for RAG.

## Technologies Used
- Python
- VS Code
- Sentence Transformers
- ChromaDB
- NumPy
- Pandas
- Jupyter Notebook

## Embedding Model
all-MiniLM-L6-v2

## Dataset
10 text documents related to Artificial Intelligence.

## Implementation
1. Created 10 AI text documents.
2. Loaded the Sentence Transformer model.
3. Generated vector embeddings.
4. Saved embeddings in embeddings.npy.
5. Created a ChromaDB collection.
6. Stored documents with metadata.
7. Created 10 user queries.
8. Performed Top-5 semantic similarity searches.
9. Compared semantic search with keyword search.
10. Exported results into CSV files.

## Output Files
- embeddings.npy
- queries.csv
- retrieved_results.csv
- keyword_comparison.csv
- chroma_db/

## Conclusion
The project successfully demonstrates how text documents
can be converted into vector embeddings and stored in
ChromaDB for semantic retrieval.

The resulting vector database can be used in the next
week's Retrieval-Augmented Generation project.
