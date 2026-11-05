
# Week 12 – Semantic Document Search

## Task Title
Building a Semantic Document Search System using LangChain and ChromaDB.

## 1. Introduction
Semantic Document Search is an information retrieval system that finds relevant documents based on their meaning rather than only matching keywords.

This project uses Sentence Transformers, LangChain and ChromaDB to retrieve the most relevant documents for natural language queries.

## 2. Objectives
- Understand semantic search using vector embeddings.
- Retrieve documents based on semantic similarity.
- Integrate LangChain with ChromaDB.
- Retrieve the Top-5 relevant documents.
- Calculate similarity scores.
- Compare semantic search with keyword-based search.
- Prepare the retrieval component for a RAG application.

## 3. Technologies Used
- Python
- Jupyter Notebook
- LangChain
- ChromaDB
- Sentence Transformers
- Pandas
- NumPy

## 4. Embedding Model
Model: all-MiniLM-L6-v2

## 5. Dataset
The project uses 20 text documents covering topics such as:
- Artificial Intelligence
- Machine Learning
- Healthcare
- Education
- Python Programming
- Cybersecurity
- Cloud Computing
- Database Management
- Renewable Energy
- Internet of Things

## 6. Implementation
1. Load the existing ChromaDB database.
2. Add 10 new documents.
3. Connect LangChain with ChromaDB.
4. Load the Sentence Transformer model.
5. Create 15 natural language queries.
6. Perform semantic similarity searches.
7. Retrieve the Top-5 relevant documents.
8. Calculate similarity scores.
9. Perform keyword-based searches.
10. Compare both search methods.
11. Save results into CSV files.

## 7. Output Files
- queries.csv
- search_results.csv
- similarity_scores.csv
- keyword_comparison.csv

## 8. Expected Results
- Total documents: 20
- Total queries: 15
- Semantic search results: 75
- Similarity score records: 75
- Keyword search results: 75

## 9. How to Run
Activate the virtual environment:

python semantic_search.py

To execute the notebook, open:

notebook/Week12_Semantic_Document_Search.ipynb

## 10. Conclusion
The project demonstrates how semantic search retrieves relevant documents by understanding the meaning of user queries.

It uses vector embeddings and similarity search to improve information retrieval. The retrieved documents can be used in the next stage of Retrieval-Augmented Generation (RAG).
