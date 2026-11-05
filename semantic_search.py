
import os
import re
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

QUERY_FILE = os.path.join(DATASET_DIR, "queries.csv")
SEARCH_FILE = os.path.join(DATASET_DIR, "search_results.csv")
SCORE_FILE = os.path.join(DATASET_DIR, "similarity_scores.csv")
KEYWORD_FILE = os.path.join(DATASET_DIR, "keyword_comparison.csv")

os.makedirs(DATASET_DIR, exist_ok=True)

# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Sentence Transformer loaded successfully!")

# ==========================================
# 3. CONNECT TO EXISTING CHROMADB
# ==========================================

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name="ai_knowledge_base"
)

print("\nConnected to ChromaDB!")
print("Total documents:", collection.count())

# ==========================================
# 4. CONNECT LANGCHAIN WITH CHROMADB
# ==========================================

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="ai_knowledge_base",
    persist_directory=DB_DIR,
    embedding_function=embedding_function
)

print("LangChain connected with ChromaDB!")

# ==========================================
# 5. CREATE 15 USER QUERIES
# ==========================================

queries = [
    "What is artificial intelligence?",
    "How do machines learn from information?",
    "Explain how the human brain inspires neural networks.",
    "How can computers understand human language?",
    "How does technology recognize objects in photographs?",
    "How can artificial intelligence create content?",
    "What can large language models do?",
    "How does technology help doctors identify illnesses?",
    "How can technology improve student learning?",
    "Which programming language is useful for AI development?",
    "How can we protect computers from hackers?",
    "How do organizations store and manage information?",
    "How can internet-based services provide data storage?",
    "What are clean energy sources that protect nature?",
    "How can smart devices communicate through the internet?"
]

pd.DataFrame({"query": queries}).to_csv(
    QUERY_FILE, index=False
)

print("\n15 user queries created successfully!")

# ==========================================
# 6. SEMANTIC SEARCH FUNCTION
# ==========================================

def semantic_search(query, top_k=5):

    results = vector_store.similarity_search_with_score(
        query,
        k=top_k
    )

    search_data = []

    for rank, (document, distance) in enumerate(results, start=1):

        # Chroma returns cosine distance.
        similarity = 1 - distance

        title = document.metadata.get("title", "Unknown")
        source = document.metadata.get("source", "Unknown")

        search_data.append({
            "query": query,
            "rank": rank,
            "document_title": title,
            "source": source,
            "similarity_score": round(similarity, 4),
            "retrieved_content": document.page_content
        })

    return search_data

# ==========================================
# 7. KEYWORD SEARCH FUNCTION
# ==========================================

stop_words = {
    "what", "is", "the", "a", "an", "how",
    "can", "do", "does", "to", "from", "of",
    "in", "on", "and", "are", "we", "which",
    "that", "it", "for", "through", "with"
}

def get_keywords(text):

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    return set(words) - stop_words


# Retrieve all documents from ChromaDB
all_data = collection.get(
    include=["documents", "metadatas"]
)

all_documents = all_data["documents"]
all_metadatas = all_data["metadatas"]
all_ids = all_data["ids"]

def keyword_search(query, top_k=5):

    query_words = get_keywords(query)

    keyword_results = []

    for i, document in enumerate(all_documents):

        document_words = get_keywords(document)

        score = len(query_words.intersection(document_words))

        keyword_results.append({
            "query": query,
            "document_id": all_ids[i],
            "document_title": all_metadatas[i].get(
                "title", "Unknown"
            ),
            "keyword_match_count": score
        })

    keyword_results.sort(
        key=lambda x: x["keyword_match_count"],
        reverse=True
    )

    return keyword_results[:top_k]

# ==========================================
# 8. SEARCH ALL 15 QUERIES
# ==========================================

semantic_results = []
similarity_results = []
keyword_results_all = []

print("\n========== SEMANTIC SEARCH RESULTS ==========")

for query in queries:

    print("\nUSER QUERY:", query)

    results = semantic_search(query, top_k=5)

    for item in results:

        print("\nRank:", item["rank"])
        print("Document:", item["document_title"])
        print("Similarity:", item["similarity_score"])
        print("Content:", item["retrieved_content"])

        semantic_results.append(item)

        similarity_results.append({
            "query": item["query"],
            "document_title": item["document_title"],
            "similarity_score": item["similarity_score"]
        })

    # Perform keyword search
    keyword_data = keyword_search(query, top_k=5)

    for item in keyword_data:
        keyword_results_all.append(item)

# ==========================================
# 9. SAVE RESULTS INTO CSV FILES
# ==========================================

pd.DataFrame(semantic_results).to_csv(
    SEARCH_FILE, index=False
)

pd.DataFrame(similarity_results).to_csv(
    SCORE_FILE, index=False
)

pd.DataFrame(keyword_results_all).to_csv(
    KEYWORD_FILE, index=False
)

print("\n========== CSV FILES SAVED ==========")

print("Search results:", SEARCH_FILE)
print("Similarity scores:", SCORE_FILE)
print("Keyword comparison:", KEYWORD_FILE)

# ==========================================
# 10. KEYWORD VS SEMANTIC COMPARISON
# ==========================================

print("\n========== SEARCH COMPARISON ==========")

for query in queries:

    semantic_top = [
        item["document_title"]
        for item in semantic_results
        if item["query"] == query
    ]

    keyword_top = [
        item["document_title"]
        for item in keyword_results_all
        if item["query"] == query
    ]

    common = set(semantic_top).intersection(keyword_top)

    print("\nQuery:", query)
    print("Common documents:", len(common))
    print("Semantic results:", semantic_top[:3])
    print("Keyword results:", keyword_top[:3])

# ==========================================
# 11. ACCEPT QUERY FROM KEYBOARD
# ==========================================

print("\n========== INTERACTIVE SEARCH ==========")

while True:

    user_query = input(
        "\nEnter your search query (or type exit): "
    ).strip()

    if user_query.lower() == "exit":
        print("Semantic search application closed!")
        break

    if not user_query:
        print("Please enter a valid query.")
        continue

    results = semantic_search(user_query, top_k=5)

    print("\nTOP-5 RELEVANT DOCUMENTS")

    for item in results:

        print("\nDocument:", item["document_title"])
        print("Similarity:", item["similarity_score"])
        print("Content:", item["retrieved_content"])

print("\nWEEK 12 SEMANTIC DOCUMENT SEARCH COMPLETED!")
