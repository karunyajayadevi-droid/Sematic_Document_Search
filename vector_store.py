
import os
import csv
import numpy as np
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer

# ----------------------------------
# 1. CREATE PROJECT FOLDERS
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_DIR = os.path.join(BASE_DIR, "dataset", "documents")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------------
# 2. CREATE 10 AI DOCUMENTS
# ----------------------------------

documents_data = {
    "document1.txt": (
        "Introduction to Artificial Intelligence",
        "Artificial Intelligence is a branch of computer science "
        "that develops machines capable of performing tasks that "
        "normally require human intelligence. AI is used in robotics, "
        "healthcare, education and many other fields."
    ),

    "document2.txt": (
        "Machine Learning",
        "Machine Learning is a subset of Artificial Intelligence. "
        "It allows computers to learn patterns from data and make "
        "predictions without being explicitly programmed. "
        "Supervised, unsupervised and reinforcement learning "
        "are major types of machine learning."
    ),

    "document3.txt": (
        "Deep Learning",
        "Deep Learning is a branch of machine learning that uses "
        "multiple layers of neural networks. It learns complex "
        "patterns from large datasets and is useful in speech "
        "recognition, image processing and language understanding."
    ),

    "document4.txt": (
        "Natural Language Processing",
        "Natural Language Processing, also called NLP, enables "
        "computers to understand, interpret and generate human "
        "language. Chatbots, language translation, sentiment "
        "analysis and text summarization use NLP techniques."
    ),

    "document5.txt": (
        "Computer Vision",
        "Computer Vision is a field of Artificial Intelligence "
        "that enables computers to understand images and videos. "
        "It is used in face recognition, object detection, "
        "medical image analysis and self-driving vehicles."
    ),

    "document6.txt": (
        "Neural Networks",
        "Neural Networks are computing systems inspired by the "
        "human brain. They consist of interconnected artificial "
        "neurons arranged in layers. Neural networks learn from "
        "training data and are widely used in deep learning."
    ),

    "document7.txt": (
        "Generative Artificial Intelligence",
        "Generative AI is a technology that creates new content "
        "such as text, images, music and code. It uses trained "
        "models to generate content based on user instructions. "
        "ChatGPT and image generation systems are examples."
    ),

    "document8.txt": (
        "Large Language Models",
        "Large Language Models, known as LLMs, are AI models "
        "trained on large amounts of text data. They can answer "
        "questions, summarize documents, translate languages "
        "and generate human-like text."
    ),

    "document9.txt": (
        "Artificial Intelligence in Healthcare",
        "AI in healthcare helps doctors analyze medical images, "
        "identify diseases and support clinical decisions. "
        "Machine learning models can examine patient data and "
        "assist in early disease detection."
    ),

    "document10.txt": (
        "Artificial Intelligence in Education",
        "AI in education supports personalized learning, automated "
        "grading and intelligent tutoring systems. AI tools can "
        "analyze student performance and recommend suitable "
        "learning materials."
    )
}

documents = []
metadatas = []
ids = []

for i, (filename, content) in enumerate(documents_data.items()):

    title, text = content

    file_path = os.path.join(DOC_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(title + "\n" + text)

    documents.append(title + ". " + text)

    metadatas.append({
        "title": title,
        "category": "Artificial Intelligence",
        "source": filename
    })

    ids.append(f"doc{i + 1}")

print("10 AI documents created successfully!")

# ----------------------------------
# 3. LOAD SENTENCE TRANSFORMER MODEL
# ----------------------------------

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

# ----------------------------------
# 4. GENERATE VECTOR EMBEDDINGS
# ----------------------------------

embeddings = model.encode(documents)

embeddings = np.array(embeddings)

print("\nEmbeddings generated successfully!")
print("Embedding shape:", embeddings.shape)

# Save embeddings as NPY file
np.save(
    os.path.join(DATASET_DIR, "embeddings.npy"),
    embeddings
)

print("Embeddings saved to embeddings.npy")

# ----------------------------------
# 5. CREATE CHROMADB COLLECTION
# ----------------------------------

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# Add documents and embeddings
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("\nDocuments stored in ChromaDB!")
print("Total documents:", collection.count())

# ----------------------------------
# 6. CREATE 10 USER QUERIES
# ----------------------------------

queries = [
    "What is artificial intelligence?",
    "How do computers learn from data?",
    "Explain neural networks and deep learning.",
    "How can computers understand human language?",
    "How does AI identify objects in pictures?",
    "What technology is inspired by the human brain?",
    "How can AI create new text and images?",
    "What are the capabilities of ChatGPT?",
    "How is AI used to detect diseases?",
    "How does AI help students in education?"
]

query_file = os.path.join(DATASET_DIR, "queries.csv")

pd.DataFrame({"query": queries}).to_csv(
    query_file, index=False
)

print("\n10 queries created successfully!")

# ----------------------------------
# 7. PERFORM SEMANTIC SEARCH
# ----------------------------------

results = []

print("\n========== SEMANTIC SEARCH RESULTS ==========")

for query in queries:

    query_embedding = model.encode([query])

    search_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    print("\nQuery:", query)

    for i in range(5):

        doc_id = search_results["ids"][0][i]
        document = search_results["documents"][0][i]
        metadata = search_results["metadatas"][0][i]
        distance = search_results["distances"][0][i]

        similarity = 1 - distance

        print("\nDocument:", metadata["title"])
        print("Similarity:", round(similarity, 4))
        print("Content:", document)

        results.append({
            "query": query,
            "document_id": doc_id,
            "document_title": metadata["title"],
            "category": metadata["category"],
            "source": metadata["source"],
            "similarity_score": round(similarity, 4),
            "retrieved_content": document
        })

# ----------------------------------
# 8. KEYWORD-BASED SEARCH COMPARISON
# ----------------------------------

print("\n========== KEYWORD SEARCH COMPARISON ==========")

comparison_results = []

for query in queries:

    query_words = set(query.lower().split())

    keyword_scores = []

    for i, document in enumerate(documents):

        document_words = set(document.lower().split())

        score = len(query_words.intersection(document_words))

        keyword_scores.append((i, score))

    keyword_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_keyword = keyword_scores[:5]

    for rank, (index, score) in enumerate(top_keyword, start=1):

        comparison_results.append({
            "query": query,
            "keyword_rank": rank,
            "document_title": metadatas[index]["title"],
            "keyword_match_count": score
        })

# ----------------------------------
# 9. SAVE RESULTS TO CSV
# ----------------------------------

results_file = os.path.join(
    DATASET_DIR, "retrieved_results.csv"
)

comparison_file = os.path.join(
    DATASET_DIR, "keyword_comparison.csv"
)

pd.DataFrame(results).to_csv(
    results_file, index=False
)

pd.DataFrame(comparison_results).to_csv(
    comparison_file, index=False
)

print("\n========== PROJECT COMPLETED ==========")

print("Semantic results:", results_file)
print("Keyword comparison:", comparison_file)
print("Embeddings:", "dataset/embeddings.npy")
print("Database:", "chroma_db")
print("\nWeek 11 Vector Embeddings project completed!")
