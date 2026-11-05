
import os
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_DIR = os.path.join(BASE_DIR, "dataset", "documents")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(DOC_DIR, exist_ok=True)

# 2. Load embedding model
print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

# 3. Connect to existing ChromaDB
client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name="ai_knowledge_base"
)

print("Existing collection connected!")
print("Documents before adding:", collection.count())

# 4. Create 10 new documents
new_documents = {
    "document11.txt": (
        "Python Programming",
        "Python is a popular programming language used in "
        "web development, automation, data science and artificial "
        "intelligence. It has simple syntax and many useful libraries."
    ),

    "document12.txt": (
        "Cybersecurity",
        "Cybersecurity protects computers, networks and data "
        "from cyber attacks. Encryption, firewalls, antivirus "
        "software and strong passwords help improve security."
    ),

    "document13.txt": (
        "Database Management Systems",
        "A Database Management System stores, organizes and "
        "manages data. SQL is used to create tables, insert records, "
        "update information and retrieve data."
    ),

    "document14.txt": (
        "Cloud Computing",
        "Cloud computing provides computing services such as "
        "storage, servers and databases through the internet. "
        "It allows users to access resources without maintaining "
        "their own physical servers."
    ),

    "document15.txt": (
        "Renewable Energy",
        "Renewable energy comes from natural sources such as "
        "sunlight, wind and water. Solar panels and wind turbines "
        "produce electricity while reducing dependence on fossil fuels."
    ),

    "document16.txt": (
        "Environmental Pollution",
        "Environmental pollution occurs when harmful substances "
        "contaminate air, water and soil. Plastic waste, vehicle "
        "emissions and industrial activities contribute to pollution."
    ),

    "document17.txt": (
        "Agriculture Technology",
        "Agricultural technology helps farmers improve crop "
        "production using modern equipment, sensors, drones and "
        "smart irrigation systems. Data analysis helps monitor "
        "soil conditions and plant health."
    ),

    "document18.txt": (
        "Digital Banking",
        "Digital banking allows customers to perform financial "
        "transactions through mobile applications and websites. "
        "Online payments, fund transfers and account management "
        "can be completed without visiting a bank."
    ),

    "document19.txt": (
        "Internet of Things",
        "The Internet of Things connects physical devices to "
        "the internet. Smart home appliances, wearable devices "
        "and industrial sensors collect and exchange information."
    ),

    "document20.txt": (
        "Transportation Technology",
        "Modern transportation technology improves the movement "
        "of people and goods. Electric vehicles, GPS navigation "
        "and intelligent traffic management systems make "
        "transportation more efficient."
    )
}

# 5. Prepare documents
documents = []
metadatas = []
ids = []

for filename, content in new_documents.items():

    title, text = content

    file_path = os.path.join(DOC_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(title + "\n" + text)

    documents.append(title + ". " + text)

    metadatas.append({
        "title": title,
        "category": "General Technology",
        "source": filename
    })

    ids.append(filename.replace(".txt", "").replace("document", "doc"))

print("\n10 new documents created!")

# 6. Generate embeddings
embeddings = model.encode(documents)

# 7. Add documents to existing collection
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("\nNew documents added successfully!")

print("Total documents in ChromaDB:", collection.count())

print("\nWEEK 12 DOCUMENT ADDITION COMPLETED!")
