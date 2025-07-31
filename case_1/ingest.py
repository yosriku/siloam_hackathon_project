# ___ ingest.py ___
# Membaca data dan membangun vector store menggunakan Chroma + SentenceTransformer (tanpa token)
import json
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Load .env
load_dotenv()

# Paths
data_path = os.path.join("assets", "doctors_final.json")
collection_name = "doctors_index"
persist_directory = "./chroma_data"

# Load data
with open(data_path, 'r', encoding='utf-8') as f:
    doctors = json.load(f)

# Create LangChain Documents
docs = []
for d in doctors:
    content = (
        f"Name: {d['name']}\n"
        f"Specialization: {d['specialization_name']} ({d['specialization_name_en']})\n"
        f"Sub-specialization: {d['sub_specialization_name']} ({d['sub_specialization_name_en']})\n"
        f"Hospital: {d['hospital_name']}"
    )
    metadata = {"id": d['id'], "name": d['name'], "specialization": d['specialization_name'], "hospital": d['hospital_name']}
    docs.append(Document(page_content=content, metadata=metadata))

# Use HuggingFaceEmbeddings (no token required)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Init Chroma vector store from documents
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)

print("Chroma vector store berhasil dibuat dan disimpan di ./chroma_data.")

