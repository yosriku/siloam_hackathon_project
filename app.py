# app.py

import os
from dotenv import load_dotenv
from flask import Flask, request, render_template

# LangChain imports
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Konfigurasi direktori vektor dan embedding
collection_name = "doctors_index"
persist_directory = "./chroma_data"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name=collection_name
)

# Gunakan Gemini Pro dari Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # gunakan model cepat
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5
)


# Buat QA chain dari retriever dan LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def search():
    answer = None
    sources = []
    if request.method == 'POST':
        query = request.form.get('query')
        result = qa_chain(query)
        answer = result.get('result')
        sources = [doc.metadata for doc in result.get('source_documents', [])]
    return render_template('search.html', answer=answer, sources=sources)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
