# ___ app.py ___
# Aplikasi Flask untuk retrieval dan generasi respons menggunakan Chroma + SentenceTransformer (tanpa token)
import os
from dotenv import load_dotenv
from flask import Flask, request, render_template
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Load .env
load_dotenv()

# Setup
collection_name = "doctors_index"
persist_directory = "./chroma_data"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name=collection_name
)

# Initialize Flask
app = Flask(__name__)

# LLM dari HuggingFace
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

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
