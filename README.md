
# 🧠 Flask RAG App with LangChain and Gemini

## 📦 Project Structure
- `app.py` : Main Flask web application for query-answering
- `ingest.py` : Script to embed and store PDF content into Chroma vector store
- `templates/` : Folder for HTML templates (`search.html`)
- `.env` : Stores your API keys (e.g. `HUGGINGFACEHUB_API_TOKEN`, `GOOGLE_API_KEY`)

## 🛠️ Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/yosriku/siloam_hackathon_project.git
cd siloam_hackathon_project

# 2. Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate     # On Windows: myenv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🔑 .env Format
```
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```

## 📥 Ingest Data
Jalankan untuk mengubah menyimpan data dalam bentuk vector:
```bash
python ingest.py
```

## 🚀 Run the App
```bash
python app.py
```
Akses melalui browser: [http://localhost:8000](http://localhost:8000)

## 🖼️ UI Preview
Form is located at `/templates/search.html`. Basic HTML with query input and result display.

## ✅ Requirements
- Python 3.10.0
- Flask
- LangChain >= 0.2.x
- Chroma
- SentenceTransformers


