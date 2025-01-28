# Required Libraries
import os
import flask
from flask import Flask, request, jsonify
from langchain.vectorstores import PostgreSQL
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader

# Initialize Flask App
app = Flask(__name__)

# Load Environment Variables for Keys and DB Info
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
POSTGRES_URL = os.getenv('POSTGRES_URL')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')

# Initialize PostgreSQL Vector Store
vector_store = PostgreSQL(
    connection_string=f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_URL}/{POSTGRES_DB}",
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OLLAMA_API_KEY),
    table_name="document_embeddings",
)

# Load Ollama Chat Model
chat_model = ChatOpenAI(model="gpt-4", temperature=0, api_key=OLLAMA_API_KEY)

@app.route("/upload", methods=["POST"])
def upload_document():
    """Endpoint for uploading documents."""
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    # Detect file type and load content
    file_type = file.filename.split('.')[-1].lower()
    if file_type == 'pdf':
        loader = PyPDFLoader(file)
    elif file_type in ['csv', 'txt']:
        loader = CSVLoader(file) if file_type == 'csv' else TextLoader(file)
    else:
        return jsonify({"error": "Unsupported file type."}), 400

    # Load and process documents
    documents = loader.load()
    vector_store.add_documents(documents)
    return jsonify({"message": "Document uploaded and processed successfully."}), 200

@app.route("/chat", methods=["POST"])
def chat_with_document():
    """Endpoint for chatting with document content."""
    query = request.json.get("query")
    chat_history = request.json.get("history", [])

    if not query:
        return jsonify({"error": "Query not provided."}), 400

    # Retrieve relevant content from the vector store
    chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        llm=chat_model,
    )

    # Generate response
    result = chain.run(query=query, chat_history=chat_history)
    return jsonify({"response": result}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

