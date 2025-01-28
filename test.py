import os
import psycopg2
from shutil import copy2
from ollama import Client
from charset_normalizer import detect
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json

# Initialize Hugging Face and Ollama
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
client = Client(host="http://localhost:11434", headers={})  # Ollama's server endpoint

# Database connection function
def connect_to_db():
    try:
        connection = psycopg2.connect(
            database="document_embeddings2",
            user="chatbot",
            password="123",
            host="localhost",
            port="5432"
        )
        return connection
    except Exception as e:
        raise RuntimeError(f"Database connection failed: {e}")

# File upload handling
def handle_file_upload(file_path, upload_dir="uploads/"):
    os.makedirs(upload_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    dest_path = os.path.join(upload_dir, file_name)
    try:
        copy2(file_path, dest_path)
        print(f"File successfully uploaded to: {dest_path}")
        return dest_path
    except Exception as e:
        raise IOError(f"File upload failed: {e}")

# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        print("Text successfully extracted from PDF.")
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

# Read file with automatic encoding detection
def read_file_with_auto_encoding(file_path):
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        detected_encoding = detect(raw_data)["encoding"]
        with open(file_path, "r", encoding=detected_encoding) as f:
            print(f"File successfully read with detected encoding: {detected_encoding}")
            return f.read()
    except Exception as e:
        raise IOError(f"Failed to read file: {e}")

# Sanitize text to remove null characters
def sanitize_text(text):
    sanitized_text = text.replace('\x00', '')
    print("Text sanitized to remove null characters.")
    return sanitized_text

# Generate embeddings with Hugging Face
def generate_embedding_with_huggingface(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        print("Embedding successfully generated.")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}")

# Query Ollama for responses
def query_ollama(content, context=""):
    try:
        prompt = f"""
        Context: {context}
        Document Content: {content}
        Generate a response based on the above content.
        """
        response = client.generate(model="llama3.2:latest", prompt=prompt)
        print("Raw Response from Ollama:", response)
        return response.get("text", "No response generated.")
    except Exception as e:
        raise RuntimeError(f"Failed to query Ollama: {e}")

# Store file data and embeddings in the database
def store_in_db(file_name, file_content, embedding):
    connection = None
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_vectors (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_content TEXT,
                embedding FLOAT8[]
            );
        """)
        cursor.execute("""
            INSERT INTO file_vectors (file_name, file_content, embedding)
            VALUES (%s, %s, %s);
        """, (file_name, file_content, embedding))
        connection.commit()
        print("Data successfully stored in the database.")
    except Exception as e:
        raise RuntimeError(f"Failed to store data in the database: {e}")
    finally:
        if connection:
            connection.close()

# Continuous Chat with Ollama
def chat_with_ollama(file_path, context=""):
    try:
        print(f"Starting chat for file: {file_path}")

        # Handle file upload
        uploaded_path = handle_file_upload(file_path)

        # Extract content
        if file_path.endswith(".pdf"):
            file_content = extract_text_from_pdf(uploaded_path)
        else:
            file_content = read_file_with_auto_encoding(uploaded_path)

        # Sanitize content
        file_content = sanitize_text(file_content)

        # Generate embedding
        print("Generating embedding...")
        embedding = generate_embedding_with_huggingface(file_content)

        # Store in database
        print("Storing data in the database...")
        store_in_db(os.path.basename(file_path), file_content, embedding)

        # Main chat loop
        print("Chat with Ollama! Type 'exit' to end the conversation.")
        while True:
            question = input("Your question: ")
            if question.lower() == 'exit':
                print("Exiting chat.")
                break

            # Query Ollama with document content and context
            response = query_ollama(file_content, context)

            # Format structured response
            structured_response = {
                "question": question,
                "answer": response,
                "embedding": embedding
            }

            # Print structured response
            print("Response from Ollama:")
            print(json.dumps(structured_response, indent=4))

    except Exception as e:
        print(f"An error occurred during processing: {e}")

# Main function
if __name__ == "__main__":
    file_path = input("Enter the file path: ").strip()
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        chat_with_ollama(file_path, context="This is additional context for Ollama.")
