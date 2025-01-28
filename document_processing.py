import os
import psycopg2
from shutil import copy2
from transformers import AutoTokenizer, AutoModel
import torch
from charset_normalizer import detect
from PyPDF2 import PdfReader

# Initialize the Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # You can replace with any model you prefer
model = AutoModel.from_pretrained("bert-base-uncased")

# Database connection function
def connect_to_db():
    try:
        connection = psycopg2.connect(
            database="document_embeddings",
            user="chat-bot",
            password="123",
            host="localhost",
            port="5432"
        )
        print("Database connection successful.")
        return connection
    except Exception as e:
        raise RuntimeError(f"Failed to connect to the database: {e}")

# File upload handling
def handle_file_upload(file_path, upload_dir="uploads/"):
    os.makedirs(upload_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    dest_path = os.path.join(upload_dir, file_name)
    try:
        copy2(file_path, dest_path)
        print(f"File uploaded to: {dest_path}")
    except Exception as e:
        raise IOError(f"Failed to copy file to {upload_dir}: {e}")
    return dest_path

# Extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        print("Text extracted from PDF.")
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

# Read file content with encoding detection
def read_file_with_auto_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        detected_encoding = detect(raw_data)['encoding']
        with open(file_path, 'r', encoding=detected_encoding) as f:
            content = f.read()
        print("Text content successfully read with detected encoding.")
        return content
    except Exception as e:
        raise IOError(f"Failed to read file with automatic encoding detection: {e}")

# Generate embeddings using Hugging Face's Transformer model
def generate_embedding_with_huggingface(text):
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Take the mean of token embeddings to get sentence embedding
        
        # Convert tensor to list for storage
        embedding = embeddings.squeeze().tolist()
        print("Embedding successfully generated.")
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding with Hugging Face: {e}")
    
    return embedding

# Store file data and embeddings in the database
def store_in_db(file_name, file_content, embedding):
    try:
        connection = connect_to_db()
        cursor = connection.cursor()

        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_vectors (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_content TEXT,
                embedding FLOAT8[]
            );
        """)

        # Insert data
        cursor.execute("""
            INSERT INTO file_vectors (file_name, file_content, embedding)
            VALUES (%s, %s, %s);
        """, (file_name, file_content, embedding))

        connection.commit()
        cursor.close()
        connection.close()
        print(f"Data successfully stored for file: {file_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to store data in the database: {e}")

# Complete workflow to process and store file
def process_and_store(file_path):
    try:
        uploaded_path = handle_file_upload(file_path)
        if uploaded_path.endswith('.pdf'):
            file_content = extract_text_from_pdf(uploaded_path)
        else:
            file_content = read_file_with_auto_encoding(uploaded_path)

        # Generate embedding using Hugging Face
        embedding = generate_embedding_with_huggingface(file_content)

        # Store in database
        store_in_db(file_name=os.path.basename(file_path), file_content=file_content, embedding=embedding)
        print(f"File processed and stored successfully: {file_path}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

# Main function
if __name__ == "__main__":
    file_path = input("Enter the file path: ").strip()
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        process_and_store(file_path)

