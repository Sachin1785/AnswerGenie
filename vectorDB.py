import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Step 1: Load your text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 2: Convert text into embeddings
def text_to_embeddings(text, model):
    sentences = text.split('\n')  # Split text by newlines, each sentence is an entry
    embeddings = model.encode(sentences)
    return embeddings

# Step 3: Store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    # Create a FAISS index for dense vectors (L2 distance)
    dimension = embeddings.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Add embeddings to FAISS index
    return index

# Step 4: Semantic search (Query to find similar texts)
def semantic_search(query, model, index, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)  # D = distances, I = indices of the top_k closest vectors
    return I[0]  # Return the indices of the closest matches

# New function to add multiple files
def add_files(file_paths, model, index):
    for file_path in file_paths:
        text = load_text_file(file_path)
        embeddings = text_to_embeddings(text, model)
        index.add(embeddings)

# Example usage     
if __name__ == "__main__":
    # Load the pre-trained transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models too

    # Load text from file
    file_path = 'UHV Chpt 1-15 all (1)[1].txt'  # Replace with your actual file path
    text = load_text_file(file_path)

    # Convert text to embeddings
    embeddings = text_to_embeddings(text, model)

    # Store embeddings in FAISS
    index = store_embeddings_in_faiss(embeddings)

    while True:
        user_input = input("Enter a query or 'add' to add files, 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'add':
            new_files = input("Enter file paths separated by commas: ").split(',')
            new_files = [file.strip() for file in new_files]
            add_files(new_files, model, index)
            print("Files added.")
        else:
            results = semantic_search(user_input, model, index)
            sentences = text.split('\n')
            print("Top matching sentences:")
            for idx in results:
                print(f"- {sentences[idx]}")
