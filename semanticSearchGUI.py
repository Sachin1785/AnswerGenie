import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import tkinter as tk
from tkinter import filedialog

# Step 1: Load your text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 2: Convert text into embeddings
def text_to_embeddings(text, model):
    sentences = text.split('\n')  # Split text by newlines, each sentence is an entry
    embeddings = model.encode(sentences)
    return embeddings

# Modify store_embeddings_in_faiss to initialize FAISS index without adding embeddings
def store_embeddings_in_faiss(model):
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    return index

# Step 4: Semantic search (Query to find similar texts)
def semantic_search(query, model, index, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)  # D = distances, I = indices of the top_k closest vectors
    return I[0]  # Return the indices of the closest matches

# Modify the add_files function to also update the sentences list
def add_files(file_paths, model, index, sentences):
    for file_path in file_paths:
        text = load_text_file(file_path)
        new_sentences = text.split('\n')
        embeddings = text_to_embeddings(text, model)
        index.add(embeddings)
        sentences.extend(new_sentences)

# New function to select files using tkinter
def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select text files", filetypes=[("Text files", "*.txt")])
    return list(root.tk.splitlist(file_paths))

# Modify the on_add_files function to pass the sentences list
def create_gui(model, index, sentences):
    def on_add_files():
        new_files = select_files()
        add_files(new_files, model, index, sentences)
        status_label.config(text="Files added.")

    def on_search():
        query = query_entry.get()
        results = semantic_search(query, model, index)
        result_text.delete(1.0, tk.END)
        for idx in results:
            result_text.insert(tk.END, f"- {sentences[idx]}\n")

    root = tk.Tk()
    root.title("Semantic Search")

    add_button = tk.Button(root, text="Add Files", command=on_add_files)
    add_button.pack()

    query_entry = tk.Entry(root, width=50)
    query_entry.pack()

    search_button = tk.Button(root, text="Search", command=on_search)
    search_button.pack()

    result_text = tk.Text(root, height=15, width=80)
    result_text.pack()

    status_label = tk.Label(root, text="")
    status_label.pack()

    root.mainloop()

# Example usage     
if __name__ == "__main__":
    # Load the pre-trained transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models too

    # Store embeddings in FAISS without adding initial empty embeddings
    index = store_embeddings_in_faiss(model)

    # Initialize sentences list
    sentences = []

    # Create and run the GUI
    create_gui(model, index, sentences)
