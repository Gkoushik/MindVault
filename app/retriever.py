import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path="faiss_index.index", embedding_path="embeddings.npy"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
        self.index_path = index_path
        self.embedding_path = embedding_path
        self.index = None  # FAISS index
        self.embeddings = None  # Document embeddings
        self.documents = []  # List of documents

    def add_documents(self, documents):
        self.documents = documents
        # Generate embeddings for the documents
        self.embeddings = self.model.encode(documents)
        # Create a FAISS index
        print(f"Number of documents: {len(self.documents)}")
        print(f"Number of embeddings: {len(self.embeddings)}")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])  # L2 distance index
        self.index.add(np.array(self.embeddings))  # Add embeddings to the index

    def save(self):
        if self.index is None or self.embeddings is None:
            raise ValueError("No documents have been added yet.")
        # Save the FAISS index
        faiss.write_index(self.index, self.index_path)
        # Save the embeddings
        np.save(self.embedding_path, self.embeddings)
        print(f"Saved index to {self.index_path} and embeddings to {self.embedding_path}")

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.embedding_path):
            raise FileNotFoundError("Index or embedding files not found. Please generate them first.")
        # Load the FAISS index
        self.index = faiss.read_index(self.index_path)
        # Load the embeddings
        self.embeddings = np.load(self.embedding_path)
        print(f"Loaded index from {self.index_path} and embeddings from {self.embedding_path}")

    def search(self, query, k=5):
        if self.index is None:
            raise ValueError("Index not loaded. Call `load()` first.")
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        # Search the FAISS index
        distances, indices = self.index.search(np.array(query_embedding), k)
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(self.documents)]
        # Return the relevant documents
        return [self.documents[idx] for idx in valid_indices]
