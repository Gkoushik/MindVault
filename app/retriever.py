import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.chunk_text_semantic import chunk_text_rolling_window
from app.text_extractor import extract_text_from_pdf


class Retriever:
    def __init__(self, index_path="faiss_index.index", embedding_path="embeddings.npy", metadata_path="metadata.json"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
        self.index_path = index_path
        self.embedding_path = embedding_path
        self.metadata_path = metadata_path
        self.index = None  # FAISS index
        self.embeddings = None  # Document embeddings
        self.documents = []  # List of documents
        self.processed_files = self._load_metadata()

    def _load_metadata(self):
        """
        Load the metadata file.
        :return: Dictionary of processed files.
        """
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as file:
                return json.load(file).get("processed_files", {})
        return {}

    def _save_metadata(self):
        """
        Save the metadata file.
        """
        metadata = {"processed_files": self.processed_files}
        with open(self.metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)

    def add_documents(self, documents):
        new_documents = []
        for doc in documents:
            if True:
                try:
                    # Extract text from the file
                    if doc.endswith(".pdf"):
                        text = extract_text_from_pdf(doc)
                    # elif doc.endswith(".docx"):
                    #     text = extract_text_from_docx(doc)
                    else:
                        with open(doc, "r") as file:
                            text = file.read()

                    # Perform semantic chunking
                    chunks = chunk_text_rolling_window(text)
                    new_documents.extend(chunks)  # Ensure chunks are strings

                    # Mark the file as processed
                    self.processed_files[doc] = "2023-10-03T10:00:00"
                except Exception as e:
                    print(f"Error processing file {doc}: {e}")

        # Generate embeddings for the new documents
        if new_documents:
            new_embeddings = self.model.encode(new_documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.documents.extend(new_documents)  # Ensure self.documents contains strings

            # Update the FAISS index
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.embeddings.shape[1])  # L2 distance index
            self.index.add(np.array(new_embeddings))  # Add new embeddings to the index

            # Save the metadata
            self._save_metadata()
        else:
            print("No new documents to process.")

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

    def search(self, query, k=10):
        if self.index is None:
            raise ValueError("Index not loaded. Call `load()` first.")
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        # Search the FAISS index
        distances, indices = self.index.search(np.array(query_embedding), k)
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(self.documents)]
        # Return the relevant documents
        return [self.documents[idx] for idx in valid_indices]
