import nltk
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def chunk_text_rolling_window(text, window_size=3, similarity_threshold=0.8):

    nltk.download('punkt_tab')
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    print(f"Number of sentences: {len(sentences)}")

    # Generate sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    print(f"Shape of sentence embeddings: {sentence_embeddings.shape}")

    # Initialize variables
    chunks = []
    current_chunk = []
    current_embedding = None

    # Iterate through sentences with a rolling window
    for i in range(len(sentences)):
        if i < window_size:
            # Add the first few sentences to the current chunk
            current_chunk.append(sentences[i])
            if i == window_size - 1:
                # Compute the embedding for the current chunk
                current_embedding = model.encode([" ".join(current_chunk)])[0]
        else:
            # Compute the similarity between the current chunk and the next sentence
            next_sentence_embedding = sentence_embeddings[i]
            similarity = cosine_similarity([current_embedding], [next_sentence_embedding])[0][0]
            if similarity >= similarity_threshold:
                # Add the sentence to the current chunk
                current_chunk.append(sentences[i])
                # Update the current chunk embedding
                current_embedding = model.encode([" ".join(current_chunk)])[0]
            else:
                # Add the current chunk to the list of chunks
                chunks.append(" ".join(current_chunk))
                # Start a new chunk with the current sentence
                current_chunk = [sentences[i]]
                current_embedding = sentence_embeddings[i]

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Print the chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk[:100]}...")  # Print first 100 characters of each chunk

    return chunks
