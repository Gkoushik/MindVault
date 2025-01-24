import nltk
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text_rolling_window(text, window_size=3, similarity_threshold=0.8):
    nltk.download('punkt_tab')
    sentences = sent_tokenize(text)
    print(f"Number of sentences: {len(sentences)}")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    print(f"Shape of sentence embeddings: {sentence_embeddings.shape}")

    chunks = []
    current_chunk = []
    current_embedding = None

    for i in range(len(sentences)):
        if i < window_size:
            current_chunk.append(sentences[i])
            if i == window_size - 1:
                current_embedding = model.encode([" ".join(current_chunk)])[0]
        else:
            next_sentence_embedding = sentence_embeddings[i]
            similarity = cosine_similarity([current_embedding], [next_sentence_embedding])[0][0]
            if similarity >= similarity_threshold:
                current_chunk.append(sentences[i])
                current_embedding = model.encode([" ".join(current_chunk)])[0]
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = sentence_embeddings[i]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
