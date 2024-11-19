import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def split_document(document_text, max_length=500):
    """
    Split document into chunks without cutting words.
    """
    words = document_text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def retrieve_answer(question, document_text):
    """
    Retrieve the most relevant answer chunk from the document using TF-IDF.
    """
    if not document_text.strip():
        return None  # Avoid errors with empty documents
    
    chunks = split_document(document_text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks + [question])
    question_vector = tfidf_matrix[-1]  # Vector of the question
    document_vectors = tfidf_matrix[:-1]  # Vectors of document chunks

    # Compute cosine similarity
    similarities = cosine_similarity(question_vector, document_vectors)
    best_match_idx = np.argmax(similarities)

    # Return the most similar chunk if similarity is above threshold
    if similarities[0, best_match_idx] > 0.4:  # Adjust threshold as needed
        return chunks[best_match_idx]
    return None
