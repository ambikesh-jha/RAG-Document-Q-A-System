import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a FAISS index and metadata dictionary
dimension = model.get_sentence_embedding_dimension()  # Embedding size for MiniLM
index = faiss.IndexFlatL2(dimension)  # L2 distance-based FAISS index
metadata = {}  # Stores mappings between index positions and document filenames


def text_to_embeddings(text):
    """
    Converts a list of text chunks into embeddings.
    Args:
        text (list of str): Text chunks to encode.
    Returns:
        numpy.ndarray: Array of embeddings.
    """
    embeddings = model.encode(text, convert_to_numpy=True)
    return embeddings


def split_document(document_text, max_length=500):
    """
    Splits a document into chunks without cutting off words.
    Args:
        document_text (str): The document text to split.
        max_length (int): Maximum length of each chunk in words.
    Returns:
        list of str: Chunks of the document.
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

def create_faiss_index(document_text, index, filename):
    """
    Adds a document's embeddings to the FAISS index.
    Args:
        document_text (str): The document text to index.
        index (faiss.IndexFlatL2): The FAISS index object.
        filename (str): The filename of the document.
    Returns:
        None
    """
    try:
        
        # Split document into chunks
        chunks = split_document(document_text)

        # Convert chunks to embeddings
        embeddings = text_to_embeddings(chunks)

        # Convert embeddings to float32 and add to FAISS index
        embeddings_np = np.array(embeddings, dtype="float32")
        index.add(embeddings_np)

        # Store metadata for retrieval
        for idx, chunk in enumerate(chunks):
            metadata[index.ntotal - len(chunks) + idx] = chunk

        print(f"Added {len(chunks)} chunks from {filename} to the FAISS index.")
    except Exception as e:
        raise ValueError(f"Failed to create FAISS index: {str(e)}")

def query_faiss_index(question, index):
    """
    Queries the FAISS index to find the most relevant document chunk for the question.
    Args:
        question (str): The user's query.
        index (faiss.IndexFlatL2): The FAISS index object.
    Returns:
        str: The most relevant chunk of text or None if no match is found.
    """
    try:
        # Convert the question to an embedding
        question_embedding = model.encode([question], convert_to_numpy=True)

        # Search the FAISS index
        distances, indices = index.search(np.array(question_embedding, dtype="float32"), k=1)

        # Check if a match was found
        if distances[0][0] < 1.0:  # Adjust the threshold as needed
            result_idx = indices[0][0]
            return metadata.get(result_idx, "No matching metadata found.")
        else:
            return "No relevant match found in the indexed documents."
    except Exception as e:
        raise ValueError(f"Failed to query FAISS index: {str(e)}")
