import openai
import json
import os
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Call OpenAI API to get an embedding vector for the provided text.
    """
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    # The API returns a list of embeddings; use the first one.
    return response['data'][0]['embedding']

def index_text_chunks(chunks, index_path="faiss_index.index", metadata_path="chunk_metadata.json"):
    """
    Create embeddings for each chunk, index them using FAISS, and save the index and metadata.

    :param chunks: A list of text chunks (strings).
    :param index_path: File path to save the FAISS index.
    :param metadata_path: File path to save chunk metadata.
    """
    # Generate embeddings for all chunks.
    embeddings = [get_embedding(chunk) for chunk in chunks]
    # Convert embeddings to a numpy array of type float32.
    embedding_matrix = np.array(embeddings).astype('float32')

    # Get the dimension from the first embedding.
    dim = embedding_matrix.shape[1]

    # Create a FAISS index (using IndexFlatL2 for simplicity).
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)  # add embeddings to index

    # Save the index.
    faiss.write_index(index, index_path)

    # Save metadata: a mapping of index position to the text chunk.
    metadata = {str(i): chunks[i] for i in range(len(chunks))}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(chunks)} chunks. FAISS index saved to {index_path}, metadata saved to {metadata_path}.")

if __name__ == "__main__":
    # Load your preprocessed text chunks from a file (e.g., sample_chunks.json)
    with open("sample_chunks.json", "r") as f:
        text_chunks = json.load(f)

    # Index the chunks.
    index_text_chunks(text_chunks)
