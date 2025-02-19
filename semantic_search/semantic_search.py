import os
import openai
import json
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root.
load_dotenv()

# Retrieve and check the API key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY in your .env file.")

openai.api_key = OPENAI_API_KEY

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get embedding vector for a text using OpenAI's API.
    """
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

def load_faiss_index(index_path="faiss_index.index"):
    """
    Load the FAISS index from disk.
    """
    return faiss.read_index(index_path)

def load_metadata(metadata_path="chunk_metadata.json"):
    """
    Load the chunk metadata from disk.
    """
    with open(metadata_path, "r") as f:
        return json.load(f)

def semantic_search(query, k=3, index_path="faiss_index.index", metadata_path="chunk_metadata.json"):
    """
    What are the main topics in this text? 
    """
    try:
        # Check if files exist
        if not os.path.exists(index_path):
            print(f"Error: Index file not found at {index_path}")
            return []
        if not os.path.exists(metadata_path):
            print(f"Error: Metadata file not found at {metadata_path}")
            return []

        print(f"Processing query: {query}")  # Debug output
        
        # Convert query to embedding
        query_embedding = np.array([get_embedding(query)]).astype('float32')
        print("Generated embedding")  # Debug output

        # Load index and search
        index = load_faiss_index(index_path)
        print("Loaded FAISS index")  # Debug output
        
        distances, indices = index.search(query_embedding, k)
        print(f"Search complete. Found {len(indices[0])} results")  # Debug output

        # Load metadata and get results
        metadata = load_metadata(metadata_path)
        print(f"Loaded metadata with {len(metadata)} entries")  # Debug output
        
        results = []
        for i, idx in enumerate(indices[0]):
            if str(idx) in metadata:
                results.append({
                    'distance': float(distances[0][i]),
                    'text': metadata[str(idx)]
                })
        return results

    except Exception as e:
        print(f"Error during semantic search: {e}")
        import traceback
        traceback.print_exc()  # Print the full error traceback
        return []

if __name__ == "__main__":
    print("Starting semantic search test...")  # Debug output
    
    # Test queries
    test_queries = [
        "What is the main topic of this text?",
        "Can you explain the key concepts?",
        "What are the important points discussed?"
    ]
    
    print("Running semantic search test...\n")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = semantic_search(query, k=2)
        if results:
            print("\nTop matching chunks:")
            for i, res in enumerate(results, 1):
                print(f"\n{i}. Distance: {res['distance']:.4f}")
                # Check if the metadata is a dict; if not, assume it's already a string
                if isinstance(res['text'], dict):
                    text_chunk = res['text'].get("text", "")
                else:
                    text_chunk = res['text']
                print(f"Text: {text_chunk[:200]}...")
        else:
            print("No results found")
        print("\n" + "="*80 + "\n")


