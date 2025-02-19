import os
import json
import argparse
import faiss
import numpy as np

from data_ingestion.parse_documents import parse_pdf, clean_text, chunk_text
from semantic_search.semantic_search import (
    get_embedding,
    load_faiss_index,
    load_metadata
)

import openai
from dotenv import load_dotenv

load_dotenv()  # Ensure .env is loaded
openai.api_key = os.getenv("OPENAI_API_KEY")

def index_multiple_docs(
    pdf_directory="data/pdfs",
    index_path="faiss_index.index",
    metadata_path="chunk_metadata.json"
):
    """
    Ingest and index multiple PDF files from a directory. Appends to an existing FAISS index if present.
    :param pdf_directory: Directory containing PDF files.
    :param index_path: Path to the FAISS index file.
    :param metadata_path: Path to the metadata JSON file.
    """
    # 1. Gather PDF files
    pdf_files = [
        f for f in os.listdir(pdf_directory)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}. Exiting.")
        return

    # 2. Load existing FAISS index & metadata if they exist
    index = None
    existing_metadata = {}
    if os.path.exists(index_path):
        index = load_faiss_index(index_path)
        existing_metadata = load_metadata(metadata_path)
        print(f"Loaded existing FAISS index with {index.ntotal} vectors.")
    else:
        print("No existing index found. A new one will be created.")

    # 3. Prepare data structures for new embeddings & metadata
    all_new_embeddings = []
    new_metadata_dict = {}

    # We'll track a running ID offset so new chunks don't overwrite old ones
    offset = 0
    if existing_metadata:
        # existing_metadata keys are strings of integer IDs
        existing_ids = list(map(int, existing_metadata.keys()))
        offset = max(existing_ids) + 1  # Start after the highest existing ID

    # 4. Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing: {pdf_path}")

        # Parse & chunk
        raw_text = parse_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text, max_chunk_size=4000)

        # For a real system, you might store doc metadata or a unique doc_id
        doc_id = pdf_file  # e.g., use the filename as doc_id

        # 5. Generate embeddings for each chunk
        for chunk in chunks:
            embedding = get_embedding(chunk)  # single call per chunk (could be batched)
            all_new_embeddings.append(embedding)

            # Add to new metadata
            new_metadata_dict[str(offset)] = {
                "doc_id": doc_id,
                "text": chunk
            }
            offset += 1

    # 6. Convert all_new_embeddings to a numpy array
    if not all_new_embeddings:
        print("No new embeddings to add. Exiting.")
        return
    embedding_matrix = np.array(all_new_embeddings).astype('float32')

    # 7. If index doesn't exist, create it. Otherwise, add to it.
    if index is None:
        # Use the dimension of the first embedding to create an index
        dim = embedding_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        print("Created a new FAISS index.")

    # 8. Add new embeddings to the index
    index.add(embedding_matrix)
    print(f"Added {len(all_new_embeddings)} new vectors to the index.")

    # 9. Save the updated index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}.")

    # 10. Update and save metadata
    # Merge new metadata into existing
    existing_metadata.update(new_metadata_dict)
    with open(metadata_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}.")

def main():
    parser = argparse.ArgumentParser(description="Index multiple PDF files.")
    parser.add_argument("--pdf_directory", type=str, default="data/pdfs", help="Path to folder with PDFs.")
    parser.add_argument("--index_path", type=str, default="faiss_index.index", help="Path to FAISS index.")
    parser.add_argument("--metadata_path", type=str, default="chunk_metadata.json", help="Path to metadata JSON.")
    args = parser.parse_args()

    index_multiple_docs(
        pdf_directory=args.pdf_directory,
        index_path=args.index_path,
        metadata_path=args.metadata_path
    )

if __name__ == "__main__":
    main()

