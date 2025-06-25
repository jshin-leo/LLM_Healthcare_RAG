# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/retriever.py
#   - Author: Jihoon Shin
#   - Date: June 25, 2025
#   - Purpose: semantic search by encoding a query, 
#              retrieving the most relevant text chunks from a FAISS vector index, 
#              and returning their associated metadata.
# ---------------------------------------------------------------------

import argparse
import json
import numpy as np
import faiss
import llm_pipeline.utils as utils

# Loads a JSON metadata file that maps vector indices to text chunks and their metadata.
def load_metadata(metadata_file):
    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)

def search_faiss(query, index, model, metadata, top_k=3):
    # Encode the query
    query_vec = model.encode([query], convert_to_numpy=True)
    
    # Searches the FAISS index for top-k similar vectors
    # D = distances, I = indices
    D, I = index.search(query_vec, top_k * 2)  # Search more to allow deduplication

    # Retrieve matched metadata and returns ranked results with distance scores.
    results = []
    seen_chunks = set()

    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        entry = metadata[idx]
        chunk_text = utils.fully_normalize_text(entry["chunk"]).lower()

        if chunk_text not in seen_chunks:
            seen_chunks.add(chunk_text)
            entry["score"] = float(dist)
            results.append(entry)

        if len(results) >= top_k:
            break

    return results

# Module Method 
def retrieve(index_file, metadata_file, query: str, top_k: int = 3):
    index = faiss.read_index(index_file)
    metadata = load_metadata(metadata_file)
    model = utils.load_embedding_model()
    return search_faiss(query, index, model, metadata, top_k)

# [testing purpose] Command-Line Interface 
def interactive_cli(index_file, metadata_file, top_k=3):
    parser = argparse.ArgumentParser(description="Search vector store using a text query.")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=top_k, help="Number of top results to return")
    args = parser.parse_args()

    print(f"\nLoading FAISS index and metadata...")
    index = faiss.read_index(index_file)
    metadata = load_metadata(metadata_file)
    model = utils.load_embedding_model()

    print(f"Searching for: \"{args.query}\" ...")
    results = search_faiss(args.query, index, model, metadata, top_k=args.top_k)

    print(f"\nTop {args.top_k} results:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Title: {r.get('title', '-')}\n    URL: {r.get('url', '-')}\n    Score (distance): {r['score']:.4f}\n    Chunk: {r['chunk'][:300]}...\n")