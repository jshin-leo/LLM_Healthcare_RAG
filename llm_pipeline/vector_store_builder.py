# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/vector_store_builder.py
#   - Author: Jihoon Shin
#   - Date: June 25, 2025
#   - Purpose: Create FAISS index from all jsonl chunk files in a folder
# ---------------------------------------------------------------------

import os
import json
import numpy as np
import faiss
from tqdm import tqdm
import llm_pipeline.utils as utils

def load_unique_chunks(chunks_dir):
    """
    Load all JSONL chunk files, remove duplicate 'chunk' texts,
    and return a list of full metadata entries (including title, URL, etc.).
    """
    seen_chunks = set()
    unique_metadata = []

    for filename in os.listdir(chunks_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(chunks_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        chunk_text = entry.get("chunk", "").strip()

                        if chunk_text and chunk_text not in seen_chunks:
                            seen_chunks.add(chunk_text)
                            unique_metadata.append(entry)

                    except json.JSONDecodeError:
                        print(f"[Warning] Skipped malformed line in {filename}")

    print(f"[INFO] Loaded {len(unique_metadata)} unique chunks from {chunks_dir}")
    return unique_metadata


def build_vector_store(chunks_dir, index_file, metadata_file):
    # Load unique chunks and full metadata
    metadata = load_unique_chunks(chunks_dir)
    texts = [m["chunk"] for m in metadata]

    print(f"Encoding {len(texts)} chunks using {utils.EMBEDDING_MODEL_NAME}...")
    model = utils.load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    print("Adding vectors to FAISS index...")
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_file), exist_ok=True)

    print(f"Saving FAISS index to: {index_file}")
    faiss.write_index(index, index_file)

    print(f"Saving metadata to: {metadata_file}")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done! Stored {len(embeddings)} vectors in FAISS index.")
