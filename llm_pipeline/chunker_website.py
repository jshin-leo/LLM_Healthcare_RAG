# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/chunker_website.py
#   - Author: Jihoon Shin
#   - Date: June 6th, 2025
#   - What: Content Chunking
#   - How: 
#       1. Prepare the list of content in content_links.txt
#       2. In Data folder, transcripts and chunks will be generated. 
# ---------------------------------------------------------------------
import os, re, json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llm_pipeline import utils

# === Setup ===
model_lm = utils.load_embedding_model()

# Mistral LLM (optional for future summary support)
tokenizer, model = utils.load_mistral_model()
mistral_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def transcript_to_raw_blocks(text):
    return [sent.text.strip() for sent in utils.NLP(text).sents if sent.text.strip()]

def similarity_guided_chunking(raw_blocks, min_chars=300, max_chars=500, similarity_threshold=0.75):
    embeddings = model_lm.encode(raw_blocks, convert_to_tensor=True)
    chunks, current = [], raw_blocks[0]
    current_len = len(current)

    for i in range(1, len(raw_blocks)):
        next_block = raw_blocks[i]
        sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
        if current_len + len(next_block) + 1 <= max_chars and (sim >= similarity_threshold or current_len < min_chars):
            current += " " + next_block
            current_len += len(next_block) + 1
        else:
            chunks.append(current.strip())
            current = next_block
            current_len = len(next_block)

    if current_len >= min_chars:
        chunks.append(current.strip())
    return chunks

def save_chunks_to_jsonl(chunks, title, url, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            record = {
                "id": f"{utils.sanitize_filename(title)}_chunk_{str(i).zfill(3)}",
                "chunk": chunk,
                "title": title.replace("_", " "),
                "url": url,
                "source": url,
                "tags": [],
                "description": ""
            }
            f.write(json.dumps(record) + "\n")

# === Main callable function ===
def process_content_file(filepath, combined_output, min_chars=100, max_chars=200, similarity_threshold=0.75):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            full_text = f.read()

        title = utils.fully_normalize_text(os.path.basename(filepath))
        url, content = utils.extract_url_content(full_text)
        content = utils.fully_normalize_text(content)
        raw_blocks = transcript_to_raw_blocks(content)

        chunks = similarity_guided_chunking(raw_blocks, min_chars, max_chars, similarity_threshold)
        chunks = [utils.deduplicate_within_chunk(c) for c in chunks]
        chunks = list({utils.fully_normalize_text(c): c for c in chunks}.values())  # Deduplicate chunks

        save_chunks_to_jsonl(chunks, title, url, combined_output)

        print(f"Processed: {filepath}")
        return combined_output

    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        return None