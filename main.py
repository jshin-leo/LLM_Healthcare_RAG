# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: main.py
#   - Author: Jihoon Shin
#   - Date: June 6th, 2025
#   - Purpose: Full RAG pipeline 
# ---------------------------------------------------------------------
import os
# Silence tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

import sys
import argparse
import llm_pipeline.utils as utils

# === Website and YouTube Processing ===
from llm_pipeline import youtube_scraper
from llm_pipeline import chunker_youtube
from llm_pipeline import chunker_website

# === FAISS Vector Store Builder ===
from llm_pipeline.vector_store_builder import build_vector_store

# === FAISS Vector Retriever ===
from llm_pipeline.retriever import retrieve, interactive_cli

# === RAG with local LLM ===
from llm_pipeline.rag_pipeline import run_rag_pipeline

# ---------------------------------------------------------------------
# === SETTINGS ===
# Folders from utils.py
WEBSITES_FOLDER = utils.WEBSITES_FOLDER
YOUTUBE_OUTPUT_DIR = utils.YOUTUBE_OUTPUT_DIR
CHUNKS_DIR = utils.CHUNKS_DIR

# Files from utils.py
YOUTUBE_LINKS_FILE = utils.YOUTUBE_LINKS_FILE
COMBINED_YOUTUBE_OUTPUT_FILE = utils.COMBINED_YOUTUBE_OUTPUT_FILE
COMBINED_WEBSITE_OUTPUT_FILE = utils.COMBINED_WEBSITE_OUTPUT_FILE
INDEX_FILE = utils.INDEX_FILE
METADATA_FILE = utils.METADATA_FILE

# ---------------------------------------------------------------------

# === YouTube: Crawl + Chunking Process ===
def run_youtube_pipeline(do_crawl):
    print("\n--- Running YouTube Pipeline ---")

    if do_crawl:    # if you want to re-crawl (Y)
        youtube_scraper.crawl_youtube_links_and_save(utils.TARGETS, YOUTUBE_LINKS_FILE)

    if not os.path.exists(YOUTUBE_LINKS_FILE):
        print(f"[ERROR] '{YOUTUBE_LINKS_FILE}' not found.")
        return

    with open(YOUTUBE_LINKS_FILE, "r", encoding="utf-8") as f:
        youtube_links = [line.strip() for line in f if line.strip()]

    print(f"Found {len(youtube_links)} YouTube link(s) to process.")

    for link in youtube_links:
        chunker_youtube.process_youtube_video(
            link,
            output_dir=YOUTUBE_OUTPUT_DIR,
            shared_output_path=COMBINED_YOUTUBE_OUTPUT_FILE
        )

# === Website: Chunking Processing ===
def run_website_pipeline():
    print("\n--- Running Website Pipeline ---")

    for filename in os.listdir(WEBSITES_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(WEBSITES_FOLDER, filename)
            chunker_website.process_content_file(
                filepath,
                combined_output=COMBINED_WEBSITE_OUTPUT_FILE
            )

# === Retriever top k ===
def run_retrieve_pipeline(args):
    print(f"\nQuery: \"{args.query}\"")
    results = retrieve(
        index_file=INDEX_FILE,
        metadata_file=METADATA_FILE,
        query=args.query,
        top_k=args.top_k
    )

    print(f"\nTop {args.top_k} results:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Title: {r.get('title', '-')}\n    URL: {r.get('url', '-')}\n    Score (distance): {r['score']:.4f}\n    Chunk: {r['chunk'][:500]}...\n")

# === MAIN ENTRY ===
if __name__ == "__main__":
    print("===== LLM for Healthcare Pipeline =====")
    
    if len(sys.argv) > 1:
        
        if sys.argv[1] == "youtube":    # Run YouTube pipeline
            do_crawl = sys.argv[2].lower() == "y" if len(sys.argv) > 2 else False   # crawling option
            run_youtube_pipeline(do_crawl)

        elif sys.argv[1] == "website":  # Run website content chunking
            run_website_pipeline()

        elif sys.argv[1] == "index":    # Build FAISS vector index 
            build_vector_store(
                chunks_dir=CHUNKS_DIR, 
                index_file=INDEX_FILE,
                metadata_file=METADATA_FILE
            )
        elif sys.argv[1] == "retrieve":  # Perform semantic search using the FAISS index
            parser = argparse.ArgumentParser()
            parser.add_argument("--query", type=str, required=True)
            parser.add_argument("--top_k", type=int, default=3)
            args = parser.parse_args(sys.argv[2:])  # parse from sys.argv[2] onward

            run_retrieve_pipeline(args)

        elif sys.argv[1] == "rag": # Run the full Retrieval-Augmented Generation (RAG) pipeline
            parser = argparse.ArgumentParser()
            parser.add_argument("--query", type=str, required=True)
            parser.add_argument("--top_k", type=int, default=3)
            parser.add_argument("--max_tokens", type=int, default=512)
            args = parser.parse_args(sys.argv[2:])  # parse from sys.argv[2] onward

            run_rag_pipeline(
                query=args.query,
                index_file=INDEX_FILE,
                metadata_file=METADATA_FILE,
                top_k=args.top_k,
                max_tokens = args.max_tokens
            )

        else:
            print("Unknown command.")
            pass
    
    else:
        print("Usage:")
        print("  python main.py youtube [Y/N]   # Run YouTube chunking pipeline (Y: re-crawl, N: use cached)")
        print("  python main.py website # Run website content chunking")
        print("  python main.py index   # Build FAISS vector index")
        print("  python main.py retrieve --query 'your question'  # semantic search")
        print("  python main.py rag --query 'your question' [--top_k N] [--max_tokens N]  # Full RAG pipeline with Mistral")
    
    print("\n===== Done. =====")
