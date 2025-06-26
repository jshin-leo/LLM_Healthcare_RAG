# ---------------------------------------------------------------------
#   Project: LLM for HealthCare 
#
#   - Title: main.py
#   - Author: Jihoon Shin
#   - GitHub: https://github.com/jshin-leo/LLM_Healthcare_RAG
#   - Date: June 6th, 2025
#   - Purpose: Full RAG pipeline 
#   
#   - Description:
#       This script serves as the main entry point for the "LLM for Healthcare" project,
#       which builds a local Retrieval-Augmented Generation (RAG) system focused on 
#       Alzheimer’s disease, related dementias (ADRD), and caregiving.
#
#       The pipeline supports:
#       - Crawling YouTube 
#       - Transcribing and chunking textual content
#       - Building a FAISS vector store for semantic search
#       - Performing question answering via a local LLM (Mistral 7B)
#       - Fine tuning Mistral 7B with Q&A datasets [In Progress] 
#
#   Usage:
#     Run individual stages (e.g., youtube, website, index) or the full RAG pipeline.
# ---------------------------------------------------------------------
import os
# Silence tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

import sys
import argparse
import llm_pipeline.utils as utils

# ===== Website and YouTube Processing =====
from llm_pipeline import youtube_scraper
from llm_pipeline import chunker_youtube
from llm_pipeline import chunker_website

# ===== FAISS Vector Store Builder =====
from llm_pipeline.vector_store_builder import build_vector_store

# ===== FAISS Vector Retriever =====
from llm_pipeline.retriever import retrieve, interactive_cli

# ===== RAG with local LLM =====
from llm_pipeline.rag_pipeline import run_rag_pipeline

# ---------------------------------------------------------------------
# ===== SETTINGS =====
# Folders from utils.py
WEBSITES_FOLDER = utils.WEBSITES_FOLDER
YOUTUBE_OUTPUT_DIR = utils.YOUTUBE_OUTPUT_DIR
CHUNKS_DIR = utils.CHUNKS_DIR

# Files from utils.py
YOUTUBE_LINKS_FILE = utils.YOUTUBE_LINKS_FILE
YOUTUBE_LINKS_EXTRACTED_FROM_PLAYLIST = utils.YOUTUBE_LINKS_EXTRACTED_FROM_PLAYLIST
YOUTUBE_PLAYLIST = utils.YOUTUBE_PLAYLIST
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
    
    seen_chunks_global = set()  # shared (for global deduplication)
    count = 0   
    for link in youtube_links:
        count += 1
        print(f"----- [{count}/{len(youtube_links)}] Processing: {link} ")
        chunker_youtube.process_youtube_video(
            link,
            output_dir=YOUTUBE_OUTPUT_DIR,
            shared_output_path=COMBINED_YOUTUBE_OUTPUT_FILE,
            seen_chunks_global=seen_chunks_global
        )

# ===== YouTube: Extract links from playlists + Chunking Process =====
def run_youtube_playlist():
    print("\n--- Running YouTube Playlist Pipeline ---")

    # Extract links from playlist
    youtube_scraper.extract_playlist_links(YOUTUBE_PLAYLIST, YOUTUBE_LINKS_EXTRACTED_FROM_PLAYLIST)

    with open(YOUTUBE_LINKS_EXTRACTED_FROM_PLAYLIST, "r", encoding="utf-8") as f:
        youtube_links = [line.strip() for line in f if line.strip()]

    print(f"Found {len(youtube_links)} YouTube link(s) to process.")

    seen_chunks_global = set()  # shared (for global deduplication)
    count = 0   
    for link in youtube_links:
        count += 1
        print(f"----- [{count}/{len(youtube_links)}] Processing: {link} ")
        chunker_youtube.process_youtube_video(
            link,
            output_dir=YOUTUBE_OUTPUT_DIR,
            shared_output_path=COMBINED_YOUTUBE_OUTPUT_FILE,
            seen_chunks_global=seen_chunks_global
        )

# ===== Website: Chunking Processing =====
def run_website_pipeline():
    print("\n--- Running Website Pipeline ---")

    seen_chunks_global = set()  # shared (for global deduplication)

    for filename in os.listdir(WEBSITES_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(WEBSITES_FOLDER, filename)
            chunker_website.process_content_file(
                filepath,
                combined_output=COMBINED_WEBSITE_OUTPUT_FILE,
                seen_chunks_global=seen_chunks_global
            )

# ===== Retriever top k =====
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

def print_usage():
    print("Usage:")
    print(f"  python main.py youtube [Y/N/playlist]   # Run YouTube chunking pipeline (Y: re-crawl, N: use cached, playlist: extract from {YOUTUBE_PLAYLIST})")
    print("  python main.py website # Run website content chunking")
    print("  python main.py index   # Build FAISS vector index")
    print("  python main.py retrieve --query 'your question'  # semantic search")
    print("  python main.py rag --query 'your question' [--top_k N] [--max_tokens N]  # Full RAG pipeline with Mistral")


# ===== MAIN ENTRY =====
if __name__ == "__main__":
    print("===== LLM for Healthcare Pipeline =====")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "youtube":    # Run YouTube pipeline
            if len(sys.argv) > 2:
                if sys.argv[2].lower() == 'playlist':   # If you are using manually selected YouTube Playlists.
                    run_youtube_playlist()
                else: 
                    if sys.argv[2].lower() == 'y':  # If crawling needed.
                        do_crawl = True
                    else:
                        do_crawl = False
                    run_youtube_pipeline(do_crawl) 
            else:
                print("Unknown command.")
                print_usage()

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
            print_usage()
    
    else:
        print_usage()

    print("\n===== Done. =====")
