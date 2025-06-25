# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/utils.py
#   - Author: Jihoon Shin
#   - Date: June 6th, 2025
#   - What: Utility methods for the project
# ---------------------------------------------------------------------
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import re, unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import spacy
import torch

# ---------------------------------------------------------------------
# Crawling target websites 
TARGETS = [
    # Example: ("https://nexusipe.org/informing/resource-center/", "/informing/resource-center/")
    # Fill this with actual targets for YouTube scraping
    ("https://nexusipe.org/informing/resource-center/", "/informing/resource-center/")
    ]

# Folders
DATA_DIR = "data"
WEBSITES_FOLDER = "data/websites_input_data"  # Folder with .txt files to process
YOUTUBE_OUTPUT_DIR = "data/transcripts"
CHUNKS_DIR = "data/chunks"
VECTOR_DIR = "data/vector_store"

# Ensure directories exist
os.makedirs(WEBSITES_FOLDER, exist_ok=True)
os.makedirs(YOUTUBE_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Files
YOUTUBE_LINKS_FILE = os.path.join(DATA_DIR, "youtube_links.txt")
COMBINED_YOUTUBE_OUTPUT_FILE = os.path.join(CHUNKS_DIR, "combined_youtube_rag.jsonl")
COMBINED_WEBSITE_OUTPUT_FILE = os.path.join(CHUNKS_DIR, "combined_website_rag.jsonl")

# [FAISS] files for vector index
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.idx")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.json")
# ---------------------------------------------------------------------
# --- Model settings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(model_name)

NLP = spacy.load("en_core_web_md")
WHISPER_MODEL = "base"

# ---------------------------------------------------------------------
# --- LLM settings ---
MODEL_PATH = "PLEASE_USE_YOUR_PATH"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

def load_mistral_model():
    # print(f"Loading Mistral model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    return tokenizer, model

# ---------------------------------------------------------------------
# Define known invisible characters to strip
INVISIBLE_CHARS = ''.join([
    '\ufeff',  # BOM
    '\u200b', '\u200c', '\u200d',  # Zero-width space/non-joiner/joiner
    '\u2060',  # Word joiner
    '\xa0'     # Non-breaking space
])

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def fully_normalize_text(text):
    # Strip leading/trailing invisible characters
    text = text.strip(INVISIBLE_CHARS)
    # Decompose characters (e.g., é → e + ´)
    text = unicodedata.normalize("NFKD", text)

    # Remove diacritics (e.g., accents)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # Replace common special characters with ASCII equivalents
    text = text.translate({
        ord('’'): "'", ord('‘'): "'", ord('“'): '"', ord('”'): '"',
        ord('–'): '-', ord('—'): '-', ord('…'): '...', ord('•'): '-',
        ord('″'): '"', ord('′'): "'", ord('‒'): '-', ord('‐'): '-',
        ord('\u00a0'): ' '
    })

    # Normalize spaces and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text

def deduplicate_within_chunk(chunk):
    lines = chunk.split('\n')
    seen = set()
    deduped = []
    for line in lines:
        norm = fully_normalize_text(line)
        if norm not in seen:
            seen.add(norm)
            deduped.append(line.strip())
    return '\n'.join(deduped).strip()

def extract_url_content(text):
    lines = text.strip().splitlines()
    url, content = "", ""
    content_started = False
    
    for line in lines:
        line = line.strip(INVISIBLE_CHARS)

        if line.startswith("[url]") and not url:
            url = line.replace("[url]", "").strip()
        elif line.startswith("[content]") and not content_started:
            content_started = True
            raw_content = line.replace("[content]", "").strip()
            content += fully_normalize_text(raw_content)
        elif content_started:
            content += "\n" + fully_normalize_text(line)

    return url, content

# ---------------------------------------------------------------------

