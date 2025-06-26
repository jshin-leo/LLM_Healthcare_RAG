# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/chunker_youtube.py
#   - Author: Jihoon Shin
#   - Date: June 5th, 2025
#   - What: YouTube Transcript Chunking
#   - How: 
#       For each video:
#          - Download audio using yt-dlp
#          - Transcribe speech using OpenAI Whisper
#          - Filter irrelevant sentences using spacy-based similarity
#          - Segment the transcript into coherent chunks using SentenceTransformer
#            (Used Sentence based chunking)
# ---------------------------------------------------------------------

import os, re, json, subprocess, torch
os.environ["PATH"] = os.path.expanduser("~/ffmpeg-7.0.2-amd64-static") + ":" + os.environ["PATH"]
from datetime import datetime
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import whisper, spacy
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from llm_pipeline import utils

# === Setup ===
nlp = utils.NLP
model_lm = utils.load_embedding_model()

# LLM for filtering and summarization
tokenizer, model = utils.load_mistral_model()
mistral_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Audio and Metadata ===
def get_video_metadata(url):
    try:
        result = subprocess.run(["yt-dlp", "--dump-json", url], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] yt-dlp metadata: {e}")
        return {}

def download_audio(url, output_path):
    print(f"Downloading audio from: {url}")
    command = [
        "yt-dlp", "--cookies", "cookies.txt",
        "--ffmpeg-location", os.path.expanduser("~/ffmpeg-7.0.2-amd64-static"),
        "-f", "bestaudio", "-x", "--audio-format", "mp3",
        "-o", output_path, url
    ]
    subprocess.run(command, check=True)
    print(f"Audio saved to {output_path}")
    return output_path

def transcribe_audio(audio_path, transcript_path=None):
    model = whisper.load_model(utils.WHISPER_MODEL)
    result = model.transcribe(audio_path, language='en')

    if transcript_path:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Full transcript saved to {transcript_path}")

    return result["text"]

# === Sentence Filtering ===
def is_high_risk(sentence):
    HIGH_RISK_KEYWORDS = [
        # Diagnosis indicators
        r"\bdiagnosed with\b",
        r"\bsymptom of\b",
        r"\bindicates (you )?have\b",
        r"\bsuggests (you )?have\b",
        r"\bconsistent with (a )?diagnosis\b",
        r"\bthis means you have\b",
        r"\bmay be suffering from\b",
        r"\bshows signs of\b",
        r"\b(?:you|they|he|she) (may|might|likely|probably)? (have|be experiencing)\b.*",
        
        # Cause and disease linkage
        r"\bcaused by\b",
        r"\bdue to (a )?(condition|disease|illness|disorder)\b",

        # Prescription or recommendation of treatment
        r"\byou should (take|use|try)\b.*",
        r"\btake (your )?(medication|medicine|pill|tablet|drug|dose)\b",
        r"\bstart taking\b.*",
        r"\byou need to take\b.*",
        r"\bprescribed\b",
        r"\bis prescribed for\b",
        r"\byou are prescribed\b",
        r"\bdoctor (may|might|will)? prescribe\b.*",
        r"\byou must\b.*(medication|treatment|drug|therapy)",
        r"\bask your doctor\b",
        r"\btalk to your doctor about\b",
        
        # Drug/treatment references
        r"\bmedication\b",
        r"\bpharmaceutical\b",
        r"\bantibiotic\b",
        r"\bpainkiller\b",
        r"\binsulin\b",
        r"\bchemo(therapy)?\b",
        r"\bradiation therapy\b",
        r"\btreatment plan\b",
        r"\bsurgery\b",
    ]
    return any(re.search(pat, sentence.lower()) for pat in HIGH_RISK_KEYWORDS)

# === diagnosis/prescription - Sentence Classification via LLM ===
def filter_diagnosis_with_mistral(sentences, log_path=None):
    filtered, removed = [], []
    for s in sentences:
        if is_high_risk(s):
            prompt = (
                "You are reviewing a sentence from a healthcare training transcript.\n"
                "Label it as:\n"
                "- Y (Yes): if it gives a direct diagnosis or prescribes a medicine or treatment, or including any medicine name.\n"
                "- N (No): if it is general advice, caregiving instruction, or educational but not a diagnosis or prescription.\n\n"
                "Only answer in Y or N. \n"
                f"Sentence: \"{s}\"\nAnswer:"
            )
            try:
                response = mistral_pipe(prompt, max_new_tokens=10, do_sample=False, return_full_text=False)
                if response[0]['generated_text'].strip().startswith("Y"):
                    removed.append(s)
                    continue
            except Exception:
                response = "N"  # Assume safe if LLM fails
                pass
        filtered.append(s)

    if removed and log_path:
        with open(log_path, "w") as f: f.write("\n".join(removed))
    return filtered

# === Summarization ===
def summarize_transcript(text):
    prompt = (
        "You are a language model helping summarize healthcare training transcripts. "
        "Divide the transcript into meaningful segments based on topic shifts. For each segment, write one concise sentence that summarizes the main idea. "
        "Only use content found in the transcript. Do NOT add new information, tips, or labels like 'Summary:'. "
        "Return only the list of summary sentences. Maintain the original order of topics. " 
        "If any segment includes storytelling or metaphor, summarize the **main point** only. DO NOT preserve stylistic language. Your task is to **extract, not reproduce**. \n\n"
        "Transcript:\n\n" + text.strip()
    )
    response = mistral_pipe(prompt, max_new_tokens=2048, do_sample=False, return_full_text=False)
    cleaned = response[0]["generated_text"].strip()
    
    # Strip repeated prompt if echoed back
    if "Transcript:" in cleaned:
        cleaned = cleaned.split("Transcript:")[-1].strip()

    if "Summary:" in cleaned:
        cleaned = cleaned.split("Summary:")[-1].strip()

    return response[0]["generated_text"].strip()

# === Filter irrelevant sentences === 
def filter_blocks_by_similarity(raw_blocks, reference_text, log_prefix=None, threshold=0.60):
    reference_doc = nlp(reference_text)
    filtered_blocks, removed_blocks = [], []

    for block in raw_blocks:
        block_doc = nlp(block)
        similarity = block_doc.similarity(reference_doc)

        if similarity >= threshold:
            filtered_blocks.append(block)
        else:
            removed_blocks.append((block, similarity))

    if removed_blocks and log_prefix:
        full_path = f"{log_prefix}_remove_log.txt"
        with open(full_path, "w", encoding="utf-8") as f:
            for text, sim in removed_blocks:
                f.write(f"[SIMILARITY: {sim:.4f}]\n{text}\n\n{'='*40}\n\n")
        print(f"Filtered {len(removed_blocks)} blocks. Logged to {full_path}.")
    elif not removed_blocks:
        print("No blocks were filtered out based on similarity threshold.")

    return filtered_blocks

# === Chunking ===
def chunk_with_similarity(blocks, min_chars=100, max_chars=200, threshold=0.75):
    embeddings = model_lm.encode(blocks, convert_to_tensor=True)
    chunks, current = [], blocks[0]

    for i in range(1, len(blocks)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
        if len(current) + len(blocks[i]) + 1 <= max_chars and (sim >= threshold or len(current) < min_chars):
            current += " " + blocks[i]
        else:
            chunks.append(current.strip())
            current = blocks[i]

    if len(current) >= min_chars:
        chunks.append(current.strip())
    return chunks

# === RAG Formatting ===
def save_chunks_jsonl(chunks, title, url, output_path, tags=None, desc=None):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({
                "id": f"{utils.sanitize_filename(title)}_chunk_{str(i+1).zfill(3)}",
                "chunk": chunk,
                "title": title,
                "url": url,
                "source": "YouTube",
                "tags": tags or [],
                "description": desc or ""
            }) + "\n")

def append_chunks_to_shared_jsonl(chunks, title, url, output_file, tags=None, description=None):
    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, chunk in enumerate(chunks, 1):
            record = {
                "id": f"{utils.sanitize_filename(title)}_chunk_{str(i).zfill(3)}",
                "chunk": chunk,
                "title": title.replace("_", " "),
                "url": url,
                "source": "YouTube", 
                "tags": tags or [],
                "description": description or ""
            }
            out_f.write(json.dumps(record) + "\n")

# === Main Callable Pipeline ===
def process_youtube_video(video_url, output_dir="/data/transcript", min_chars=100, max_chars=200, 
                            shared_output_path="combined_youtube_rag.jsonl", seen_chunks_global=None):
    os.makedirs(output_dir, exist_ok=True)
    audio_path = None
    transcript_path = None

    try:
        metadata = get_video_metadata(video_url)
        title = metadata.get("title", "untitled")
        base_name = f"{utils.sanitize_filename(title)[:100]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        base_path = os.path.join(output_dir, base_name)

        # Download audio
        audio_path = f"{base_path}.mp3"
        download_audio(video_url, audio_path)

        # Transcribe audio
        transcript_path = f"{base_path}_transcript.txt"
        full_text = transcribe_audio(audio_path, transcript_path)

        # Sentence filtering
        sentences = [s.text.strip() for s in nlp(full_text).sents if s.text.strip()]
        filtered = filter_diagnosis_with_mistral(sentences, log_path=base_path + "_removed.txt")

        # Summarize and filter
        summary = summarize_transcript("\n".join(filtered))
        cleaned = filter_blocks_by_similarity(
            raw_blocks=filtered,
            reference_text=summary,
            log_prefix=base_path,
            threshold=0.60
        )

        # Chunk + deduplicate
        chunks = chunk_with_similarity(cleaned, min_chars, max_chars)
        chunks = [utils.deduplicate_within_chunk(c) for c in chunks]
        
        # Global deduplication
        if seen_chunks_global is not None:
            chunks = utils.global_deduplicate_chunks(chunks, seen_chunks_global)

        # Save chunks
        append_chunks_to_shared_jsonl(
            chunks, 
            title, 
            video_url, 
            shared_output_path, 
            tags=metadata.get("tags"), 
            description=metadata.get("description")
        )

        # Clean up audio
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Deleted audio file: {audio_path}")
        
        print(f"Processed: {video_url}")
        print(f"Chunks saved to shared file: {shared_output_path}")
        return shared_output_path

    except Exception as e:
        print(f"[ERROR] Failed to process {video_url}: {e}")

        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Deleted audio file: {audio_path}")

        if transcript_path and os.path.exists(transcript_path):
            os.remove(transcript_path)
            print(f"Deleted transcript file: {transcript_path}")

        return None
