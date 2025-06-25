# LLM for Healthcare (RAG)
This project builds a local Retrieval-Augmented Generation (RAG) pipeline using healthcare video and web content. It extracts transcripts, filters sensitive content, chunks them, indexes with FAISS, and enables local LLM-based question answering.
![AI_Chatbot](https://github.com/user-attachments/assets/a6e1d21f-ec33-49cf-bfcb-09849e23fbc2)

## Features
- YouTube and website crawling
- Whisper transcription
- Diagnosis & prescription filtering
- Sentence-level chunking using SentenceTransformer
- FAISS vector index
- Local RAG querying using Mistral 7B model

## Folder Structure
```
LLM_healthcare_rag/
├── main.py # Entry point script to run the full RAG pipeline
├── data/ # Transcripts, chunks, and vector store files
├── llm_pipeline/ # Core pipeline components
│ ├── chunker_youtube.py # YouTube crawling & chunking pipeline
│ ├── chunker_website.py # Web content chunking
│ ├── retriever.py # FAISS semantic search
│ ├── rag_pipeline.py # Full RAG generation pipeline
│ ├── utils.py # Shared constants and helper methods
│ ├── vector_store_builder.py # FAISS index builder
│ ├── youtube_scraper.py # YouTube link crawler
```
## Usage
### 1. Install dependencies
```
pip install -r requirements.txt
```
### 2. Run pipelines
#### Step-by-step
```
python main.py youtube Y        # Crawl and chunk YouTube
python main.py website          # Chunk web articles
python main.py index            # Build FAISS index
python main.py retrieve --query "What are signs of dementia?"
python main.py rag --query "How to manage memory loss?" --top_k 5
```
