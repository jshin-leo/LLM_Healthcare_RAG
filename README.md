# LLM for Healthcare (RAG)
This project implements a local Retrieval-Augmented Generation (RAG) pipeline tailored for healthcare content. It extracts transcripts from YouTube videos and web articles, filters sensitive medical statements, chunks the content, builds a FAISS vector store, and enables local question answering using a fine-tuned Mistral 7B model.
![AI_Chatbot](https://github.com/user-attachments/assets/6e0e811c-1375-49d8-a523-77981b051aa5)

## Features
- YouTube crawling
- Whisper-based audio transcription
- Diagnosis & prescription filtering
- Sentence-level chunking with semantic similarity
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

### ⚠️Responsible Use
This tool is intended for educational and research purposes only. Use ethically and responsibly. Please ensure that you:
- Use publicly available, non-restricted content.
- Respect copyright and privacy laws.
- Do not use this tool to scrape or redistribute copyrighted, sensitive, or private content without proper authorization.

### Attribution
- Icons used in the diagram are sourced from [Flaticon](https://www.flaticon.com/)
