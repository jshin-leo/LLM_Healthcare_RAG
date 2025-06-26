# LLM for Healthcare (RAG)
This project implements a local Retrieval-Augmented Generation (RAG) pipeline tailored for healthcare content, with a focus on Alzheimer’s disease, related dementias (ADRD), and caregiving. It extracts transcripts from YouTube videos and web articles, filters sensitive medical statements, chunks the content, builds a FAISS vector store, and enables local question answering using a fine-tuned Mistral 7B model.
  
**Official project**: https://github.com/LeonG19/llm-healthcare  
This work is currently in progress as part of the **Security and Intelligence Lab** at **UTEP**, under the supervision of Dr. Piplai.  

## My Contribution 
This GitHub repository serves as a personal implementation to deepen my understanding of Retrieval-Augmented Generation (RAG) pipelines. I independently developed most components as a standalone project, including YouTube crawling, transcription and chunking, content filtering, FAISS-based semantic search, the RAG pipeline, and overall system integration.  
**Note**: Web crawling for data collection was done by Arman, and Q&A dataset collection was contributed by Emilia.
![AI_Chatbot](https://github.com/user-attachments/assets/6e0e811c-1375-49d8-a523-77981b051aa5)

## Features
- YouTube crawling
- Whisper-based audio transcription
- Diagnosis & prescription filtering
- Sentence-level chunking with semantic similarity
- FAISS vector index
- Local RAG querying using Mistral 7B model
- Fine tune with Q&A datasets (in progress)

## Folder Structure
```
LLM_healthcare_rag/
├── main.py # Entry point script to run the full RAG pipeline
├── data/ # Transcripts, chunks, and vector store files
├── llm_pipeline/ # Core pipeline components
│ ├── chunker_youtube.py # YouTube crawling & chunking  
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
python main.py youtube [Y/N/playlist]       # Run YouTube chunking pipeline (Y: re-crawl, N: use cached, playlist: extract from playlists)
python main.py website                      # Run website content chunking
python main.py index                        # Build FAISS vector index
python main.py retrieve --query "your question"                  # Semantic search
python main.py rag --query "your question" [--top_k N] [--max_tokens N]  # Full RAG pipeline with Mistral
```

### ⚠️Responsible Use
This tool is intended for educational and research purposes only. Use ethically and responsibly. Please ensure that you:
- Use publicly available, non-restricted content.
- Respect copyright and privacy laws.
- Do not use this tool to scrape or redistribute copyrighted, sensitive, or private content without proper authorization.

### Attribution
- Icons used in the diagram are sourced from [Flaticon](https://www.flaticon.com/)
