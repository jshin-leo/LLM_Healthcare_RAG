# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/rag_pipeline.py
#   - Author: Jihoon Shin
#   - Date: June 25, 2025
#   - Purpose: Supports querying local Mistral model with retrieved chunks.
# ---------------------------------------------------------------------

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_pipeline.retriever import retrieve
import llm_pipeline.utils as utils

def build_prompt(query, context_chunks):
    context_text = "\n\n".join([f"- {c['chunk']}" for c in context_chunks])
    return f"""You are a helpful assistant answering healthcare-related questions using the context provided.

Context:
{context_text}

Question: {query}
Answer:"""


def run_rag_pipeline(query, index_file, metadata_file, top_k=3, max_tokens=512):
    print(f"Retrieving top {top_k} chunks for query: {query}")
    context_chunks = retrieve(
        index_file=index_file,
        metadata_file=metadata_file,
        query=query,
        top_k=top_k
    )

    print(f"Building prompt...")
    prompt = build_prompt(query, context_chunks)

    tokenizer, model = utils.load_mistral_model()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nAnswer:\n")

    # ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/rag_pipeline.py
#   - Author: Jihoon Shin
#   - Date: June 25, 2025
#   - Purpose: Supports querying local Mistral model with retrieved chunks.
# ---------------------------------------------------------------------

import torch
from llm_pipeline.retriever import retrieve
import llm_pipeline.utils as utils

def build_prompt(query, context_chunks):
    context_text = "\n\n".join([f"- {c['chunk']}" for c in context_chunks])
    prompt = f"""You are a helpful assistant answering healthcare-related questions using the context provided.

Context:
{context_text}

Question: {query}
Answer:"""
    return prompt


def run_rag_pipeline(query, index_file, metadata_file, top_k=3, max_tokens=512):
    print(f"Retrieving top {top_k} chunks for query: {query}")
    context_chunks = retrieve(
        index_file=index_file,
        metadata_file=metadata_file,
        query=query,
        top_k=top_k
    )

    print(f"Building prompt...")
    prompt = build_prompt(query, context_chunks)
    print(f"Prompt: {prompt}")

    tokenizer, model = utils.load_mistral_model()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    final_response = response.replace(prompt, "").strip()
    
    print("\nAnswer:\n")    
    print(final_response)

    return final_response