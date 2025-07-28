"""
Docling-based Document Processing System for Challenge 1b

This script processes PDF documents using Docling with optimized settings.
It is designed to run seamlessly both locally and inside the provided Docker container.

Key Features:
- Fast PDF processing with 8 threads
- OCR disabled for speed
- Image placeholder mode to ignore images
- CPU-only processing
- Semantic search and ranking for document sections

Author: Hitesh & Gemini
Version: 5.2 (Final Portable Version)
"""

import os
import sys
import json
import argparse
import datetime
import gc
import re
import warnings
from typing import List, Dict
from collections import Counter, defaultdict
import numpy as np

# LlamaIndex imports for document processing and embeddings
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

# Docling imports for custom configuration
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

# Suppress a harmless warning from PyTorch on CPU-only systems
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but no accelerator is found")

# --- Configuration ---
LOCAL_EMBEDDING_MODEL_PATH = "./all-MiniLM-L6-v2-model"

# --- DYNAMIC PATH CONFIGURATION ---
# This block makes the script work both locally and inside Docker.
DOCKER_DOCLING_PATH = "/app/custom_docling_models"
LOCAL_DOCLING_PATH = "./custom_docling_models"

if os.path.isdir(DOCKER_DOCLING_PATH):
    DOCLING_ARTIFACTS_PATH = DOCKER_DOCLING_PATH
    print(f"Running in Docker mode. Using Docling models from: {DOCLING_ARTIFACTS_PATH}")
else:
    DOCLING_ARTIFACTS_PATH = LOCAL_DOCLING_PATH
    print(f"Running in Local mode. Using Docling models from: {DOCLING_ARTIFACTS_PATH}")


def initialize_services():
    """
    Initializes the HuggingFace embedding model for semantic similarity search.
    """
    print("--- Initializing Embedding Service ---")
    
    if not os.path.isdir(LOCAL_EMBEDDING_MODEL_PATH):
        print(f"FATAL ERROR: The embedding model directory was not found at '{LOCAL_EMBEDDING_MODEL_PATH}'", file=sys.stderr)
        return False
    
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=LOCAL_EMBEDDING_MODEL_PATH,
            device="cpu"
        )
        Settings.transformations = [MarkdownNodeParser()]
        print("Embedding Service initialized successfully.")
        return True
    except Exception as e:
        print(f"\nAn error occurred during service initialization: {e}", file=sys.stderr)
        return False

def load_documents(pdf_directory: str, document_infos: List[Dict]) -> List[Document]:
    """
    Loads PDF documents using Docling with optimized settings.
    """
    print("\n--- Loading Documents with Optimized Docling Settings ---")
    
    if not os.path.isdir(DOCLING_ARTIFACTS_PATH):
        print(f"FATAL ERROR: Docling models not found at '{DOCLING_ARTIFACTS_PATH}'", file=sys.stderr)
        print("Please ensure the models are available at this path.", file=sys.stderr)
        return None
    
    try:
        required_paths = [os.path.join(pdf_directory, doc['filename']) for doc in document_infos]
        existing_paths = [p for p in required_paths if os.path.exists(p)]
        
        if not existing_paths:
            print("Error: No specified documents found.", file=sys.stderr)
            return None
        
        print(f"Found {len(existing_paths)} document(s) to process.")

        # Configure accelerator options to force CPU execution and set thread count.
        accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CPU, # Matches --device cpu
            num_threads=8                  # Matches --num-threads 8
        )

        # Create PDF pipeline options with all performance settings.
        pipeline_options = PdfPipelineOptions(
            artifacts_path=DOCLING_ARTIFACTS_PATH,
            run_force_ocr=False,
            do_table_structure=True,
            accelerator_options=accelerator_options
        )
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
        custom_doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        docling_reader = DoclingReader(
            doc_converter=custom_doc_converter,
            md_export_kwargs={"image_placeholder": ""}
        )
        
        reader = SimpleDirectoryReader(
            input_files=existing_paths,
            file_extractor={".pdf": docling_reader}
        )
        
        print("Reading files with Docling (Fast Mode, 8 Threads, CPU Only, No OCR)...")
        documents = reader.load_data(show_progress=True)
        print("Finished reading files.")
        return documents
        
    except Exception as e:
        print(f"An error occurred during document loading: {e}", file=sys.stderr)
        return None

def process_collection(collection_path: str):
    """
    Main processing function that implements the "Retrieve and Re-Rank" strategy.
    """
    print(f"\n======= Processing Collection: {collection_path} =======")
    
    input_json_path = os.path.join(collection_path, 'challenge1b_input.json')
    pdf_dir_path = os.path.join(collection_path, 'PDFs')
    output_json_path = os.path.join(collection_path, 'challenge1b_output.json')

    print(f"Reading input from {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    persona = input_data['persona']['role']
    job_to_be_done = input_data['job_to_be_done']['task']
    documents_info = input_data['documents']
    
    documents = load_documents(pdf_dir_path, documents_info)
    if not documents:
        print("Halting processing due to document loading failure.", file=sys.stderr)
        return
    
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(documents)
    if not nodes:
        print("Halting processing as no nodes were created from documents.", file=sys.stderr)
        return
    print(f"Successfully created {len(nodes)} nodes from documents.")

    print("Building vector index...")
    index = VectorStoreIndex(nodes)

    print("Retrieving relevant nodes from index...")
    query = f"As a {persona}, I need to {job_to_be_done}. I am looking for information on cities, activities, nightlife, entertainment, beaches, food, and budget-friendly options."
    
    retriever = index.as_retriever(similarity_top_k=30)
    retrieved_nodes_with_scores = retriever.retrieve(query)

    print("Analyzing retrieved nodes to generate output...")
    
    section_scores = defaultdict(float)
    section_details = {}
    
    for node_with_score in retrieved_nodes_with_scores:
        node = node_with_score.node
        score = node_with_score.score
        
        header_from_meta = node.metadata.get('Header_3') or node.metadata.get('Header_2') or node.metadata.get('Header_1')
        if header_from_meta:
            header = header_from_meta
        else:
            match = re.search(r"^#+\s+(.*)", node.get_content(), re.MULTILINE)
            header = match.group(1).strip() if match else "Introduction"
        
        doc_name = node.metadata.get('file_name')
        page_num = node.metadata.get('page_label', 1)
        
        section_key = (doc_name, header)
        section_scores[section_key] += score
        
        if section_key not in section_details:
            section_details[section_key] = {'page_number': page_num}

    ranked_sections_by_score = sorted(section_scores.items(), key=lambda item: item[1], reverse=True)
    
    extracted_sections = []
    for rank, ((doc_name, header), score) in enumerate(ranked_sections_by_score[:5], 1):
        extracted_sections.append({
            "document": doc_name,
            "section_title": header,
            "importance_rank": rank,
            "page_number": section_details.get((doc_name, header), {}).get('page_number', 1)
        })

    subsection_analysis = []
    for node_with_score in retrieved_nodes_with_scores[:5]:
        node = node_with_score.node
        
        raw_text = node.get_content().strip()
        text_no_headers = re.sub(r'^#+\s+.*$', '', raw_text, flags=re.MULTILINE)
        cleaned_text = ' '.join(text_no_headers.split()).strip()

        subsection_analysis.append({
            "document": node.metadata.get('file_name'),
            "refined_text": cleaned_text,
            "page_number": node.metadata.get('page_label', 1)
        })

    output_data = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in documents_info],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    print(f"\nWriting final output to {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"======= Finished Processing Collection: {collection_path} =======")
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a document collection for Challenge 1b.")
    parser.add_argument("collection_path", type=str, help="The path to the collection directory (e.g., 'Challenge_1b/Collection 1/').")
    args = parser.parse_args()

    if not os.path.isdir(args.collection_path):
        print(f"Error: Directory not found at {args.collection_path}", file=sys.stderr)
        sys.exit(1)
            
    if not initialize_services():
        print("Could not initialize AI services. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    process_collection(args.collection_path)
