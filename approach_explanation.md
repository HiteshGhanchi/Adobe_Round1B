# Challenge 1b: Multi-Collection PDF Analysis Solution

## Overview
This project is a high-performance, offline-first document analysis pipeline designed for Challenge 1b of the Adobe India Hackathon 2025. The solution ingests collections of PDF documents and, based on a given persona and task, performs a deep semantic analysis to extract and rank the most relevant sections and text snippets.

The entire system is containerized using Docker, ensuring a reproducible and isolated environment that runs entirely on CPU and adheres to all specified performance and resource constraints.

## Core Methodology: A Docling-First Hierarchical Approach
Our solution is built on a "Docling-First Hierarchical Approach." This strategy prioritizes the high-fidelity structural data provided by Adobe's Docling library over less reliable generative methods, ensuring speed, accuracy, and compliance with all project constraints.

### The pipeline operates as follows:

1) Optimized Document Parsing: We use the DoclingReader to process all PDFs in a collection. The reader is explicitly configured for maximum performance on a CPU-only environment by disabling OCR, ignoring images, setting the table recognition model to its fastest mode, and leveraging 8 threads to match the evaluation hardware.

2) Hierarchical Node Creation: We utilize LlamaIndex's MarkdownNodeParser to intelligently split the documents. This parser understands the structured Markdown generated by Docling, creating text chunks ("nodes") that retain crucial metadata like section headers and page numbers.

3) Semantic Indexing: All structurally-aware nodes are converted into numerical vectors using a lightweight, locally-stored embedding model (all-MiniLM-L6-v2). These vectors are then loaded into an in-memory VectorStoreIndex, creating a unified, instantly searchable knowledge base for the entire collection.

4) Score-Based Section Ranking: To identify the most important sections, we formulate a detailed query from the user's persona and job_to_be_done. A vector retriever finds the top 30 most semantically similar nodes. Our key innovation is an algorithm that ranks sections by aggregating the relevance scores of all the nodes they contain, providing a highly accurate measure of importance.

5) Direct Snippet Extraction: For the subsection_analysis, the refined_text is the full, untruncated content of the top 5 most relevant nodes identified by our semantic search. This ensures the output is both highly relevant and factually grounded in the source documents.

## Project Structure
The repository contains the following key files:

* `Dockerfile`: Defines the instructions to build the self-contained, offline-ready Docker image.

* `requirements.txt`: Lists all necessary Python libraries.

* `process_challenge.py`: The main Python script that contains the entire analysis pipeline.

* `download_mini_model.py`: A helper script used during the Docker build to download the embedding model.

* `setup_essential_models.py`: A helper script for local setup to copy essential Docling models from the cache.

* `Collection 1/`, `Collection 2/`, `Collection 3/`: The data folders, each containing a PDFs/ subdirectory and a `challenge1b_input.json` file.

## Execution Instructions
The project is designed to be run via Docker.

### Prerequisites
* Docker must be installed and running.

* An internet connection is required only for the initial docker build step.

### Step 1: Build the Docker Image
This command builds the Docker image, installing all dependencies and downloading all necessary AI models.

Navigate to the project's root directory in your terminal and run:

```
docker build -t adobe-challenge-1b .
```

### Step 2: Run the Analysis
Once the image is built, you can run the analysis on any collection. This step runs completely offline.

The command mounts the local collection folder into the container, allowing the script to read the PDFs and write the output file back to your local machine.

<b>To process Collection 1:</b>

```
docker run --rm -v "$(pwd)/Collection 1":/app/"Collection 1" adobe-challenge-1b "Collection 1"
```

<b>To process Collection 2:</b>

```
docker run --rm -v "$(pwd)/Collection 2":/app/"Collection 2" adobe-challenge-1b "Collection 2"
```

<b>To process Collection 3:</b>

```
docker run --rm -v "$(pwd)/Collection 3":/app/"Collection 3" adobe-challenge-1b "Collection 3"
```

Upon completion, a `challenge1b_output.json` file will be created inside the respective collection folder.
