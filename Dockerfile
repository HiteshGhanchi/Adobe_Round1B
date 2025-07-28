# Stage 1: Builder Stage to Download All Docling Models
# We use this temporary stage to run the official download tool.
FROM python:3.11-slim AS builder

# Install docling and its CPU-only PyTorch dependency.
RUN pip install docling --extra-index-url https://download.pytorch.org/whl/cpu

# Run the official docling model downloader.
RUN docling-tools models download


# Stage 2: Final Application Image
FROM python:3.11-slim

WORKDIR /app

# Copy over the file that defines our Python dependencies
COPY requirements.txt .

# Install all necessary packages using pip.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# --- Simplified Model Handling ---
# Create the directory for our custom models
RUN mkdir -p /app/custom_docling_models

# THE FIX: Copy the ENTIRE 'models' directory from the builder stage's cache.
# This is simpler and more robust than copying individual sub-folders.
COPY --from=builder /root/.cache/docling/models /app/custom_docling_models
RUN rm -r ./custom_docling_models/ds4sd--CodeFormula/
RUN rm -r ./custom_docling_models/EasyOcr/

# Copy and run the script to download the small embedding model.
COPY download_mini_model.py .
RUN python download_mini_model.py

# Copy the main application script into the container
COPY process_challenge.py .

# Set the entrypoint for the container.
ENTRYPOINT ["python", "process_challenge.py"]

# The default command is empty, as the user must provide the collection path.
CMD [""]
