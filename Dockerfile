# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies using pip with cache mounting
# This preserves the pip cache across builds, even if requirements.txt changes
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --upgrade pip && \
    pip install --timeout 300 torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --timeout 300 -r requirements.txt

# Pre-download the embedding model used by SemanticCache to avoid download on startup
# We use a cache mount for HuggingFace to persist the model across builds
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-download NLTK data (used by LlamaIndex and other NLP tools)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# Pre-download Docling models to avoid download on first use
RUN --mount=type=cache,target=/root/.cache/docling \
    python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# Copy the rest of the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
