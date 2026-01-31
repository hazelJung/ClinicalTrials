# Use official Python 3.12 image (Debian-based for stability)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for RDKit/Mordred)
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies only from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add current directory to PYTHONPATH so we can import 'pksmart' without installing it
ENV PYTHONPATH=/app

# Default command: run prediction verification
CMD ["python", "pksmart_predict.py", "--smiles", "CC(=O)Oc1ccccc1C(=O)O"]
