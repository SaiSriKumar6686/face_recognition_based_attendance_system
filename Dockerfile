# Use official Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies required by OpenCV and InsightFace
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Exclude some heavy libraries not strictly needed for the demo if possible,
# or just install them all.
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data/embeddings data/demo_uploads data/seed_images data/reference_images data/cctv_crops

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH="/"
ENV PYTHONIOENCODING="utf-8"

# Expose the port Flask runs on
EXPOSE 7860

# Command to run the demo (HuggingFace routes traffic to port 7860)
CMD ["python", "scripts/demo.py", "--port", "7860", "--host", "0.0.0.0"]
