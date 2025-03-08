# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for development
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install all dependencies (including dev ones)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run when container starts
CMD ["python", "app.py"] 