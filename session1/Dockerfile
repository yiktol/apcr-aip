# Use official Python runtime as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl netcat-openbsd gcc ntpsec-ntpdate && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port that Streamlit runs on
EXPOSE 8501

# Configure Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health

# Run Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]