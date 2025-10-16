# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install required tools
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install LocalAI manually
RUN curl -L https://github.com/mudler/LocalAI/releases/download/v3.6.0/local-ai-linux-amd64 -o /usr/local/bin/localai \
    && chmod +x /usr/local/bin/localai

# Download a small model for LocalAI (adjust if needed)
RUN mkdir -p /models && \
    curl -L https://huggingface.co/mys/ggml-mpt-7b-instruct-q4/resolve/main/mpt-7b-instruct-q4.gguf -o /models/mpt-7b-instruct.gguf

# Copy the application code
COPY . .

# Expose Django port
EXPOSE 8000

# Start both LocalAI and Django
CMD /usr/local/bin/localai --models-path /models --port 8080 > /tmp/localai.log 2>&1 & \
    echo "Starting LocalAI..." && sleep 15 && \
    echo "Running migrations..." && python manage.py migrate && \
    echo "Starting Django..." && gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT

