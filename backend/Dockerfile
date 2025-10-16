# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git build-essential wget && \
    rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install them
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download LocalAI binary directly (for Linux x86_64)
RUN curl -Lo /app/localai https://github.com/mudler/LocalAI/releases/download/v2.21.1/local-ai-Linux-x86_64 && \
    chmod +x /app/localai

# Download model for LocalAI
RUN mkdir -p /models && \
    curl -L https://huggingface.co/mys/ggml-mpt-7b-instruct-q4/resolve/main/mpt-7b-instruct-q4.gguf \
    -o /models/mpt-7b-instruct.gguf

# Create LocalAI model configuration
RUN echo 'name: mpt-7b-instruct\n\
backend: llama\n\
parameters:\n\
  model: mpt-7b-instruct.gguf\n\
  temperature: 0.7\n\
  top_p: 0.9\n\
context_size: 2048\n\
f16: true\n\
threads: 4' > /models/mpt-7b-instruct.yaml

# Copy backend code
COPY backend/ ./backend/

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting LocalAI..."\n\
/app/localai --models-path /models --address 0.0.0.0:8080 &\n\
LOCALAI_PID=$!\n\
\n\
echo "Waiting for LocalAI to be ready..."\n\
for i in {1..60}; do\n\
  if curl -s http://localhost:8080/readyz > /dev/null 2>&1; then\n\
    echo "LocalAI is ready!"\n\
    break\n\
  fi\n\
  if [ $i -eq 60 ]; then\n\
    echo "LocalAI failed to start in time"\n\
    kill $LOCALAI_PID 2>/dev/null || true\n\
    exit 1\n\
  fi\n\
  echo "Waiting for LocalAI... ($i/60)"\n\
  sleep 2\n\
done\n\
\n\
echo "Starting Django..."\n\
gunicorn backend.wsgi:application --bind 0.0.0.0:10000 --timeout 300 --workers 2' > /app/start.sh && \
chmod +x /app/start.sh

# Expose Render web service port
EXPOSE 10000

# Use the startup script
CMD ["/app/start.sh"]


