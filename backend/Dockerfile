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

# Download ultra-small TinyLlama Q2_K quantization (~400MB)
RUN mkdir -p /models && \
    curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
    -o /models/tinyllama.gguf

# Create LocalAI model configuration with minimal memory settings
RUN echo 'name: tinyllama\n\
backend: llama\n\
parameters:\n\
  model: tinyllama.gguf\n\
  temperature: 0.7\n\
  top_p: 0.9\n\
context_size: 512\n\
batch_size: 128\n\
f16: false\n\
low_vram: true\n\
threads: 1\n\
mmap: true\n\
mlock: false' > /models/tinyllama.yaml

# Copy backend code
COPY backend/ ./backend/

# Create startup script with memory-optimized settings
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting LocalAI with ultra-low memory TinyLlama..."\n\
/app/localai --models-path /models --address 0.0.0.0:8080 --threads 1 --context-size 512 &\n\
LOCALAI_PID=$!\n\
\n\
echo "Waiting for LocalAI to be ready..."\n\
for i in {1..90}; do\n\
  if curl -s http://localhost:8080/readyz > /dev/null 2>&1; then\n\
    echo "LocalAI is ready!"\n\
    break\n\
  fi\n\
  if [ $i -eq 90 ]; then\n\
    echo "LocalAI failed to start in time"\n\
    echo "Checking LocalAI logs..."\n\
    kill $LOCALAI_PID 2>/dev/null || true\n\
    exit 1\n\
  fi\n\
  echo "Waiting for LocalAI... ($i/90)"\n\
  sleep 2\n\
done\n\
\n\
echo "Starting Django with single worker to save memory..."\n\
gunicorn backend.wsgi:application --bind 0.0.0.0:10000 --timeout 300 --workers 1 --threads 2' > /app/start.sh && \
chmod +x /app/start.sh

# Expose Render web service port
EXPOSE 10000

# Use the startup script
CMD ["/app/start.sh"]


