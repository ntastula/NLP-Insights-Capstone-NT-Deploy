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

# Download TinyLlama model (much smaller - ~700MB vs 4GB+)
RUN mkdir -p /models && \
    curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -o /models/tinyllama.gguf

# Create LocalAI model configuration for TinyLlama
RUN echo 'name: tinyllama\n\
backend: llama\n\
parameters:\n\
  model: tinyllama.gguf\n\
  temperature: 0.7\n\
  top_p: 0.9\n\
context_size: 2048\n\
f16: true\n\
threads: 2' > /models/tinyllama.yaml

# Copy backend code
COPY backend/ ./backend/

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting LocalAI with TinyLlama..."\n\
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


