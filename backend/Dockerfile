# Use lightweight Python image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Download LocalAI binary directly (no systemd dependencies)
RUN curl -L https://github.com/mudler/LocalAI/releases/latest/download/local-ai-linux-amd64 -o /usr/local/bin/localai \
    && chmod +x /usr/local/bin/localai

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Collect embeddings and static files
RUN python backend/download_embeddings.py
RUN python manage.py collectstatic --noinput

# Expose ports for Django and LocalAI (can be overridden by ENV)
EXPOSE 8000 8080

# Set environment variables (can also be defined in Render dashboard)
ENV LLM_PROVIDER=localai
ENV LOCALAI_URL=http://localhost:8080/v1/generate
ENV LOCALAI_MODEL=mpt-7b-instruct

# Start LocalAI and Django
CMD /usr/local/bin/localai --port 8080 & \
    python manage.py migrate && \
    gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT


