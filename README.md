# NLP Insights Capstone

A Python and React-based application for advanced text analysis, including keyness, clustering, sentiment, and sensorimotor norms analysis. 

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Quick Start Example](#quick-start-example)
- [API Documentation](#api-documentation)
- [Frontend Dependencies](#frontend-dependencies)
- [Backend Dependencies](#backend-dependencies)
- [Python Version](#python-version)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Project Link](#project-link)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

NLP Insights Capstone allows users to analyse their own text against corpora, using a variety of NLP and statistical techniques.  
It includes interactive visualisations, AI-generated summaries, and multiple types of analyses for deeper insights into text content.

![Demo](images/demo.gif)
*Application demo showing keyness analysis workflow*

### Screenshots

<details>
<summary>Click to view feature screenshots</summary>

#### Keyness Analysis
![Keyness Analysis](images/keyness-analysis.png)

#### Clustering Visualization
![Clustering](images/clustering.png)

#### Sentiment Analysis
![Sentiment](images/sentiment-analysis.png)

</details>

---

## Features

### 1. Keyness Analysis
- Compare your text against a genre corpus or another of your own texts.
- Identify significant keywords.
- Visualise results with Plotly.
- Keyness statistics:
  - **NLTK**: Log-likelihood, Effect Size, Keyness Score
  - **Scikit-learn**: Chi², P-value
  - **Gensim**: TF-IDF
  - **spaCy**: Chi², P-value, Log-likelihood, Effect Size, Positive/Negative Keyness
- AI-powered chart summaries, synonyms, and concept suggestions.

### 2. Clustering Analysis
- Cluster your text using ConceptNet or spaCy embeddings.
- Visualise clusters with Plotly scatterplots.
- AI-generated summaries of clusters: themes, topics, thematic flow, overused/underused concepts.
- Inspect top terms and cluster membership.

### 3. Sentiment Analysis
- Compute mean sentiment, magnitude, standard deviation, and composition.
- SentiArt lexicon-based ratings for overall sentiment and five core emotions.
- Identify top positive and negative words.

### 4. Sensorimotor Norms Analysis
- Match keywords against sensorimotor norms.
- Get statistics on matches for the full text.

---

## Prerequisites

Before installing, ensure you have the following:

- **Python 3.12** (Python 3.13+ may cause compatibility issues)
- **Node.js** (v14 or higher) and **npm**
- **Git**
- **Ollama** with **Llama2** model (for AI-generated summaries)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nathan-J-22450784/NLP-Insights-Capstone.git
cd NLP-Insights-Capstone
```

### 2. Set up Python virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt      # development
pip install -r requirements-lock.txt  # production
```

### 4. Download spaCy models (optional)

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

The required spaCy models (`en_core_web_sm` for keyness, `en_core_web_md` for clustering) are included in `requirements.txt`, so this step should not be needed.

### 5. Install Ollama and Llama2

**Install Ollama:**

Visit the [Ollama website](https://ollama.ai/) and follow the installation instructions for your operating system.

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# Download installer from https://ollama.ai/download
```

**Pull the Llama2 model:**

```bash
ollama pull llama2
```

**Start Ollama service:**

```bash
ollama serve
```

This will run Ollama on `http://localhost:11434` by default.

### 6. Install frontend dependencies

```bash
cd frontend
npm install       # or yarn install
```

---

## Configuration

### Environment Variables

Create a `.env` file in the root directory (optional, if you need to configure Ollama endpoint):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### Django Settings

Ensure your Django settings are configured for development:

```bash
python manage.py migrate
```

---

## Usage

### 1. Start Ollama (if not already running)

```bash
ollama serve
```

### 2. Start Backend

```bash
python manage.py runserver
```

The backend will run on `http://localhost:8000`

### 3. Start Frontend

```bash
cd frontend
npm start
```

The frontend will run on `http://localhost:3000`

### 4. Access the app

Open your browser at: http://localhost:3000

---

## Quick Start Example

### Minimal Python workflow to test analyses

**Keyness Analysis**

```python
from nlp_analysis.keyness import analyse_keyness

text = "This is an example text to analyse."
result = analyse_keyness(text, corpus="news_genre")
print(result.keywords)
```

**Clustering Analysis**

```python
from nlp_analysis.clustering import cluster_texts

texts = ["Text one", "Text two", "Text three"]
clusters = cluster_texts(texts, model="en_core_web_md")
print(clusters)
```

**Sentiment Analysis**

```python
from nlp_analysis.sentiment import analyse_sentiment

sentiment = analyse_sentiment("I love NLP projects!")
print(sentiment.summary)
```

**Sensorimotor Norms**

```python
from nlp_analysis.sensorimotor import analyse_norms

norms = analyse_norms("Your text goes here")
print(norms.matches)
```

---

## API Documentation

The backend exposes the following REST API endpoints:

### Corpus Management
- `GET /api/corpora/` - List all available corpora
- `GET /api/corpus-preview/` - Get preview of a specific corpus
- `GET /api/corpus-preview-keyness/` - Get corpus preview for keyness analysis
- `GET /api/corpus-meta-keyness/` - Get corpus metadata for keyness
- `POST /api/create-temp-corpus/` - Create a temporary corpus from uploaded text

### File Upload
- `POST /api/upload-files/` - Upload text files for analysis

### Keyness Analysis
- `POST /api/analyse-keyness/` - Perform keyness analysis on text
- `POST /api/get-keyness-summary/` - Get AI-generated summary of keyness results
- `POST /api/summarise-keyness-chart/` - Get AI summary of keyness visualisation
- `POST /api/get-synonyms/` - Get synonyms for keywords
- `POST /api/get-concepts/` - Get related concepts for keywords

### Clustering Analysis
- `POST /api/clustering-analysis/` - Perform clustering analysis
- `POST /api/summarise-clustering-chart/` - Get AI summary of clustering results
- `POST /api/analyse-themes/` - Analyse themes in clusters
- `POST /api/analyse-thematic-flow/` - Analyse thematic flow across clusters
- `POST /api/analyse-overused-themes/` - Identify overused/underused themes

### Sentiment Analysis
- `POST /api/analyse-sentiment/` - Perform sentiment analysis
- `POST /api/get-sentences/` - Get sentence-level analysis

**Base URL:** `http://localhost:8000/api/`

**Note:** Most POST endpoints expect JSON payloads. Refer to the frontend code for specific request formats.

---

## Frontend Dependencies

All frontend dependencies are listed in `frontend/package.json`. Key packages include:

- React
- Plotly.js
- Other UI and charting libraries

No separate requirements.txt is needed for the frontend — just run `npm install` in the `frontend/` folder.

---

## Backend Dependencies

Backend Python packages are listed in:

- `requirements.txt` (flexible for development)
- `requirements-lock.txt` (pinned for deployment)

**Key packages:**

- numpy, scipy, scikit-learn
- nltk, gensim, spacy
- django, djangorestframework, django-cors-headers
- num2words, python-docx, mammoth, pandas, requests

**SpaCy Models:**

- `en_core_web_sm` → Keyness analysis
- `en_core_web_md` → Clustering analysis

---

## Python Version

This project has been tested with **Python 3.12**.  
Using Python 3.13 or higher may cause compatibility issues with some packages.

---

## Development

- Use a virtual environment for Python dependencies.
- Keep frontend dependencies up-to-date with `npm update`.
- Run tests and follow PEP8 for Python code.
- Frontend code is located in the `frontend/` folder.

---

## Troubleshooting

### Common Issues

**spaCy models not downloading:**
```bash
python -m spacy download en_core_web_sm --user
python -m spacy download en_core_web_md --user
```

**Port conflicts:**
- Backend (Django): Default port 8000. Change with `python manage.py runserver 8001`
- Frontend (React): Default port 3000. Change in `package.json` or set `PORT=3001` environment variable
- Ollama: Default port 11434. Configure in Ollama settings

**CORS issues:**
- Ensure `django-cors-headers` is properly configured in Django settings
- Check that frontend URL is in `CORS_ALLOWED_ORIGINS`

**Ollama connection errors:**
- Verify Ollama is running: `ollama list`
- Check Ollama service: `curl http://localhost:11434/api/tags`
- Ensure llama2 model is installed: `ollama pull llama2`

**Python version compatibility:**
- If using Python 3.13+, consider downgrading to Python 3.12
- Use `python --version` to check your current version

---

## License

This project is licensed under the MIT License.

---

## Project Link

Project Link: [https://github.com/Nathan-J-22450784/NLP-Insights-Capstone](https://github.com/Nathan-J-22450784/NLP-Insights-Capstone)

---

## Acknowledgments

- **spaCy** - Industrial-strength NLP library
- **NLTK** - Natural Language Toolkit
- **Gensim** - Topic modeling and document similarity
- **Plotly** - Interactive visualisation library
- **Django REST Framework** - Web API framework
- **Ollama & Llama2** - Local LLM for AI-generated summaries
- **ConceptNet** - Semantic network for word embeddings
- **SentiArt** - Sentiment and emotion lexicon

Special thanks to all open-source contributors whose libraries made this project possible.