import requests
from pathlib import Path
from io import BytesIO
import torch
import numpy as np

# ======================
# Configuration
# ======================
MODEL_URL = "https://huggingface.co/ntastula/numberbatch-quantized/resolve/main/numberbatch-en-top50k-fp16.pt"

_model = None

def load_embeddings():
    """
    Downloads and loads ConceptNet Numberbatch embeddings on demand.
    """
    print("⬇️ Streaming ConceptNet embeddings from Hugging Face...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    buffer = BytesIO(response.content)

    print("📡 Loading embeddings from stream...")
    model = torch.load(buffer, map_location=torch.device("cpu"))
    print(f"✅ Embeddings loaded: {len(model)} vectors, shape {next(iter(model.values())).shape}")
    return model


def get_model():
    """
    Lazy-load the embeddings — loads once, then reuses.
    """
    global _model
    if _model is None:
        print("⚙️ Loading ConceptNet embeddings (first-time use)...")
        _model = load_embeddings()
        print("✅ ConceptNet embeddings ready.")
    return _model
