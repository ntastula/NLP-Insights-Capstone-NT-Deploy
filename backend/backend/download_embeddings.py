import requests
from pathlib import Path

# ======================
# Configuration
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LOCAL_PT_PATH = DATA_DIR / "numberbatch-en-top50k-fp16.pt"
HF_PT_URL = "https://huggingface.co/ntastula/numberbatch-quantized/resolve/main/numberbatch-en-top50k-fp16.pt"

# ======================
# Download embeddings if missing
# ======================
def ensure_embeddings(local_path=LOCAL_PT_PATH, url=HF_PT_URL):
    if local_path.exists():
        print(f"📂 Embeddings already exist: {local_path}")
        return local_path
    print(f"⬇️ Downloading embeddings from {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Download complete: {local_path}")
    return local_path

# ======================
# ConceptNet embeddings loader (deferred loading)
# ======================
class ConceptNetEmbeddings:
    def __init__(self, local_path=LOCAL_PT_PATH, hf_url=HF_PT_URL):
        self.local_path = local_path
        self.hf_url = hf_url
        self._data = None
        self._vocab = None
        self._embeddings = None

        # Ensure file exists but do NOT load
        ensure_embeddings(self.local_path, self.hf_url)

    def load(self):
        """Call this at runtime, after user text is uploaded."""
        if self._data is not None:
            return

        import torch
        import numpy as np

        print(f"📂 Loading embeddings from {self.local_path} ...")
        self._data = torch.load(self.local_path, map_location="cpu")

        # Handle dict or tuple format
        if isinstance(self._data, dict):
            self._vocab = self._data.get("vocab")
            self._embeddings = np.array(self._data.get("embeddings"), copy=False)
        elif isinstance(self._data, (list, tuple)) and len(self._data) == 2:
            self._vocab, self._embeddings = self._data
            self._embeddings = np.array(self._embeddings, copy=False)
        else:
            raise ValueError("❌ Unrecognised embedding file format.")

        if isinstance(self._vocab, list):
            self._vocab = {word: i for i, word in enumerate(self._vocab)}

    def get_vector(self, word):
        if self._data is None:
            self.load()
        idx = self._vocab.get(word)
        if idx is not None:
            return self._embeddings[idx]
        return None

    def get_vectors_for_corpus(self, words):
        if self._data is None:
            self.load()
        return {w: self._embeddings[self._vocab[w]] for w in words if w in self._vocab}

