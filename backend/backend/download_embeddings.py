import requests
from pathlib import Path
from io import BytesIO
import torch
import numpy as np

# ======================
# Configuration
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LOCAL_PT_PATH = DATA_DIR / "numberbatch-en-top50k-fp16.pt"
HF_PT_URL = "https://huggingface.co/ntastula/numberbatch-quantized/resolve/main/numberbatch-en-top50k-fp16.pt"

# ======================
# Step 1: Download embeddings stream from Hugging Face
# ======================
def download_embeddings_stream(url=HF_PT_URL):
    print(f"⬇️ Streaming embeddings from {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    buf = BytesIO()
    for chunk in r.iter_content(chunk_size=8192):
        buf.write(chunk)
    buf.seek(0)
    print(f"✅ Stream download complete")
    return buf

# ======================
# Step 2: ConceptNet embeddings loader (module-level load)
# ======================
class ConceptNetEmbeddings:
    def __init__(self, local_path=LOCAL_PT_PATH, hf_url=HF_PT_URL):
        self.local_path = local_path
        self.hf_url = hf_url
        self._data = None
        self._vocab = None
        self._embeddings = None

        # Load embeddings immediately
        self._load_embeddings()

    def _load_embeddings(self):
        if self.local_path.exists():
            self._load_from_disk()
        else:
            self._load_from_stream()

    def _load_from_disk(self):
        print(f"📂 Loading embeddings from local file: {self.local_path} ...")
        with open(self.local_path, "rb") as f:
            self._data = torch.load(f, map_location="cpu")
        self._process_data()

    def _load_from_stream(self):
        buf = download_embeddings_stream(self.hf_url)
        print("📡 Loading embeddings from stream ...")
        self._data = torch.load(buf, map_location="cpu")
        self._process_data()

    def _process_data(self):
        # Handle both dict and tuple formats
        if isinstance(self._data, dict):
            self._vocab = self._data.get("vocab")
            self._embeddings = np.array(self._data.get("embeddings"), dtype=np.float16, copy=False)
        elif isinstance(self._data, (list, tuple)) and len(self._data) == 2:
            self._vocab, self._embeddings = self._data
            self._embeddings = np.array(self._embeddings, dtype=np.float16, copy=False)
        else:
            raise ValueError("❌ Unrecognised embedding file format.")

        # Convert vocab list to dict if needed
        if isinstance(self._vocab, list):
            self._vocab = {word: i for i, word in enumerate(self._vocab)}

        print(f"✅ Embeddings loaded: {len(self._vocab)} words, shape {self._embeddings.shape}")

    def get_vector(self, word):
        idx = self._vocab.get(word)
        if idx is not None:
            return self._embeddings[idx]
        return None

    def get_vectors_for_corpus(self, words):
        filtered = {w: self._embeddings[self._vocab[w]] for w in words if w in self._vocab}
        print(f"✅ Loaded {len(filtered)} vectors for corpus ({len(words)} requested)")
        return filtered

# ======================
# Module-level load
# ======================
print("Loading ConceptNet embeddings (module-level)...")
model = ConceptNetEmbeddings()
print("✅ ConceptNet embeddings ready.")

