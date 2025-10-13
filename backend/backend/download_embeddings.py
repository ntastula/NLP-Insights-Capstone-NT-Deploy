import numpy as np
from pathlib import Path
import requests
import gzip
import shutil

# ======================
# Configuration
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
DATA_DIR.mkdir(exist_ok=True)

# Full file download
EMBEDDINGS_URL = "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz"
FULL_PATH_GZ = DATA_DIR / "numberbatch-en.txt.gz"
FULL_PATH = DATA_DIR / "numberbatch-en.txt"
SUBSET_PATH = DATA_DIR / "numberbatch-en-top50k-fp16.npz"


# ======================
# Step 1: Download if missing
# ======================
def download_full_embeddings():
    if FULL_PATH.exists():
        print(f"✅ Full embeddings already available at {FULL_PATH}")
        return FULL_PATH

    print("⬇️ Downloading ConceptNet Numberbatch embeddings...")
    r = requests.get(EMBEDDINGS_URL, stream=True)
    r.raise_for_status()

    with open(FULL_PATH_GZ, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("📦 Unzipping embeddings...")
    with gzip.open(FULL_PATH_GZ, "rb") as f_in:
        with open(FULL_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    FULL_PATH_GZ.unlink(missing_ok=True)
    print(f"✅ Download complete! Saved to {FULL_PATH}")
    return FULL_PATH


# ======================
# Step 2: Create subset file
# ======================
def make_subset(n_words=50_000):
    if SUBSET_PATH.exists():
        print(f"✅ Subset already exists at {SUBSET_PATH}")
        return SUBSET_PATH

    if not FULL_PATH.exists():
        download_full_embeddings()

    print(f"📄 Creating subset of top {n_words} embeddings...")
    vocab, vectors = [], []

    with open(FULL_PATH, "r", encoding="utf-8") as f:
        header_skipped = False
        for i, line in enumerate(f):
            # skip header
            if not header_skipped and not line[0].isalpha():
                header_skipped = True
                continue

            if i >= n_words:
                break

            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float16)
            vocab.append(word)
            vectors.append(vec)

    embeddings = np.vstack(vectors)
    np.savez_compressed(SUBSET_PATH, vocab=np.array(vocab), embeddings=embeddings)
    print(f"✅ Saved subset to {SUBSET_PATH}")
    return SUBSET_PATH


# ======================
# Step 3: Loader class
# ======================
class ConceptNetEmbeddings:
    def __init__(self, path=SUBSET_PATH):
        if not path.exists():
            make_subset()
        print("📦 Loading ConceptNet embeddings (NumPy)...")
        data = np.load(path)
        self.vocab = data["vocab"]
        self.embeddings = data["embeddings"]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        print(f"✅ Loaded {len(self.vocab)} embeddings ({self.embeddings.shape[1]} dims, dtype={self.embeddings.dtype})")

    def get_vector(self, word):
        i = self.word_to_idx.get(word.lower())
        return self.embeddings[i] if i is not None else None


if __name__ == "__main__":
    make_subset()

