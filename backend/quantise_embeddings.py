from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "backend" / "data"  # adjust to your folder structure

INPUT_PATH = DATA_DIR / "numberbatch-en-top50k.txt"
OUTPUT_PATH_FP16 = DATA_DIR / "numberbatch-en-top50k-fp16.pt"
OUTPUT_PATH_INT4 = DATA_DIR / "numberbatch-en-top50k-4bit.pt"  # optional

def load_embeddings(txt_path):
    """
    Load ConceptNet embeddings from text file into a Python dict.
    Returns vocab list and numpy array of embeddings.
    """
    vocab = []
    vectors = []

    print(f"📄 Loading embeddings from {txt_path}...")
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading lines"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word, *vec = parts
            vocab.append(word)
            vectors.append(np.array(vec, dtype=np.float32))

    embeddings = np.vstack(vectors)
    print(f"✅ Loaded {len(vocab)} embeddings, shape: {embeddings.shape}")
    return vocab, embeddings

def save_fp16(vocab, embeddings, out_path):
    """
    Save embeddings as float16 PyTorch tensor
    """
    print(f"💾 Saving embeddings as float16 to {out_path}...")
    tensor = torch.tensor(embeddings, dtype=torch.float16)
    torch.save({"vocab": vocab, "embeddings": tensor}, out_path)
    print("✅ Done saving float16 file.")

def save_4bit(vocab, embeddings, out_path):
    """
    Optional: Save 4-bit quantised embeddings using PyTorch (basic linear quantisation)
    """
    print(f"💾 Saving embeddings as 4-bit quantised to {out_path}...")
    tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Simple linear 4-bit quantisation (0-15)
    min_val = tensor.min(dim=1, keepdim=True).values
    max_val = tensor.max(dim=1, keepdim=True).values
    scale = (max_val - min_val) / 15
    quantised = torch.round((tensor - min_val) / scale).to(torch.uint8)

    torch.save({"vocab": vocab, "quantised": quantised, "min": min_val, "scale": scale}, out_path)
    print("✅ Done saving 4-bit file.")

if __name__ == "__main__":
    vocab, embeddings = load_embeddings(INPUT_PATH)
    save_fp16(vocab, embeddings, OUTPUT_PATH_FP16)
    # Uncomment below if you want 4-bit too
    # save_4bit(vocab, embeddings, OUTPUT_PATH_INT4)
