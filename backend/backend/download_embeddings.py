import requests
import gzip
import shutil
from pathlib import Path

# Make data folder right next to this file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Paths for downloaded embeddings
EMBEDDINGS_URL = "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz"
DEST_PATH = DATA_DIR / "numberbatch-en.txt.gz"
FINAL_PATH = DATA_DIR / "numberbatch-en.txt"

def download_embeddings():
    if FINAL_PATH.exists():
        print("✅ Embeddings already available.")
        return FINAL_PATH

    print("⬇️ Downloading ConceptNet Numberbatch embeddings...")
    r = requests.get(EMBEDDINGS_URL, stream=True)
    with open(DEST_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("📦 Unzipping...")
    with gzip.open(DEST_PATH, "rb") as f_in:
        with open(FINAL_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Delete the original .gz file
    DEST_PATH.unlink(missing_ok=True)

    print(f"✅ Download complete! Embeddings saved to {FINAL_PATH}")
    return FINAL_PATH
