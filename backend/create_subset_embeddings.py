from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "backend" / "data"
DATA_DIR.mkdir(exist_ok=True)

FULL_PATH = DATA_DIR / "numberbatch-en.txt"  # full embeddings file
SUBSET_PATH = DATA_DIR / "numberbatch-en-top50k.txt"  # output subset

def make_subset(n_words=50_000):
    if SUBSET_PATH.exists():
        print(f"✅ Subset already exists at {SUBSET_PATH}")
        return SUBSET_PATH

    print(f"📄 Creating subset of top {n_words} embeddings...")

    with open(FULL_PATH, "r", encoding="utf-8") as f_in:
        header_skipped = False
        with open(SUBSET_PATH, "w", encoding="utf-8") as f_out:
            for i, line in enumerate(f_in):
                # skip possible header line
                if not header_skipped and not line[0].isalpha():
                    header_skipped = True
                    continue
                if i >= n_words:
                    break
                f_out.write(line)

    print(f"✅ Subset saved to {SUBSET_PATH}")
    return SUBSET_PATH

if __name__ == "__main__":
    make_subset()
