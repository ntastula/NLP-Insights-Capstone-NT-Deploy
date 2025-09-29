import os
import re
import json
import requests
from collections import Counter
from pathlib import Path

# Folder where JSON output will be saved
CORPUS_DIR = Path("corpus")
CORPUS_META_DIR = Path("corpus_meta_keyness")

GENRE_BOOKS = {
    "fantasy": [55, 28885, 15948, 86, 67090], # The Wonderful Wizard of Oz, Alice's Adventures in Wonderland, The Hollow Land, A Connecticut Yankee in King Arthur's Court, The Worm Ouroboros
    "horror": [345, 84, 43, 209, 10002], # Dracula, Frankenstein, The Strange Case of Dr Jekyll & Mr Hyde, The Turn of the Screw, The House on the Borderland
    "mystery": [1661, 58866, 583, 1872, 76677], # The Adventures of Sherlock Holmes, The Murder on the Links, The Woman in White, The Red House Mystery, The Girl from Scotland Yard
    "romance": [42671, 768, 1399, 64317, 4276], # Pride and Prejudice, Wuthering Heights, Anna Karenina, The Great Gatsby, North and South
    "science_fiction": [164, 36, 1250, 62, 1951], # Twenty Thousand Leagues Under the Sea, The War of the Worlds, The Eyes Have It, A Princess of Mars, The Coming Race
    "general_fiction": [1231, 2554, 71865, 98, 174],  # On the Track, Crime and Punishment, Mrs Dalloway, A Tale of Two Cities, The Picture of Dorian Gray
    "poetry": [20158, 12242, 3011, 213, 26],  # The Works of Lord Byron Vol 4, Poems by Emily Dickinson, The Lady of the Lake, The Man From Snowy River, Paradise Lost
    "young_adult": [30142, 45, 17396, 271, 236] # Little Brother, Anne of Green Gables, The Secret Garden, Black Beauty, The Jungle Book
}

def download_gutenberg_text(book_id: int) -> str:
    """Download raw text from Project Gutenberg by ID."""
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
    ]
    for url in urls:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            print(f"Failed to download {book_id} from {url}: {e}")
    return ""

def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg header/footer and return clean text."""
    start_marker = "*** START OF"
    end_marker = "*** END OF"

    start_idx = text.find(start_marker)
    if start_idx != -1:
        text = text[start_idx:].split("\n", 1)[1]

    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    return text.strip()

def tokenize(text: str):
    """Lowercase, strip punctuation, and split into tokens longer than 2 chars."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in text.split() if len(t) > 2]
    return tokens

import re

def extract_title_author(file_path):
    """Extract title and author from the first few lines of a text file, with standardised casing."""
    title, author = "Unknown Title", "Unknown Author"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Grab the first 10 lines
            lines = [line.strip() for _, line in zip(range(10), f)]
            lines = [line for line in lines if line]  # remove empty lines

        # First non-empty line is the title
        if lines:
            raw_title = lines[0]
            # Standardise to Title Case
            title = raw_title.title()
            # Fix apostrophe cases like Alice'S -> Alice's
            title = re.sub(r"'([A-Z])", lambda m: "'" + m.group(1).lower(), title)

        # Look for an author line (e.g., "by Charles Dickens")
        for line in lines[1:]:
            match = re.search(r"by\s+(.+)", line, re.IGNORECASE)
            if match:
                raw_author = match.group(1).strip()
                author = raw_author.title()
                break

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return title, author


def process_file(file_path):
    """Process a single book file to extract metadata and word frequencies."""
    title, author = extract_title_author(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Preview = first 500 characters
        preview = text[:500]

        # Simple frequency count (case insensitive)
        words = re.findall(r"\b\w+\b", text.lower())
        freq = Counter(words)

        return {
            "title": title,
            "author": author,
            "preview": preview,
            "frequencies": freq,
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def build_genre_corpus(genre_folder: Path):
    genre = genre_folder.name
    previews = []
    all_tokens = []

    for txt_file in sorted(genre_folder.glob("*.txt")):
        print(f"Processing {txt_file.name} for genre {genre}...")
        book_data = process_file(txt_file)
        if not book_data:
            continue

        previews.append({
            "title": book_data["title"],
            "author": book_data["author"],
            "snippet": book_data["preview"],
        })

        all_tokens.extend(book_data["frequencies"].elements())

    freq = dict(Counter(all_tokens))
    total_tokens = len(all_tokens)

    out_data = {
        "genre": genre,
        "version": "2025-09-18",
        "previews": previews,
        "doc_count": len(previews),
        "total_tokens": total_tokens,
        "counts": freq
    }

    out_file = CORPUS_META_DIR / f"{genre}_keyness.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {out_file}")

if __name__ == "__main__":
    # Make sure output folder exists
    CORPUS_META_DIR.mkdir(parents=True, exist_ok=True)

    # Loop over each genre folder in corpus/
    for genre_folder in sorted(CORPUS_DIR.iterdir()):
        if genre_folder.is_dir():
            build_genre_corpus(genre_folder)
