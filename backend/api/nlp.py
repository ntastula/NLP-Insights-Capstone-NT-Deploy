from functools import lru_cache
import spacy
from spacy.cli import download

# If the model ever isn’t present, try to download it once, else fall back to a blank English pipeline
@lru_cache(maxsize=1)
def get_nlp():
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "senter", "attribute_ruler"])
    except Exception:
        try:
            # one-time best effort
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", exclude=["ner", "senter", "attribute_ruler"])
        except Exception:
            # last-resort: no POS tagger, so filter functions should tolerate this
            nlp = spacy.blank("en")
    nlp.max_length = 1_000_000
    return nlp