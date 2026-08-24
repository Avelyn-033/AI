# -*- coding: utf-8 -*-
"""
Shared NLP preprocessing utilities.
Imported by BOTH train.py and app.py so the cleaning logic used when the
model was trained is guaranteed to be identical to the logic used at
inference time in the Streamlit app.
"""

import re
import nltk

# Download required NLTK resources once (quiet + safe to re-run).
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Lowercase -> strip punctuation/numbers -> tokenize -> remove stopwords -> lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words and len(w) > 2]
    return " ".join(tokens)
