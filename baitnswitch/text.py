"""Text normalization shared by training, the API and the browser extension.

extension/scorer.js re-implements `normalize` and `TOKEN_PATTERN` exactly;
tests/test_parity.py checks the two agree. Change both together.
"""

import re
import unicodedata

# Tokens are runs of ascii letters/digits, plus "!" and "?" on their own:
# punctuation and numbers are strong clickbait signals ("17 things...!").
TOKEN_PATTERN = r"[a-z0-9]+|[!?]"

_URL = re.compile(r"https?://\S+|www\.\S+")
_RETWEET = re.compile(r"^rt\s+@\w+:?")
_MENTION = re.compile(r"@\w+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Lowercase, strip accents, URLs, retweet prefixes, @mentions and '#'."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.translate(_QUOTES).lower()
    text = _URL.sub(" ", text)
    text = _RETWEET.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", " ")
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9]+", normalize(text)))
