"""Load the three labelled corpora into one frame of (text, target, dataset).

Webis-Clickbait-16/17 annotators judged the *tweet* (post_text), not the
linked article title, so the tweet is what we train on. The 2021 version of
this project paired article titles with tweet labels, which is most of why
its Webis F1 was 0.25.
"""

from pathlib import Path

import pandas as pd

from .text import normalize, word_count

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "chakraborty": ("chakraborty2016.csv", "text"),
    "webis16": ("webis_clickbait16.csv", "post_text"),
    "webis17": ("webis_clickbait17.csv", "post_text"),
}


def load_corpus(data_dir: Path = DATA_DIR, min_words: int = 2) -> pd.DataFrame:
    frames = []
    for name, (filename, column) in SOURCES.items():
        raw = pd.read_csv(data_dir / filename)
        frames.append(pd.DataFrame({
            "raw": raw[column],
            "target": raw["target"].astype(int),
            "dataset": name,
        }))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["raw"])
    df["text"] = df["raw"].map(normalize)
    df = df[df["raw"].map(word_count) >= min_words]
    # The same tweet text can appear more than once (and occasionally with
    # different labels); keep one copy so it can't land in both train and test.
    df = df.drop_duplicates("text", keep="first")
    return df.reset_index(drop=True)
