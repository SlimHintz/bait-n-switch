"""The classifier: word 1-2gram TF-IDF into logistic regression.

Deliberately linear so it can be exported as plain weights and run inside
the browser extension with no server (see `export_weights`).
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .text import TOKEN_PATTERN, normalize

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def build_pipeline(C: float = 4.0, max_features: int = 50_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])


@dataclass
class Classifier:
    pipeline: Pipeline
    threshold: float = 0.5

    def probabilities(self, headlines: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(headlines)[:, 1]

    def score(self, headlines: list[str]) -> list[dict]:
        probs = self.probabilities(headlines)
        return [
            {"headline": h, "probability": round(float(p), 4), "clickbait": bool(p >= self.threshold)}
            for h, p in zip(headlines, probs)
        ]

    def save(self, directory: Path = MODEL_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / "model.joblib")
        meta = {"threshold": self.threshold, "sklearn_version": sklearn.__version__}
        (directory / "model.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> "Classifier":
        meta = json.loads((directory / "model.json").read_text())
        if meta["sklearn_version"] != sklearn.__version__:
            raise RuntimeError(
                f"model was trained with scikit-learn {meta['sklearn_version']}, "
                f"but {sklearn.__version__} is installed; retrain with `python -m baitnswitch.train`"
            )
        return cls(joblib.load(directory / "model.joblib"), meta["threshold"])


def export_weights(classifier: Classifier, path: Path) -> None:
    """Write everything extension/scorer.js needs to reproduce predict_proba."""
    tfidf: TfidfVectorizer = classifier.pipeline.named_steps["tfidf"]
    clf: LogisticRegression = classifier.pipeline.named_steps["clf"]
    terms = tfidf.get_feature_names_out()
    payload = {
        "threshold": classifier.threshold,
        "intercept": round(float(clf.intercept_[0]), 6),
        # term -> [idf, coefficient]
        "features": {
            str(t): [round(float(i), 5), round(float(w), 5)]
            for t, i, w in zip(terms, tfidf.idf_, clf.coef_[0])
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
