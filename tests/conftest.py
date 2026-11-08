import pytest

from baitnswitch.data import load_corpus
from baitnswitch.model import Classifier, build_pipeline


@pytest.fixture(scope="session")
def classifier() -> Classifier:
    """A small but real model: trained on a 4k sample so the suite stays fast."""
    df = load_corpus().sample(4000, random_state=0)
    return Classifier(build_pipeline().fit(df["text"], df["target"]), threshold=0.45)


HEADLINES = [
    "You Won't Believe What This Dog Did Next",
    "17 Things Only People Who Grew Up In The '90s Will Understand",
    "Senate passes budget bill after overnight session",
    "Here's why everyone is obsessed with this one weird trick!",
    "Earthquake of magnitude 6.1 strikes off the coast of Chile",
    "RT @BuzzFeed: This quiz will tell you which pasta you are http://t.co/abc",
    "Café owner’s “secret” recipe revealed #foodie",
    "a",
    "",
]
