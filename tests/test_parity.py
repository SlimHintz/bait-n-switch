"""The extension's scorer.js must reproduce the Python model's probabilities."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from baitnswitch.model import export_weights
from baitnswitch.text import normalize

from .conftest import HEADLINES

EXTENSION = Path(__file__).resolve().parent.parent / "extension"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_js(weights_path: Path, texts: list[str]) -> dict:
    script = f"""
      import {{ readFileSync }} from "node:fs";
      import {{ createScorer, normalize }} from "{(EXTENSION / 'scorer.js').as_uri()}";
      const scorer = createScorer(JSON.parse(readFileSync({json.dumps(str(weights_path))}, "utf8")));
      const texts = JSON.parse(readFileSync(0, "utf8"));
      console.log(JSON.stringify({{
        normalized: texts.map(normalize),
        probabilities: texts.map(scorer.probability),
      }}));
    """
    out = subprocess.run([NODE, "--input-type=module", "-e", script], input=json.dumps(texts),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_js_matches_python(classifier, tmp_path):
    weights = tmp_path / "weights.json"
    export_weights(classifier, weights)
    js = run_js(weights, HEADLINES)

    assert js["normalized"] == [normalize(h) for h in HEADLINES]
    expected = classifier.probabilities(HEADLINES)
    # Weights are rounded to 5 decimals in the export, so allow a little slack.
    assert js["probabilities"] == pytest.approx(expected.tolist(), abs=1e-3)
