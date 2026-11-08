"""JSON API plus a one-page demo.

    uv run flask --app baitnswitch.app run          # development
    uv run gunicorn "baitnswitch.app:create_app()"  # production

POST /api/predict  {"headlines": ["...", ...]}  -> scores for each headline
POST /api/scan     {"url": "https://..."}       -> headlines found on the page, scored
"""

from flask import Flask, jsonify, render_template, request

from .fetch import FetchError, extract_headlines, fetch_html
from .model import Classifier

MAX_HEADLINES = 200
MAX_HEADLINE_CHARS = 500


def create_app(classifier: Classifier | None = None) -> Flask:
    app = Flask(__name__)
    model = classifier or Classifier.load()

    def error(message: str, status: int = 400):
        return jsonify({"error": message}), status

    @app.get("/")
    def index():
        return render_template("index.html", threshold=model.threshold)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        headlines = body.get("headlines")
        if isinstance(body.get("headline"), str):
            headlines = [body["headline"]]
        if not isinstance(headlines, list) or not headlines or not all(isinstance(h, str) for h in headlines):
            return error('send {"headlines": ["...", ...]}')
        if len(headlines) > MAX_HEADLINES:
            return error(f"at most {MAX_HEADLINES} headlines per request")
        headlines = [h[:MAX_HEADLINE_CHARS] for h in headlines]
        return jsonify({"threshold": model.threshold, "results": model.score(headlines)})

    @app.post("/api/scan")
    def scan():
        body = request.get_json(silent=True) or {}
        url = body.get("url")
        if not isinstance(url, str) or not url.strip():
            return error('send {"url": "https://..."}')
        try:
            headlines = extract_headlines(fetch_html(url.strip()))[:MAX_HEADLINES]
        except FetchError as exc:
            return error(str(exc), 422)
        results = model.score(headlines) if headlines else []
        flagged = sum(r["clickbait"] for r in results)
        return jsonify({
            "url": url,
            "threshold": model.threshold,
            "total": len(results),
            "clickbait": flagged,
            "clickbait_share": round(flagged / len(results), 3) if results else None,
            "results": results,
        })

    return app
