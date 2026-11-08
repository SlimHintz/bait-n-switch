import pytest

from baitnswitch import app as app_module
from baitnswitch.app import create_app
from baitnswitch.fetch import FetchError


@pytest.fixture
def client(classifier):
    return create_app(classifier).test_client()


def test_predict_scores_each_headline(client):
    res = client.post("/api/predict", json={"headlines": ["17 things you need to know", "Court upholds ruling"]})
    assert res.status_code == 200
    results = res.get_json()["results"]
    assert [r["headline"] for r in results] == ["17 things you need to know", "Court upholds ruling"]
    assert results[0]["probability"] > results[1]["probability"]


def test_predict_accepts_single_headline(client):
    res = client.post("/api/predict", json={"headline": "This one trick will change your life"})
    assert len(res.get_json()["results"]) == 1


@pytest.mark.parametrize("body", [None, {}, {"headlines": []}, {"headlines": "not a list"}, {"headlines": [1, 2]}])
def test_predict_rejects_bad_input(client, body):
    res = client.post("/api/predict", json=body) if body is not None else client.post("/api/predict", data="x")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_scan_reports_share(client, monkeypatch):
    html = "<h2>You won't believe these 21 cats</h2><h2>Senate passes budget bill after debate</h2>"
    monkeypatch.setattr(app_module, "fetch_html", lambda url: html)
    data = client.post("/api/scan", json={"url": "https://news.example.com"}).get_json()
    assert data["total"] == 2
    assert data["clickbait_share"] == data["clickbait"] / 2


def test_scan_surfaces_fetch_errors_without_internals(client, monkeypatch):
    def refuse(url):
        raise FetchError("URL points to a private or reserved address")
    monkeypatch.setattr(app_module, "fetch_html", refuse)
    res = client.post("/api/scan", json={"url": "http://169.254.169.254/"})
    assert res.status_code == 422
    assert res.get_json() == {"error": "URL points to a private or reserved address"}


def test_index_renders(client):
    assert b"Bait" in client.get("/").data
