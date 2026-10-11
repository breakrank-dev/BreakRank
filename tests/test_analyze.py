"""Tests for POST /analyze, against the breakrank-fixture rows in the database."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
FIXTURE = "breakrank-fixture"


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    # Same reason as in test_breakages.py: without this the lifespan never
    # runs, MODEL_VERSION stays None, and tests can pass for the wrong reason.
    with TestClient(app):
        yield


def analyze(requirements, **extra):
    return client.post("/analyze", json={"requirements": requirements, **extra})


def fixture_release(body, version="2.2.0"):
    package = next(p for p in body["packages"] if p["package"] == FIXTURE)
    return next(r for r in package["releases"] if r["version"] == version)


def test_lifespan_ran():
    assert analyze(f"{FIXTURE}==0.0.1").json()["model_version"] is not None


def test_fixture_release_is_listed_with_its_changes():
    response = analyze(f"{FIXTURE}==0.0.1", per_release=20)
    assert response.status_code == 200
    release = fixture_release(response.json())
    assert release["total"] > 0
    assert len(release["breakages"]) == release["total"]
    assert release["ranking"] in ("model", "usage_fallback")
    assert 0 <= release["n_imported"] <= release["total"]


def test_same_ranking_and_total_as_the_release_endpoint():
    one_release = client.get(f"/packages/{FIXTURE}/releases/2.2.0/breakages?limit=20").json()
    release = fixture_release(analyze(f"{FIXTURE}==2.1.99", per_release=20).json())

    def key(b):
        return (b["symbol"], b["kind"], b["sub_target"])

    assert release["total"] == one_release["total"]
    assert [key(b) for b in release["breakages"]] == [key(b) for b in one_release["breakages"]]


def test_private_symbols_are_never_shown():
    release = fixture_release(analyze(f"{FIXTURE}==0.0.1", per_release=20).json())
    assert not any("_parse_header" in b["symbol"] for b in release["breakages"])
    assert all(b["is_private"] is False for b in release["breakages"])


def test_per_release_limit():
    release = fixture_release(analyze(f"{FIXTURE}==0.0.1", per_release=1).json())
    assert len(release["breakages"]) == 1


def test_package_names_match_the_way_pip_matches_them():
    body = analyze("BreakRank_Fixture==0.0.1").json()
    assert [p["package"] for p in body["packages"]] == [FIXTURE]


def test_up_to_date_and_not_tracked():
    body = analyze(f"{FIXTURE}==999.0\nsurely-not-a-tracked-package-xyz==1.0\n").json()
    assert body["packages"] == []
    assert [u["package"] for u in body["up_to_date"]] == [FIXTURE]
    assert body["not_tracked"] == ["surely-not-a-tracked-package-xyz"]


def test_lines_we_cannot_use_come_back_with_reasons():
    body = analyze("flask>=2\n-r base.txt\n").json()
    assert [(s["line_number"], s["reason"]) for s in body["skipped"]] == [
        (1, "not pinned to one exact version (use ==)"),
        (2, "pip option, not a package"),
    ]


def test_empty_input_is_fine():
    body = analyze("").json()
    assert body["packages"] == [] and body["skipped"] == []


def test_input_too_long_is_rejected():
    assert analyze("a==1\n" * 5000).status_code == 422


def test_per_release_out_of_range_is_rejected():
    assert analyze(f"{FIXTURE}==0.0.1", per_release=0).status_code == 422
    assert analyze(f"{FIXTURE}==0.0.1", per_release=21).status_code == 422
