import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.db import engine
from api.main import app

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    """TestClient only runs startup/shutdown inside a context manager.

    Without this, config.MODEL_VERSION stays None for the whole test run and
    several assertions pass vacuously — a None model version matches no
    predictions, so the usage fallback looks correct even if it isn't.
    """
    with TestClient(app):
        yield

BASE = "/packages/breakrank-fixture/releases/2.2.0/breakages"


def test_health_reports_a_model_version():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_fixture_model_never_wins_selection():
    """The fixture's trained_at is pinned to 2020 for exactly this reason."""
    mv = client.get("/health").json()["model_version"]
    assert mv is not None, "lifespan did not run; model was never resolved"
    assert mv != "v0-fake"


def test_breakages_returns_fixture_rows():
    r = client.get(BASE)
    assert r.status_code == 200
    body = r.json()
    assert body["package"] == "breakrank-fixture"
    assert body["version"] == "2.2.0"
    assert len(body["breakages"]) > 0


def test_private_symbols_excluded_by_default():
    """Decision 3."""
    assert all(not b["is_private"] for b in client.get(BASE).json()["breakages"])


def test_private_symbols_included_when_requested():
    """Decision 3, the opt-in half."""
    default = client.get(BASE).json()
    with_private = client.get(f"{BASE}?include_private=true").json()
    assert with_private["total"] > default["total"]


def test_unscored_falls_back_to_usage_ordering():
    """Decision 2. Fixture predictions sit under v0-fake, which is never the
    resolved model, so these rows are always unscored."""
    body = client.get(BASE).json()
    assert body["ranking"] == "usage_fallback"
    counts = [b["user_count"] for b in body["breakages"]]
    assert counts == sorted(counts, reverse=True)


def test_ranking_field_is_always_present():
    """Decision 2: the client must know which ordering it got."""
    assert client.get(BASE).json()["ranking"] in ("model", "usage_fallback")


def test_analysis_status_is_returned():
    """Decisions 5 and 6: an empty list means different things."""
    assert client.get(BASE).json()["analysis_status"] == "analysed"


def test_unknown_package_returns_not_tracked():
    """Decision 1."""
    r = client.get("/packages/definitely-not-real/releases/1.0.0/breakages")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_tracked"


def test_unknown_version_of_known_package_returns_not_tracked():
    """Decision 1: the version matters, not just the package."""
    r = client.get("/packages/breakrank-fixture/releases/9.9.9/breakages")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_tracked"


def test_limit_is_capped():
    assert client.get(f"{BASE}?limit=999").status_code == 422


def test_limit_must_be_positive():
    assert client.get(f"{BASE}?limit=0").status_code == 422


def test_total_is_the_pre_limit_count():
    """total drives 'showing 20 of 187' in the UI."""
    body = client.get(f"{BASE}?limit=2").json()
    assert len(body["breakages"]) == 2
    assert body["total"] > 2


def test_explanation_mentions_the_symbol():
    """Decision 4."""
    top = client.get(BASE).json()["breakages"][0]
    assert len(top["explanation"]) > 10
    assert top["symbol"] in top["explanation"]


def test_explanation_includes_parameter_name():
    """Decisions 4 and 9: sub_target reaches the sentence."""
    body = client.get(f"{BASE}?limit=100").json()
    verbose = next(b for b in body["breakages"] if b["sub_target"] == "verbose")
    assert "verbose" in verbose["explanation"]


def test_two_parameters_removed_from_one_function_are_distinct():
    """Regression for migration 004.

    The pre-004 unique key (release_id, symbol_path, kind) collapsed these
    into one row via ON CONFLICT DO NOTHING. Silently.
    """
    body = client.get(f"{BASE}?limit=100").json()
    read_csv = [b for b in body["breakages"] if b["symbol"] == "pandas.read_csv"]
    assert len(read_csv) == 2
    assert {b["sub_target"] for b in read_csv} == {"verbose", "squeeze"}


def _a_scored_release():
    """Find any real release with predictions under the current model."""
    from api import config

    with engine.connect() as conn:
        return conn.execute(
            text("""
                SELECT pk.name, r.version
                FROM prediction p
                JOIN breakage b ON b.id = p.breakage_id
                JOIN release r  ON r.id = b.release_id
                JOIN package pk ON pk.id = r.package_id
                WHERE p.model_version = :mv
                LIMIT 1
            """),
            {"mv": config.MODEL_VERSION},
        ).first()


def test_real_data_ranks_by_model():
    """Decision 2, the other branch. Skips if the pipeline hasn't loaded."""
    found = _a_scored_release()
    if not found:
        pytest.skip("no scored real data in the database")

    name, version = found
    body = client.get(f"/packages/{name}/releases/{version}/breakages").json()
    assert body["ranking"] == "model"
    assert any(b["score"] is not None for b in body["breakages"])