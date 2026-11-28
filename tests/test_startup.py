"""The API must refuse to start when MODEL_VERSION names a model with no scores."""

import pytest

from api.services.model_check import check_model_override


def test_a_mistyped_model_version_stops_startup(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "no-such-model-xyz")
    with pytest.raises(RuntimeError, match="no-such-model-xyz"):
        check_model_override()


def test_no_override_is_fine(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    check_model_override()  # must not raise
