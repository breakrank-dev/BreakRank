"""Tests for reading a requirements.txt. No database needed."""

from types import SimpleNamespace

from packaging.version import Version

from api.services.requirements import parse_requirements, releases_after


def pins_of(text):
    pins, _ = parse_requirements(text)
    return {p.name: str(p.version) for p in pins}


def reasons_of(text):
    _, skipped = parse_requirements(text)
    return {s.line_number: s.reason for s in skipped}


def test_exact_pins_are_read():
    assert pins_of("cryptography==46.0.7\nrequests==2.32.3\n") == {
        "cryptography": "46.0.7",
        "requests": "2.32.3",
    }


def test_names_are_matched_the_way_pip_matches_them():
    assert pins_of("Typing_Extensions==4.12.2") == {"typing-extensions": "4.12.2"}


def test_comments_blank_lines_extras_and_markers():
    text = """
# a comment line

cryptography[ssh]==46.0.7  # trailing comment
requests==2.32.3 ; python_version >= "3.9"
"""
    assert pins_of(text) == {"cryptography": "46.0.7", "requests": "2.32.3"}


def test_pip_compile_hashes_and_line_continuations():
    text = (
        "cryptography==46.0.7 \\\n"
        "    --hash=sha256:aaaa \\\n"
        "    --hash=sha256:bbbb\n"
        "requests==2.32.3\n"
    )
    pins, skipped = parse_requirements(text)
    assert {p.name: p.line_number for p in pins} == {"cryptography": 1, "requests": 4}
    assert skipped == []


def test_unpinned_lines_are_reported_with_a_reason():
    reasons = reasons_of("flask>=2\nnumpy\ndjango==4.*\nrequests>=2,<3\n")
    assert reasons == {
        1: "not pinned to one exact version (use ==)",
        2: "no version given (pin it with == to compare)",
        3: "not pinned to one exact version (use ==)",
        4: "not pinned to one exact version (use ==)",
    }


def test_options_urls_and_junk_are_reported_not_dropped():
    reasons = reasons_of(
        "-r base.txt\n"
        "--index-url https://example.org/simple\n"
        "-e .\n"
        "mypkg @ https://example.org/mypkg.zip\n"
        "this is not a requirement\n"
    )
    assert reasons == {
        1: "pip option, not a package",
        2: "pip option, not a package",
        3: "pip option, not a package",
        4: "installed from a URL or a path, not from PyPI",
        5: "could not read this line as a requirement",
    }


def test_a_package_pinned_twice_keeps_the_first_line():
    pins, skipped = parse_requirements("requests==2.32.3\nRequests==2.31.0\n")
    assert [(p.name, str(p.version)) for p in pins] == [("requests", "2.32.3")]
    assert skipped[0].line_number == 2
    assert "already pinned on line 1" in skipped[0].reason


def release(id, version, status="analysed"):
    return SimpleNamespace(id=id, version=version, analysis_status=status)


def test_releases_after_the_pin_oldest_first():
    rows = [release(3, "47.0.0"), release(1, "46.0.6"), release(2, "46.0.7"), release(4, "45.0.0")]
    latest, after = releases_after(Version("46.0.6"), rows)
    assert latest.raw == "47.0.0"
    assert [r.raw for r in after] == ["46.0.7", "47.0.0"]


def test_pre_releases_are_skipped_unless_you_are_on_one():
    rows = [release(1, "1.0.0"), release(2, "2.0.0rc1"), release(3, "1.1.0")]
    latest, after = releases_after(Version("1.0.0"), rows)
    assert latest.raw == "1.1.0"
    assert [r.raw for r in after] == ["1.1.0"]

    _, after = releases_after(Version("2.0.0b1"), rows)
    assert [r.raw for r in after] == ["2.0.0rc1"]


def test_versions_compare_as_versions_not_as_text():
    rows = [release(1, "1.9.0"), release(2, "1.10.0")]
    latest, after = releases_after(Version("1.9.0"), rows)
    assert latest.raw == "1.10.0"  # as text, "1.9.0" > "1.10.0"
    assert [r.raw for r in after] == ["1.10.0"]


def test_already_on_the_latest():
    latest, after = releases_after(Version("2.0"), [release(1, "1.0"), release(2, "2.0")])
    assert latest.raw == "2.0"
    assert after == []
