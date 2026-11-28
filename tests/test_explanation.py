from api.services.explanation import render


def test_one_package_is_singular():
    s = render("OBJECT_REMOVED", "pkg.f", 1, {})
    assert "1 of the ~1,500 scanned packages imports it." in s


def test_many_packages_is_plural():
    s = render("OBJECT_REMOVED", "pkg.f", 8, {})
    assert "8 of the ~1,500 scanned packages import it." in s


def test_no_usage_sentence_when_nobody_imports_it():
    assert "scanned" not in render("OBJECT_REMOVED", "pkg.f", 0, {})


def test_one_inheriting_class_is_singular():
    s = render("OBJECT_REMOVED", "pkg.C", 0, {}, inherited_by=1)
    assert "It affects 1 inheriting class." in s


def test_deprecation_wording():
    s = render("OBJECT_REMOVED", "pkg.f", 0, {"was_deprecated_in": "45.0.0"})
    assert "already marked deprecated in 45.0.0" in s