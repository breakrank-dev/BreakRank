# Pipeline changes required — API contract freeze, 3 September 2026

**For: Vaibhav (ML / pipeline side)**

This document is self-contained. If you paste it into an AI assistant, it has
everything needed to act correctly without guessing. Read the context section
first — the three changes only make sense against it.

---

## Context: what this project is

BreakRank ranks breaking changes in Python package dependencies by how much they
actually matter, so a developer doesn't have to read 187 detected API changes to
find the five that will break their code.

There are two halves:

- **Pipeline (yours).** Downloads PyPI packages, runs `griffe` to diff
  consecutive releases and find candidate API changes, AST-scans ~1,500
  downstream packages to build a symbol usage index, derives labels from that
  index, trains a LightGBM learning-to-rank model, and writes scores to the
  database nightly.
- **Web (mine).** PostgreSQL schema, FastAPI service on Render, Next.js frontend
  on Vercel, scheduled jobs.

The database is the boundary. Your pipeline writes to it; my API reads from it.
Neither of us calls the other's code directly.

A key architectural decision that affects you: **the API never runs the model.**
Scores are precomputed by your nightly job and stored in the `prediction` table.
The API is purely a read layer. If a version pair has no precomputed score, the
API returns null rather than scoring on demand.

---

## Current database schema

This is the schema as it exists in Neon right now, after migration 003. Write
against exactly this. Do not add, rename, or drop columns without telling me
first — my endpoints are built on these names.

```sql
CREATE TABLE package (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    download_rank INTEGER,
    github_repo   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE release (
    id          SERIAL PRIMARY KEY,
    package_id  INTEGER NOT NULL REFERENCES package(id) ON DELETE CASCADE,
    version     TEXT NOT NULL,
    released_at TIMESTAMPTZ,
    bump_type   TEXT CHECK (bump_type IN ('major', 'minor', 'patch', 'other')),
    UNIQUE (package_id, version)
);

CREATE TABLE breakage (
    id            SERIAL PRIMARY KEY,
    release_id    INTEGER NOT NULL REFERENCES release(id) ON DELETE CASCADE,
    symbol_path   TEXT NOT NULL,
    kind          TEXT NOT NULL,
    is_private    BOOLEAN NOT NULL DEFAULT FALSE,
    module_depth  INTEGER,
    is_top_level  BOOLEAN,
    in_dunder_all BOOLEAN,
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (release_id, symbol_path, kind)
);

CREATE TABLE usage_index (
    symbol_path TEXT PRIMARY KEY,
    user_count  INTEGER NOT NULL DEFAULT 0,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_run (
    id              SERIAL PRIMARY KEY,
    version         TEXT NOT NULL UNIQUE,
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    pr_auc          REAL,
    precision_at_10 REAL,
    ndcg_at_20      REAL,
    notes           TEXT
);

CREATE TABLE prediction (
    breakage_id   INTEGER NOT NULL REFERENCES breakage(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL REFERENCES model_run(version) ON DELETE CASCADE,
    score         REAL NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (breakage_id, model_version)
);
```

Two things about this schema that affect how you write to it:

**It is safe to re-run ingestion.** The `UNIQUE` constraints on `release` and
`breakage` mean a repeated run cannot create duplicate rows. Use
`ON CONFLICT ... DO NOTHING` or `DO UPDATE` on every insert so a re-run after a
crash or timeout is a no-op rather than an error.

**Before writing a `prediction` row you must have inserted its `model_run`
row.** `prediction.model_version` is a foreign key to `model_run.version`, so
the model has to be registered first.

---

## Change 1 — `is_private` must use this exact definition

**Status: blocking. Affects the ~40% of rows the API hides by default.**

The API excludes private symbols from results unless `?include_private=true` is
passed. That filter reads the `is_private` column your pipeline writes. If our
definitions of "private" differ, roughly 40% of rows are wrongly shown or hidden
and neither of us would notice for weeks — the query would still succeed and
still return plausible-looking data.

The agreed definition:

> A symbol is private if **any** dot-separated component of its path begins with
> a single underscore, excluding dunder names (`__init__`, `__all__`, etc.).
> Membership in the module's `__all__` overrides this and marks the symbol
> public.

Worked examples:

| Symbol path | is_private | Why |
|---|---|---|
| `pandas.DataFrame.append` | `false` | no component starts with `_` |
| `pandas.core.frame._parse_header` | `true` | leaf starts with `_` |
| `pandas.core._internals.Block` | `true` | a middle component starts with `_` — nobody outside the package can reach it |
| `pandas._libs.tslibs.Timestamp` | `true` | `_libs` is private |
| `numpy.__version__` | `false` | dunder, excluded from the rule |
| `attrs._make.attrib` | `false` **if** `attrib` is in `attrs.__all__` | the package explicitly exported it |

Reference implementation:

```python
def is_private_symbol(symbol_path: str, dunder_all: set[str] | None = None) -> bool:
    """Private if any path component starts with a single underscore.

    Dunder names are excluded. Membership in __all__ overrides to public.
    """
    if dunder_all and symbol_path.split(".")[-1] in dunder_all:
        return False

    for component in symbol_path.split("."):
        if component.startswith("_") and not component.startswith("__"):
            return True
    return False
```

Note the `__all__` check uses the leaf name, since that is what a package
exports.

**Do not drop private symbols from the database.** They are needed as negative
training examples — they are exactly the "changed but nobody cares" cases the
model has to learn to rank low. They are stored and hidden, not discarded.

---

## Change 2 — `breakage.explanation` is now `breakage.detail`, and it is JSONB

**Status: blocking. The old column no longer exists.**

Migration 003 dropped `explanation TEXT` and added `detail JSONB NOT NULL
DEFAULT '{}'::jsonb`. Any insert referencing `explanation` will now fail with
`UndefinedColumn`.

**The reasoning**, because it changes what you write rather than just where:

`explanation` was going to be a human-readable sentence shown in the UI —
something like *"pandas.DataFrame.append was removed. 412 packages call it."*
The original plan had the pipeline writing that sentence into the database.

We moved sentence-building to the API. Wording is presentation, and if the
sentence is stored, rewording it means re-running ingestion across thousands of
releases instead of redeploying the API in two minutes. We will want to reword
it, probably the week before the demo.

So the split is:

- **Pipeline writes facts** that only the pipeline knows.
- **API composes the sentence** from `kind`, `symbol_path`, the `user_count`
  from `usage_index`, and your `detail` payload.

Write whatever you know. Suggested keys, all optional:

```python
detail = {
    "griffe_message": "...",        # griffe's own description, verbatim
    "griffe_kind": "...",           # griffe's BreakageKind enum name
    "parameter": "verbose",         # for PARAMETER_* kinds
    "old_value": "None",            # for default-value changes
    "new_value": "0",
    "was_deprecated_in": "2.1.0",   # if a deprecation preceded the removal
    "moved_to": "pandas.io.x.y",    # if the symbol was relocated, not removed
}
```

Keys can differ by breakage kind — that is the reason for JSONB rather than
more columns. A `PARAMETER_REMOVED` has a parameter name; an `OBJECT_REMOVED`
does not. Typed columns for every possible field would be mostly nulls.

The rule we settled on: **queryable attributes are typed columns, the variable
payload is JSONB.** So `kind`, `is_private`, `symbol_path`, `module_depth` stay
as real columns because the API filters and sorts on them. Everything else goes
in `detail`.

Insert it as a JSON string:

```python
import json

conn.execute(
    text("""
        INSERT INTO breakage
            (release_id, symbol_path, kind, is_private,
             module_depth, is_top_level, in_dunder_all, detail)
        VALUES (:r, :s, :k, :priv, :depth, :top, :in_all, :detail)
        ON CONFLICT (release_id, symbol_path, kind) DO UPDATE
            SET detail = EXCLUDED.detail
    """),
    {..., "detail": json.dumps(detail)},
)
```

If a case ever needs information the API cannot reconstruct, put it in
`griffe_message` and I will append it verbatim to the rendered sentence. Nothing
you know gets thrown away — we are only deciding where the sentence is
assembled.

---

## Change 3 — breakages must be per consecutive release pair

**Status: check before the 300-package run. May already be correct.**

Please confirm this is what the pipeline already does, because changing it after
a large run means redoing the run.

The `/analyze` endpoint takes a user's `requirements.txt` — say they pin
`pandas==2.1.0` — and reports what breaks if they upgrade to the latest,
`2.2.0`. But `2.1.1`, `2.1.2` and `2.1.3` shipped in between.

**Breaking changes hide in patch releases. That is the entire premise of the
project** — measured studies put it around a third of non-major releases
carrying at least one breaking change despite the version number promising
otherwise. If we only diff `2.1.0` against `2.2.0` directly, we miss anything
introduced in `2.1.2` and then re-exported by `2.2.0`, and we contradict our own
thesis.

So the pipeline needs a breakage row for **every consecutive release pair**:

```
2.1.0 -> 2.1.1
2.1.1 -> 2.1.2
2.1.2 -> 2.1.3
2.1.3 -> 2.2.0
```

not a single `2.1.0 -> 2.2.0` diff.

`breakage.release_id` points at the **later** release of each pair — the release
that introduced the change. The API then walks the range `(current, latest]`,
collects breakages from every release in it, deduplicates by symbol keeping the
highest score, and reports `via_version` so the UI can say which release
introduced each one.

Two consequences worth flagging:

- **Volume.** Per-pair diffing produces more rows than per-package diffing.
  Storage is not a problem — Neon's free tier is ~500 MB and breakage rows are
  small — but the griffe runtime multiplies by the number of releases per
  package. Worth knowing before the 300-package run rather than during it.
- **Release ordering.** Consecutive pairs require correctly sorted versions,
  which means PEP 440 ordering, not string sorting. `2.10.0` sorts after `2.9.0`
  numerically but before it alphabetically. Use `packaging.version.parse` for
  the sort key.

---

## Things that have not changed

So nothing gets rewritten unnecessarily:

- All other column names and types are as originally planned.
- The `usage_index` table is unchanged — `symbol_path` primary key,
  `user_count`, `computed_at`.
- `model_run` metric column names are unchanged: `pr_auc`, `precision_at_10`,
  `ndcg_at_20`.
- Nothing about the model, features, labelling method, or training approach is
  affected by any of this. These are storage and boundary decisions only.

---

## What to do

1. Read the three changes above.
2. Confirm change 3 — is your pipeline already diffing consecutive pairs?
3. Apply changes 1 and 2 before the 300-package run, so the run produces data
   that matches what the API expects.
4. Reply on PR #7 with anything you disagree with. The contract is frozen, which
   means it doesn't change without a conversation — not that it can't change.

The full contract, with the reasoning behind all eight decisions, is in
`docs/api-contract.md` on the `feat/database-schema` branch.

---

## If you are pasting this into an AI assistant

Everything it needs is above — schema, definitions, worked examples, reference
code. It should not need to guess at column names or invent a `is_private` rule.

One caution: read what it produces before running it. The `is_private`
definition in particular is subtle (the "any component" part, and the `__all__`
override), and an assistant that skims will implement the obvious
`symbol.startswith("_")` version, which is wrong for
`pandas.core._internals.Block`. That single mistake would silently corrupt about
40% of the labels — and the pipeline would run without any error at all.
