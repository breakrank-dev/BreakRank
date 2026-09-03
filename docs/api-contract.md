# BreakRank API Contract — froze, 3rd sept

Base URL: https://<render-service>.onrender.com
All timestamps: ISO 8601, UTC, e.g. "2024-01-19T00:00:00Z"
All errors: {"error": {"code": "not_found", "message": "..."}}

## GET /health
{"ok": true, "model_version": "v0-fake"}

## GET /packages/{name}/releases
404 with code "not_tracked" if the package is not in our index.
200 with an empty array if tracked but not yet analysed.
{
  "package": "pandas",
  "releases": [
    {"version": "2.2.0", "released_at": "2024-01-19T00:00:00Z",
     "bump_type": "minor", "breakage_count": 5}
  ]
}

## GET /packages/{name}/releases/{version}/breakages?limit=20
limit: default 20, max 100.
Private symbols excluded unless ?include_private=true.
ranking is "model" when scores exist, "usage_fallback" when they don't.
{
  "package": "pandas",
  "version": "2.2.0",
  "model_version": "v0-fake",
  "ranking": "usage_fallback",
  "breakages": [
    {"symbol": "pandas.DataFrame.append", "kind": "OBJECT_REMOVED",
     "score": null, "explanation": "pandas.DataFrame.append was removed. 412 packages call it.",
     "user_count": 412}
  ]
}

## POST /analyze
body: {"requirements": "pandas==2.1.0\nrequests==2.31.0"}
Synchronous. Reads precomputed scores; never runs the model per request.
Aggregates breakages across all releases in (current, latest], deduplicated
by symbol keeping the highest score.
Input capped at 100 packages.
{
  "model_version": "v0-fake",
  "ranking": "usage_fallback",
  "results": [
    {"package": "pandas", "current": "2.1.0", "latest": "2.2.0",
     "breakages": [
       {"symbol": "pandas.DataFrame.append", "kind": "OBJECT_REMOVED",
        "score": null, "user_count": 412, "via_version": "2.1.2",
        "explanation": "..."}
     ]}
  ],
  "ignored": [
    {"line": "-r dev.txt", "reason": "unparseable"},
    {"line": "pandas", "reason": "no_version_pinned"}
  ]
}

## GET /stats
{"model_version": "v0-fake", "pr_auc": 0.0, "precision_at_10": 0.0,
 "ndcg_at_20": 0.0, "trained_at": "...", "baselines": {}}

## Decisions

1. Unknown package → 404 with code `not_tracked` (we can't tell "doesn't exist"
   from "not in our index" without a live PyPI call, so we don't claim to).
   Tracked package with no analysed releases → 200 with an empty array.
2. Unscored breakage → `score: null`. Fallback ordering is `user_count DESC,
   symbol_path ASC`, and the response carries `"ranking": "usage_fallback"`
   or `"model"` so the client knows which it got.
3. Private symbols excluded unless `?include_private=true`. Definition:
   a symbol is private if ANY dot-separated component starts with a single
   underscore, excluding dunders; membership in `__all__` overrides to public.
4. Pipeline stores structured facts in `breakage.detail` (JSONB). The API
   composes the display sentence and returns it as `explanation`. Wording is
   presentation, and changing presentation must not require re-running ingestion.
5. Unparseable/unusable requirements lines are skipped, returned in `ignored`
   with a reason: `unparseable`, `not_tracked`, `no_version_pinned`,
   `limit_exceeded`. Input capped at 100 packages.
6. `/analyze` is synchronous and reads precomputed scores only. The model is
   never loaded in the API process. Cache miss → null scores, usage fallback.
7. `/analyze` aggregates breakages across every release in `(current, latest]`,
   deduplicated by symbol keeping the highest score, with `via_version`
   naming the release that introduced each one.
8. The API serves the newest `model_run` by `trained_at`, resolved at startup
   and cached. `MODEL_VERSION` env var pins a specific version for rollback.