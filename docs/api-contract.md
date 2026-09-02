# BreakRank API Contract — DRAFT, pending sign-off

Base URL: https://<render-service>.onrender.com
All timestamps: ISO 8601, UTC, e.g. "2024-01-19T00:00:00Z"
All errors: {"error": {"code": "not_found", "message": "..."}}

## GET /health
{"ok": true, "model_version": "v0-fake"}

## GET /packages/{name}/releases
404 if the package is unknown.
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
score is null when no model has scored it yet.
{
  "package": "pandas",
  "version": "2.2.0",
  "model_version": "v0-fake",
  "breakages": [
    {"symbol": "pandas.DataFrame.append", "kind": "OBJECT_REMOVED",
     "score": 0.94, "explanation": "...", "user_count": 412}
  ]
}

## POST /analyze
body: {"requirements": "pandas==2.1.0\nrequests==2.31.0"}
Synchronous. Reads precomputed scores; never runs the model per request.
Unparseable lines are skipped and returned in "ignored".
{
  "results": [
    {"package": "pandas", "current": "2.1.0", "latest": "2.2.0",
     "breakages": [...]}
  ],
  "ignored": ["-r dev.txt"]
}

## GET /stats
{"model_version": "v0-fake", "pr_auc": 0.0, "precision_at_10": 0.0,
 "ndcg_at_20": 0.0, "trained_at": "...", "baselines": {}}

 ## Open questions for Vaibhav

Proposed answers in brackets. Confirm or push back, then this becomes frozen.

1. Unknown package name → 404, or 200 with empty list? [404]
2. No model scored yet → score: null, frontend shows "not scored"? [yes]
3. Private symbols excluded by default, opt-in via ?include_private=true? [yes]
4. Who writes `explanation` — pipeline or API? [pipeline]
5. Unparseable requirements lines → skip and report in "ignored"? [skip]
6. /analyze stays synchronous since scores are precomputed? [yes]

Frozen on: ____