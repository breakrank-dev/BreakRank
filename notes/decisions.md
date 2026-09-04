# Engineering decisions

A running log of choices made on the web/backend side of BreakRank, with the
reasoning behind each one. Written at the time the decision was made, not
reconstructed afterwards.

The point of this file is that in April, when the report is being written and
someone asks "why Render?" or "why isn't that an ORM?", the answer is here with
a date on it instead of being guessed at four months late.

---

## 1 September 2026 — Day 1

### Deployment moved from week 6 to week 1

The original plan had deployment happening in week 6, after the model existed.
We moved it to week 1 and shipped a live URL on day one, with hardcoded
placeholder numbers on the page.

The project started two weeks behind schedule and the demo date (30 October)
can't move. Deployment is the risk that most commonly kills student projects,
because it's left until the end and then turns out to take days. Getting a
public HTTPS URL working immediately — even one showing invented data — removes
that risk entirely and means October can be spent on the model instead of on
infrastructure.

The principle: ship the pipe before the payload. Every layer of the
architecture should exist and talk to the next one before any of it does
anything useful.

### Migrated from Hugging Face Spaces to Render

The plan specified Hugging Face Spaces with the Docker SDK, on the free tier.
When we went to create the Space, Docker was marked as paid — Hugging Face moved
the Docker SDK behind a paid plan around July 2026, without an announcement or
a documentation update.

We switched the API to Render's free tier. Same code, one configuration change:
Render supplies the port via a `$PORT` environment variable rather than
requiring the hardcoded 7860 that Hugging Face expects.

The tradeoff we accepted: Render's free instances sleep after roughly 15 minutes
idle and take 30–60 seconds to wake. That matters for the demo, so the plan is
to ping the service shortly before presenting, and possibly add a scheduled
keep-alive job later. It's a known cost, not a surprise.

### Kept the Dockerfile even though Render doesn't use it

Render builds from a build command and a start command, so the Dockerfile in
`api/` is not part of the deployment path. We kept it anyway, updated to use
`$PORT` so the container behaves the same way the deployment does.

Reasons: it documents exactly how to run the API, it's an escape hatch if
Render's free tier changes the way Hugging Face's did, and it keeps the service
portable to any container host.

### Repository moved to a GitHub organisation

The repo was originally under one member's personal account. GitHub Apps
(Vercel, and later anything else) can only be installed by the account that owns
the repository, which meant every integration needed the owner to be online.

Moving to an organisation makes both members owners, so either can install
integrations independently. This matters most in November, when the GitHub bot
in Phase 6 is itself a GitHub App that will need installing and reinstalling
during development.

---

## 3 September 2026 — Day 2

### Numbered SQL migrations instead of a migration framework

The schema is six tables. We applied it with numbered `.sql` files
(`001_initial_schema.sql`, `002_indexes.sql`, ...) run by a small Python script,
rather than adding Alembic.

Six tables doesn't justify the dependency, and raw SQL is easier to read, audit
and explain than generated migration code. If the schema grows substantially,
revisiting this is cheap.

The runner records applied filenames in a `schema_migration` table, so running
it repeatedly is a no-op. That's the same idempotency principle used in the
ingestion jobs — one idea applied in two places.

### Idempotency enforced at the database level

`release` has `UNIQUE (package_id, version)`. `breakage` has
`UNIQUE (release_id, symbol_path, kind)`.

These aren't just data-hygiene constraints. The ingestion pipeline processes
thousands of packages and will be interrupted — by timeouts, rate limits, a
crashed job, a manual Ctrl+C. It has to be safe to re-run from the start.
Putting uniqueness in the schema means a re-run cannot silently create duplicate
rows, regardless of what the application code does or fails to do.

The database is the last line of defence for correctness. Constraints there hold
even when the code above them is wrong.

### `prediction` uses a composite primary key

`prediction` is keyed on `(breakage_id, model_version)` rather than having its
own `SERIAL id`.

One breakage can be scored by several model versions, and we want to keep all of
them — comparing v1 against v3 on the same rows is how we'll show the model
improved. The pair is the natural key, so making it the primary key is both
correct and free of an extra index.

### All timestamps are `TIMESTAMPTZ`

Release dates come from PyPI in UTC and we're developing in IST. Storing without
a timezone would mean "released yesterday" is quietly wrong by five and a half
hours, in a project where release ordering is the basis of the temporal
train/test split.

### Indexes: what exists and what deliberately doesn't

The hot query is: *given a package name and a version, return that release's
breakages joined to their model score and downstream usage count, ordered by
score, limited to 20.* Every index exists to serve that, or the homepage query.

Added explicitly:

- `breakage (symbol_path)` — for the join to `usage_index`
- `release (released_at DESC)` — for "most recent releases" on the homepage
- `breakage (release_id) WHERE is_private = FALSE` — a partial index, because
  almost every query excludes private symbols, and a partial index stays smaller
  and faster than a full one
- `prediction (model_version)` — the nightly scoring job filters on it

Deliberately absent: indexes on `package.name`, `release (package_id, version)`,
`usage_index.symbol_path`, and `prediction (breakage_id, model_version)`.
PostgreSQL creates indexes automatically to enforce `PRIMARY KEY` and `UNIQUE`
constraints, so declaring them again would waste storage and slow every write
for no benefit.

Performance was not measured today. With five seed rows PostgreSQL will scan the
table regardless of what indexes exist, so any `EXPLAIN ANALYZE` numbers would
be meaningless. This gets measured in week 5 against real data, with before and
after figures recorded here.

### Seed data added so the API can be built before real data exists

`db/seed.py` inserts one package, one release, and five breakages spanning the
full range: a high-usage removal (412 downstream callers), a mid one, a low one,
a private symbol with zero usage, and a parameter removal.

This is the same principle as deploying early. The API endpoints and the
frontend pages can be built and tested now, weeks before the ingestion pipeline
delivers anything, and every piece of real data that arrives later lights up
something that already works.

Every insert uses `ON CONFLICT`, so the seed script is safe to re-run. It is
clearly labelled as fake and gets removed before the demo.

---

## 3 September 2026 — API contract frozen

The contract lives in `docs/api-contract.md`. Eight questions came up while
writing it. All eight are decisions about what the system says when it doesn't
know something — and in every case the choice was between an honest answer and a
confident-looking wrong one.

**1. Unknown package returns 404 with the code `not_tracked`, not `not_found`.**

There are three different situations: the package doesn't exist on PyPI, it
exists but isn't in our index, or it's tracked but has no analysed releases yet.
We can't distinguish the first two without making a live PyPI call from a read
endpoint, which we won't do. So we don't claim to. `not_found` would assert the
package doesn't exist; `not_tracked` says only what we actually know.

The third case is genuinely different — the row exists — so it returns 200 with
an empty array. An empty collection is not the same as a missing resource.

**2. An unscored breakage returns `score: null`, and the list falls back to
ordering by usage.**

Until a model is trained, every score is null. Returning `0.0` would render as
"0% risk" in the UI, which is a lie — we don't know the risk, we haven't
measured it.

The consequence is that a "ranked" list with all-null scores isn't ranked at
all, so the fallback ordering is `user_count DESC, symbol_path ASC` (the second
term only for stable ordering). The response carries `"ranking": "model"` or
`"ranking": "usage_fallback"` so the client knows which it got.

This has a useful side effect: the product is genuinely useful before the model
exists, and that fallback ordering is exactly the baseline the model has to beat
in the evaluation. The comparison is built into the system rather than bolted on
at the end.

**3. Private symbols are excluded by default, with `?include_private=true` to
opt in — and the definition of "private" is written down.**

Roughly 40% of what griffe finds are internal symbols nobody outside the package
can call. Returning them by default would make the ranked list mostly noise,
which is the exact problem the project claims to solve. But they aren't deleted,
because the model needs them as negative training examples.

The definition matters more than the default. `is_private` is computed by the
pipeline and filtered on by the API, so if the two sides disagree about what it
means, roughly 40% of rows are wrongly shown or hidden and nobody notices for
weeks. The agreed rule:

> A symbol is private if any dot-separated component of its path begins with a
> single underscore, excluding dunder names. Membership in the module's
> `__all__` overrides this and marks it public.

"Any component" matters because `pandas.core._internals.Block` has a
public-looking leaf but sits under a private module. The `__all__` override
matters because a package that deliberately exports `_foo` has said it's public.

**4. The pipeline stores structured facts; the API composes the sentence.**

`explanation` is the human-readable line shown next to each row. The initial
plan had the pipeline writing it into the database. We changed that.

Wording is presentation, and presentation belongs to the layer that presents.
If the sentence lives in the database, rewording it — which we will want to do,
probably the week before the demo — means re-running ingestion across thousands
of releases instead of redeploying the API in two minutes.

So `breakage.explanation TEXT` became `breakage.detail JSONB` in migration 003.
The pipeline writes facts only the pipeline knows (which parameter changed, what
griffe's own message said, whether it was deprecated in an earlier release), and
the API renders the display string from `kind`, `symbol_path`, `user_count` and
`detail`.

JSONB rather than more columns because the useful facts differ by breakage kind
— a `PARAMETER_REMOVED` has a parameter name, an `OBJECT_REMOVED` doesn't.
Typed columns for every possible field would be mostly nulls. The rule we
settled on: queryable attributes are typed columns, the variable payload is
JSONB.

**5. Unparseable requirements lines are skipped and reported with a reason.**

Real `requirements.txt` files contain `-r dev.txt`, `-e .`, git URLs, comments,
and unpinned packages. Rejecting the whole file over two odd lines means a user
pastes a real file, gets an error, and leaves.

There are four distinct reasons a line produces no result, and lumping them
together loses the meaning: `unparseable`, `not_tracked`, `no_version_pinned`,
`limit_exceeded`. Each ignored line is returned as an object with its reason.

Input is capped at 100 packages. A public endpoint with no input bound is a
denial-of-service waiting to happen on a free tier.

**6. `/analyze` is synchronous, and the model never runs in the API process.**

Scores are precomputed nightly, so a 40-package request is one database query
with an `IN` clause — well under 100ms. Async would add complexity and buy
nothing; the real latency is Render's cold start, which async doesn't fix.

The corollary is the load-bearing part: on a cache miss (a version pair with no
precomputed prediction), the API returns null scores with usage fallback
ordering rather than loading the model and scoring on demand. This means the API
never imports LightGBM, never loads a model file, and never holds model memory
on a free tier. Its only job is to read from PostgreSQL and shape JSON.

**7. `/analyze` aggregates across intermediate releases, not just current to
latest.**

Someone on `pandas 2.1.0` upgrading to `2.2.0` passes through `2.1.1`, `2.1.2`
and `2.1.3`. Breaking changes hide in patch releases — that is the entire
premise of the project. Diffing `2.1.0` against `2.2.0` directly would miss
anything introduced and then re-exported, and would contradict our own thesis.

So the response aggregates every release in the range `(current, latest]`,
deduplicated by symbol keeping the highest score, with a `via_version` field
naming the release that introduced each change.

This affects the pipeline too: it needs a breakage row per consecutive release
pair, not one row per package.

**8. The API serves the newest model, with an environment variable to pin an
older one.**

`prediction` holds scores from every model version, so the API has to choose.
It resolves the newest `model_run` by `trained_at` once at startup and caches
it. A `MODEL_VERSION` environment variable overrides that.

The override costs five minutes and means that if a retrain goes badly the night
before the demo, we set the variable on Render and roll back in thirty seconds
without touching code or data.

---

## Notes for later

- Narrow CORS from `allow_origins=["*"]` to the Vercel domain before the demo.
- Add an Ignored Build Step on Vercel so ML branches don't trigger frontend
  builds and generate failure emails.
- Measure the hot query with `EXPLAIN ANALYZE` in week 5, once there's real
  data, and record before/after index figures here.
- Remove the seed data before the demo.
