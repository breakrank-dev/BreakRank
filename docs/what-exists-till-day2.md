# What exists so far, and why

A plain-language walkthrough of everything built on Day 1 and Day 2, written so
that each piece makes sense on its own and the next piece of work has somewhere
to attach.

If you read only one thing: **the whole system is already connected end to end.
Nothing real flows through it yet.** Every remaining task is replacing a fake
part with a real one, in a path that already works.

---

## 1. The shape of the whole thing

BreakRank answers one question: *if I upgrade this Python package, what will
break, and which of those things actually matter to me?*

When you upgrade a package, tools can already tell you that 187 things changed
in its API. Almost all of those 187 are irrelevant — internal renames, private
helpers, things nobody calls. Five of them will break real code. BreakRank
exists to find those five and put them at the top.

Two people build it. The split is along a database:

```
Vaibhav's side                    Your side
──────────────                    ─────────
downloads packages from PyPI
runs griffe to find API changes
scans 1,500 packages for usage
trains the ranking model
writes scores nightly
        │
        ▼
   ┌─────────────┐
   │  PostgreSQL │  ◄──── the boundary
   └─────────────┘
        │
        ▼
                                  FastAPI reads it
                                  Next.js displays it
                                  scheduled jobs run it
```

Neither of you calls the other's code. He writes rows; you read rows. That's why
the schema and the API contract mattered so much on Day 2 — they *are* the
interface between two people working in parallel.

---

## 2. What runs where

Four services, all free, each doing one job.

| Piece | Where it runs | What it does |
|---|---|---|
| Frontend | **Vercel** | The website people see |
| API | **Render** | Answers questions about the data |
| Database | **Neon** | Stores everything |
| Scheduled jobs | **GitHub Actions** | Runs the nightly pipeline (not built yet) |

A request travels like this:

```
your browser
   → Vercel (serves the page)
   → Render (the page asks the API a question)
   → Neon (the API asks the database)
   → back up the chain
```

All four links exist and work right now. That's the thing worth understanding —
the hard part of most student projects is getting that chain to exist at all,
and it's done.

---

## 3. Why each technology

### Next.js (frontend)

A framework for building websites in React. You describe what a page should look
like; it handles the rest.

Chosen because it deploys to Vercel with zero configuration — the two are made
by the same company. Given a fixed deadline, "deploying takes one click" is worth
more than any feature comparison.

### TypeScript

JavaScript with type checking. If the API returns `score: null` and the page
tries to display it as a number, TypeScript complains while you're writing the
code rather than the page breaking in front of an examiner.

### Tailwind CSS

Styling written as class names directly in the markup — `text-5xl font-bold`
instead of a separate stylesheet. Faster for someone who isn't a designer, and
your schedule has three weeks of UI work you'd rather compress into one.

### FastAPI (backend)

A Python web framework. You write a normal function, add a decorator above it,
and it becomes a web endpoint.

Three reasons it fits:

- **Python.** The pipeline is Python. One language across the project means one
  environment, one set of tools, and Vaibhav can read your code.
- **Automatic validation.** You declare that `limit` is an integer between 1 and
  100. Anything else is rejected before your function runs. You don't write that
  check.
- **Automatic documentation.** Visiting `/docs` gives an interactive page listing
  every endpoint with its exact response shape. That's generated from your code,
  so it can't drift out of date — which makes it the honest version of the API
  contract.

### Uvicorn

The program that actually listens on a port and runs FastAPI. FastAPI describes
what to do with a request; Uvicorn is what receives it.

### PostgreSQL, on Neon

A relational database. Chosen over the alternatives because the data is
genuinely relational — packages have releases, releases have breakages,
breakages have scores — and because it enforces rules you don't want to trust
application code with.

Neon specifically because its free tier is a real PostgreSQL database over the
internet, with no credit card. Both of you connect to the same one.

### SQLAlchemy Core

The Python library that talks to PostgreSQL. Deliberately **Core**, not the ORM.

An ORM hides SQL behind Python objects. Convenient, but you end up unable to
explain the query your code generated. With Core you write the SQL yourself, so
when someone asks what your main query does and why it's indexed that way, you
know — because you wrote it.

### Render (API hosting)

The plan was Hugging Face Spaces. Hugging Face moved Docker-based Spaces to a
paid plan around July 2026, so the API moved to Render's free tier instead.

The cost: free instances sleep after about 15 minutes idle and take 30–60
seconds to wake. Manageable, but it's why the first page load after a quiet
period is slow, and why you'll want to warm it up before demoing.

---

## 4. Day 1 — the pipe

The idea behind Day 1 was **ship the pipe before the payload.**

The original schedule had deployment in week 6. Deployment is the thing that
kills student projects, because it's left until last and then turns out to take
days. So it got moved to day one, with a page showing an invented number.

Now every real feature is added to a path that already works, instead of being
built in the dark and deployed at the end and hoping.

### What was built

**The repository skeleton.** Folders for API code, ML code, tests, and scheduled
jobs. Empty, but the shape agreed in advance so two people don't invent
conflicting structures.

**`.gitignore`.** A list of files git must never record. Two entries matter:

- `.env` — holds your database password. Committing it to a public repository
  means bots find it within minutes.
- `.venv/` — your installed Python packages, hundreds of megabytes, rebuildable
  from `requirements.txt`.

**A virtual environment (`.venv`).** A private copy of Python for this project.
Without it, packages install system-wide and two projects wanting different
versions of the same library conflict. This is why every terminal session starts
with the activate command — you're stepping into that private copy.

**`.env` and the database connection.** The connection string lives in a file
git ignores. The code reads it at runtime. Nothing secret is ever in the
repository.

The connection string needed two adjustments Neon doesn't give you:

- `+psycopg` inserted after `postgresql` — tells SQLAlchemy which driver to use
- `pool_pre_ping=True` — Neon closes idle connections; this checks a connection
  is alive before using it, instead of failing with a confusing error

**The `/health` endpoint.** Returns `{"ok": true, "model_version": "v0-fake"}`.

It does nothing useful, and it's important. It's how the frontend proves it can
reach the API, how Render knows the service started, and how you'll know in
October whether a problem is the API or something else. Every real service has
one.

The `"v0-fake"` is deliberate. Anything showing invented data says so.

**A deployed frontend and API, talking to each other.** The page fetches
`/health` and displays the result. Small, and it proves the entire chain.

**CORS.** Browsers block a page on one domain from calling a different domain.
Your page is on `vercel.app` and your API is on `onrender.com` — different
origins, so the API has to explicitly say the frontend is allowed. Without this
the browser silently refuses the request.

Currently set to allow any origin, which is fine while developing. Narrow it to
your Vercel domain before the demo.

### Two workflow habits started

**Branches and pull requests.** Every change happens on its own branch, then
gets proposed as a pull request before joining `main`. With two people this
prevents overwriting each other. It also leaves a readable trail — someone
opening the repository sees a sequence of described changes rather than sixty
commits saying "update".

**The decisions log.** `notes/decisions.md` records each choice and *why*, on the
day it was made. In April, writing the report, "why Render?" has a dated answer
instead of a guess.

---

## 5. Day 2 — the database and the contract

### The six tables

| Table | One row per | Filled by |
|---|---|---|
| `package` | tracked PyPI package | pipeline |
| `release` | version of a package | pipeline |
| `breakage` | API change found in a release | pipeline |
| `usage_index` | symbol, with how many packages call it | pipeline |
| `model_run` | trained model, with its scores | pipeline |
| `prediction` | one breakage's score under one model | pipeline |

Every table is written by Vaibhav and read by you.

The two central tables:

**`breakage`** is one detected API change — `pandas.DataFrame.append` was
removed in version 2.2.0. This is the raw material. There will be roughly 20,000
of these rows.

**`usage_index`** is the other half, and it's the interesting one:
`pandas.DataFrame.append` → 412 packages call this. Built by scanning 1,500
packages' source code.

The project's central claim lives in the join between those two tables. A change
alone tells you nothing. A change plus its usage count tells you whether it
matters.

### Design decisions worth being able to explain

**Uniqueness constraints make re-running safe.**

`release` has `UNIQUE (package_id, version)`. `breakage` has
`UNIQUE (release_id, symbol_path, kind)`.

The pipeline processes thousands of packages and *will* be interrupted —
timeouts, rate limits, a crash, a Ctrl+C. It has to be safe to start again from
the beginning. These constraints mean a re-run cannot create duplicate rows, no
matter what the application code does.

The principle: the database is the last line of defence. Rules there hold even
when the code above them is wrong.

**`ON DELETE CASCADE`.** Delete a package and its releases go, then their
breakages, then those predictions. No orphaned rows pointing at nothing.

**`prediction` is keyed on `(breakage_id, model_version)`, not its own id.**

One breakage can be scored by several models, and you want to keep all of them —
comparing v1 against v3 on the same rows is how you show the model improved. The
pair is the natural key.

**Everything is `TIMESTAMPTZ`, never `TIMESTAMP`.**

Release dates come from PyPI in UTC; you're in IST. A timestamp with no timezone
is wrong by five and a half hours — in a project where release *ordering* is the
basis of the train/test split.

**Indexes serve one query.**

An index is a lookup structure that makes finding rows fast, at the cost of
slightly slower writes and some storage. You don't add them everywhere; you add
them for the queries you actually run.

The query that matters:

> Given a package name and version, return that release's breakages, joined to
> their score and usage count, ordered by score, limited to 20.

That sentence justifies every index. Four were added; several obvious ones were
deliberately *not*, because PostgreSQL creates indexes automatically for
`PRIMARY KEY` and `UNIQUE` constraints. Adding them again wastes space and slows
writes for nothing.

Knowing what you *didn't* index, and why, is the better half of that answer.

### Migrations

Changes to the database are numbered SQL files — `001_initial_schema.sql`,
`002_indexes.sql`, `003_breakage_detail.sql` — applied by a script that records
which have run in a `schema_migration` table.

Two reasons this exists rather than clicking around in Neon's editor:

- **Reproducibility.** Anyone can build the same database from an empty one by
  running the files in order.
- **History.** The `.sql` files are in git, so schema changes are reviewable like
  code.

Migration 003 already proved its worth. On Day 2 the `explanation` column was
dropped and replaced with `detail`. That change is now a permanent, dated,
explained record rather than something someone did once and forgot.

### Seed data

`db/seed.py` inserts one package, one release, and five breakages — a
high-usage removal, a mid one, a low one, a private symbol with zero usage, and
a parameter removal.

It's fake, and it's the point. The API and the frontend can be built and tested
now, weeks before the pipeline produces anything. Same idea as deploying early:
every piece of real data that arrives later lights up something that already
works.

Written to be safe to run twice — the same idempotency discipline as the
pipeline itself.

### The API contract

`docs/api-contract.md` states the exact shape of every endpoint's response,
agreed before either side builds against it.

Without it, you'd guess what the pipeline produces, he'd guess what the API
needs, and the mismatch would surface in October. The contract is "frozen",
meaning it doesn't change without a conversation — not that it can't change.

Writing it surfaced eight questions. Every one turned out to be the same
question in different clothes: **what does the system say when it doesn't
know?**

| # | Situation | Decision |
|---|---|---|
| 1 | Package not in our index | 404 `not_tracked` — not `not_found`, because we can't tell "doesn't exist" from "we don't have it" |
| 2 | No model trained yet | `score: null`, order by usage instead, and say `ranking: "usage_fallback"` |
| 3 | Private symbols | hidden by default, opt in with a query parameter, definition written down |
| 4 | Who writes the display sentence | the API, not the pipeline |
| 5 | Unusable requirements lines | skip and report with a reason, don't reject the file |
| 6 | Is `/analyze` slow enough for async | no — scores are precomputed, the model never runs in the API |
| 7 | What does `/analyze` compare | every release between current and latest, not just the endpoints |
| 8 | Which model version to serve | newest, with an env variable to pin an older one |

Two are worth understanding properly, because they shape a lot of later work.

**Decision 2 — null scores.** For the next several weeks no model exists, so
every score is null. Returning `0.0` would display as "0% risk", which is a lie
— you don't know the risk, you haven't measured it. So `null`, and the list
falls back to ordering by usage count.

Useful side effect: the product works before the model does. And that fallback
ordering is exactly the baseline the model has to beat, so the comparison is
built into the system rather than bolted on at evaluation time.

**Decision 4 — where the sentence is built.** `explanation` is the readable line
in the UI: *"pandas.DataFrame.append was removed. 412 packages call it."*

Originally the pipeline was going to write that sentence into the database. It
was moved to the API, because wording is presentation — and if the sentence is
stored, rewording it means re-running ingestion over thousands of releases
instead of redeploying the API in two minutes. You *will* want to reword it,
probably the week before the demo.

So the pipeline stores facts (`detail`, as JSONB), and the API builds the
sentence from them. The rule: **queryable attributes are typed columns, the
variable payload is JSONB.**

This is also why `breakage.explanation` became `breakage.detail` on Day 2 — a
decision that changed a table built the previous day. Cheap now, expensive in
November.

---

## 6. Where things stand

**Built and working:**

- Live website on Vercel
- API on Render, answering `/health`
- PostgreSQL on Neon, six tables, indexed, with migrations
- Fake data to build against
- A frozen contract covering all five endpoints
- Branch-and-PR workflow, decisions log

**Not built yet:**

- Any real endpoint (Day 3)
- Any page that shows real data (Day 4)
- Tests, CI (Day 3)
- The ingestion pipeline, usage index, labels, model (Vaibhav)
- Scheduled jobs
- The `/analyze` upload flow
- The GitHub bot (Phase 6)

Against the original schedule, two weeks were lost to a late start and roughly
all of them have been recovered — deployment that was planned for week 6 is
already done.

---

## 7. The one idea underneath all of it

Almost every choice on Day 1 and Day 2 comes from the same instinct: **be honest
about what you don't know, and build the connection before the content.**

- Deploy with fake numbers, but label them fake
- Return `null` for an unscored breakage instead of `0.0`
- Return `not_tracked` instead of `not_found`
- Fall back to usage ordering and *tell the client* that's what happened
- Skip unparseable lines but report which ones and why

That's the same instinct behind the project itself. The pitch isn't "we detect
everything" — it's "detection is solved, and the honest hard part is knowing
which detections matter." A system built to say what it doesn't know is a system
whose answers you can trust when it does know.

Worth carrying into every remaining decision.
