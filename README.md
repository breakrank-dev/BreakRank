# BreakRank

**Ranks Python dependency breaking changes by real-world impact.**

A tool such as griffe can list every API change of a release: 134 for redis
8.0.0 alone. Most of them concern names a given project never uses. BreakRank
ranks them, and checks its ranking against labels derived automatically by
statically analysing 1,500 downstream packages — no hand labelling.

- **Website:** https://break-rank.vercel.app
- **API docs:** https://breakrank.onrender.com/docs

> **Status, 28 November 2026.** Started 1 September 2026. The site and the API are live, and the
> command-line fixer `breakrank-fix` 0.1.0 is in this repository. See Status and Limitations below.

---

## The problem

Your project imports forty libraries you did not write. They get updated.
Sometimes an update deletes a function, or makes an optional parameter
required, and your code breaks at 3 a.m. with a red CI run and nobody
knowing why.

[Dependabot](https://docs.github.com/code-security/dependabot) tells you a new
version exists. It does not tell you whether that version will break you.

[griffe](https://mkdocstrings.github.io/griffe/) goes further — give it two
versions of a library and it lists every difference in the public API. That
is accurate and genuinely useful. The problem is the volume: 134 differences
for redis 8.0.0, and 2,296 for one release of sglang. In our data, 3.6% of the
changes (605 of 16,580, in releases before 28 July 2026) touch a name that any
scanned package uses. When a tool warns you two hundred times, you stop
reading it.

**BreakRank is the ranking layer on top.**

## How it works

```
Track A: the 500 most-downloaded packages, their newest releases
         → download sdists → griffe → candidate changes
Track B: top 1,500 packages, latest version
         → AST scan → which symbols the ecosystem actually uses

A + B → automatic labels → rankers compared on a time split → live API
                                                            → offline fixer
```

The label is computed, not hand-written: *does any downstream package in the
top 1,500 import or call this symbol?* That is a **proxy** for "this broke
someone", not the same thing — see Limitations.

**What ships:** each release's changes are listed by how short their public
import path is, shortest first. On releases published before 28 July 2026,
that rule beats a popularity baseline at every evaluation date, by 2.5x to
4.8x. A learned ranker (LightGBM LambdaRank) did not beat the rule, so the
rule ships. Changes at the same depth are listed in no particular order.
Releases from 28 July 2026 on are held back, unseen, for the final evaluation.

## Using it

- **One release:** pick a package and version on the website (or one of the
  presets) to see its breaking changes in order, with how many of the ~1,500
  scanned packages import each one.
- **A whole project:** paste a `requirements.txt` on the
  [/analyse](https://break-rank.vercel.app/analyse) page. For every pinned
  (`==`) package, you get the changes between your version and the newest
  release BreakRank holds, release by release.
- A release with no changes says why: analysed and nothing found, can't
  confirm it was compared, no source code, and so on.
- **In your own code, offline:** `breakrank-fix` reads a project, lists the
  lines that use a changed name in the same order as the site, and prints the
  mechanical fixes as a diff. Nothing is written unless you add `--apply`.

```bash
pipx install .                                 # from a clone of this repository; Python 3.11 or later
breakrank-fix redis --to 8.0.0 --env .venv     # run inside the project; --env is its virtual environment
breakrank-fix --list                           # the packages the tool holds
```

The tool carries its data with it (21,452 changes of 3,204 releases of 351
packages) and needs no network. It rewrites two kinds of change, a moved
parameter and a changed default, and only flags the rest. Read Limitations
before trusting a rewrite.

## Status

| Milestone | Target | Done |
|---|---|---|
| 5,000 real API-change rows | 13 Sep 2026 | ☑ |
| 20,000 labelled rows + model beating baselines | 27 Sep 2026 | ☑ |
| Public live URL | 11 Oct 2026 | ☑ |
| `requirements.txt` → ranked risk list | 25 Oct 2026 | ☑ |
| v1 demo ready | 30 Oct 2026 | ☑ |

The first two were cleared on 5 Sep 2026; the learned model was later beaten by the path-length rule, which ships (NOTES §36).
The demo checkpoint was met on 28 Oct 2026: `breakrank-fix` installed from a fresh clone and run with the network off (NOTES §38.17).
The fixer was then evaluated on ten open-source projects, and the reading of 230 of its calls by hand ended on 26 Nov 2026 (NOTES §38.18).

## Running the pipeline locally (Vaibhav)

```bash
git clone https://github.com/breakrank-dev/BreakRank.git
cd BreakRank
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/day1_check.py       # verify the environment
python ml/ingest/packages.py       # step 1: the package list
python scripts/day1_demo.py        # end-to-end on 6 packages → data/day1_changes.csv
```

macOS note: LightGBM needs Apple's OpenMP runtime — `brew install libomp`
before `pip install`.

## Running the API, website and tests locally (Varad)

On Windows, use Git Bash. You need Python 3.12 and Node.js.

**API**
```bash
python -m venv .venv
source .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env               # then put your DATABASE_URL in .env
uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000/docs.

**Website**
```bash
cd web
npm install
cp .env.example .env.local         # points the site at the live API
npm run dev
```
Open http://localhost:3000.

**Tests**
```bash
python -m pytest
```
The tests read from the database in `DATABASE_URL` and use the test package
`breakrank-fixture`. CI runs them against a throwaway Postgres instead (see
`.github/workflows/ci.yml`).

**Database migrations**
```bash
python -m db.apply_migrations
```
Each file in `db/migrations/` runs once, in order. This changes the database,
so only run it when you mean to.

## Limitations

- The training label is a proxy: "a downstream package references this
  symbol", not "this change actually broke a build".
- Usage counts come from import statements, so the scan sees module-level
  symbols far better than methods called on objects. A count of zero means
  "not seen in the scan", not "nobody uses it".
- Coverage is limited to packages that publish source distributions.
- Python only, for now.
- 149 of the 500 packages failed in the ingest for lack of memory, and the
  usage scan covers the latest release of about 1,500 packages only.
- The ranking figures are measured on releases before 28 July 2026. Later
  releases are held back for one final evaluation, so no figure on unseen
  data is given yet.
- The path-length rule cannot order changes at the same depth, and it fails
  for a name that is deep but widely used. The cryptography 47.0.0 preset on
  the site shows such a case.
- `breakrank-fix`: in all 230 calls that were read by hand, the tool had
  named the right library function. Its rewriting is another matter: of 200
  rewrites read, 6 were needed, 164 changed nothing, 19 changed the meaning
  of the call and 11 would fail under the new version (NOTES §38.18). It only
  prints a diff, and nothing is written without `--apply`.

## Credit where it is due

The candidate changes come from [**griffe**](https://mkdocstrings.github.io/griffe/),
an existing open-source tool by the mkdocstrings project. BreakRank does not
detect API changes — griffe does that, and does it well. BreakRank's
contribution is deciding **which of those changes matter**.

## Licensing

Package sources are downloaded from PyPI, analysed, and discarded; only
derived signatures and counts are stored. No third-party source code is
redistributed. This project is MIT licensed — see [LICENSE](LICENSE).

## Deployment

- Website: Vercel (deploys from `main`)
- API: Render free tier (deploys from `main`)
- Database: PostgreSQL on Neon

Originally planned for Hugging Face Spaces. Hugging Face moved the Docker
SDK to a paid plan in July 2026, so the API was migrated to Render.
A Dockerfile is kept in the repo root for local container runs and portability.

---

Final-year AI/ML project · 2 people · built Aug 2026 – Apr 2027