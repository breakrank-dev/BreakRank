# BreakRank

**Ranks Python dependency breaking changes by real-world impact.**

Rule-based tools flag around 200 API changes per major release and about five
of them actually matter to anyone. BreakRank ranks them, trained on labels
derived automatically by statically analysing 1,500 downstream packages —
no hand labelling.

> 🚧 **In development.** Started 1 September 2026. Live demo target: 30 October 2026.

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
is accurate and genuinely useful. The problem is that it finds ~200
differences per major release and roughly five of them matter. When a tool
warns you two hundred times, you stop reading it.

**BreakRank is the ranking layer on top.**

## How it works

```
Track A: top 300 packages, every version
         → download sdists → griffe → candidate changes
Track B: top 1,500 packages, latest version
         → AST scan → which symbols the ecosystem actually uses

A + B → automatic labels → features → LightGBM ranker → live API
```

The label is computed, not hand-written: *does any downstream package in the
top 1,500 import or call this symbol?* That is a **proxy** for "this broke
someone", not the same thing — see Limitations.

## Status

| Milestone | Target | Done |
|---|---|---|
| 5,000 real API-change rows | 13 Sep 2026 | ☐ |
| 20,000 labelled rows + model beating baselines | 27 Sep 2026 | ☐ |
| Public live URL | 11 Oct 2026 | ☐ |
| `requirements.txt` → ranked risk list | 25 Oct 2026 | ☐ |
| v1 demo ready | 30 Oct 2026 | ☐ |

## Running the pipeline locally

```bash
git clone https://github.com/YOURUSERNAME/breakrank.git
cd breakrank
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/day1_check.py       # verify the environment
python ml/ingest/packages.py       # step 1: the package list
python scripts/day1_demo.py        # end-to-end on 6 packages → data/day1_changes.csv
```

macOS note: LightGBM needs Apple's OpenMP runtime — `brew install libomp`
before `pip install`.

## Limitations

*(Filled in properly by week 11 — this section is not optional and not a
weakness. Write it honestly.)*

- The training label is a proxy: "a downstream package references this
  symbol", not "this change actually broke a build".
- Coverage is limited to packages that publish source distributions.
- Python only, for now.

## Credit where it is due

The candidate changes come from [**griffe**](https://mkdocstrings.github.io/griffe/),
an existing open-source tool by the mkdocstrings project. BreakRank does not
detect API changes — griffe does that, and does it well. BreakRank's
contribution is deciding **which of those changes matter**.

## Licensing

Package sources are downloaded from PyPI, analysed, and discarded; only
derived signatures and counts are stored. No third-party source code is
redistributed. This project is MIT licensed — see [LICENSE](LICENSE).

---

Final-year AI/ML project · 2 people · built Aug 2026 – Apr 2027
