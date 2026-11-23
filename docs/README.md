# Where to look

A map of this folder, for someone who has to check a claim rather than
read six thousand lines. The README at the repo root says what BreakRank
is; this page says where each thing it says is justified.

## The files

| File | What it is | Read it when |
|---|---|---|
| [`NOTES.md`](NOTES.md) | The findings notebook: 35 dated sections, every number with the run that produced it. The pipeline side's lab book. | You need to know *how we know* something. |
| [`../notes/decisions.md`](../notes/decisions.md) | Engineering decisions on the web and database side, numbered, with reasoning written at the time. | Someone asks "why Render?", "why no ORM?", "why 404 and not an empty list?". |
| [`api-contract.md`](api-contract.md) | The agreement between the pipeline and the API, frozen 3 Sep 2026, plus decisions 9 to 14. Parts of the endpoint sketches were superseded by later decisions (see 17 in `decisions.md`); the column definitions still hold. | You need the meaning of a shared column, or the privacy rule. |
| [`pipeline-changes-required.md`](pipeline-changes-required.md) | The summary of the contract freeze sent to the pipeline side. | Historical. |
| [`DAY1_GUIDE.md`](DAY1_GUIDE.md), [`TEAMMATE_DAY1.md`](TEAMMATE_DAY1.md), [`what-exists-till-day2.md`](what-exists-till-day2.md), [`DATES.md`](DATES.md) | Setup guides and the original schedule from the first week. | Onboarding, or reconstructing the timeline. |

The pipeline's own frozen vocabulary is in code: `ml/contract.py` and
`ml/holdout.py`. The scripts named below live in `scripts/` and
`ml/model/`.

## How to read NOTES.md

Three rules, all stated in the file itself but easy to miss.

1. **The latest "For the report" block wins.** The notebook never rewrites
   history; it appends. Several sections carry superseded figures and say so
   at the top. As of 10 Oct 2026 the current wording for the report, the
   slides, the site and the demo is in **§35.2**. §1 and the preamble were
   last rewritten in September and quote older numbers.
2. **No lift is a single number.** Every headline is a range over seven cut
   dates fixed in advance (§5.6, §11.2). Quote the range, or the median with
   the range beside it.
3. **PR-AUC is quoted with its floor.** The floor is the positive rate, not
   0.5 (§5.5, decision 13). Since 10 Oct, PR-AUC averages over tied scores
   (§35); numbers from before that are on scikit-learn's convention and the
   text names which.

## Where each headline claim comes from

Claims as the root README and the live site state them, and the section
that produced each one.

| Claim | Section | Produced by |
|---|---|---|
| Ranking by shortest public import path beats a popularity baseline at every one of seven cut dates, by 2.5x to 4.8x | §34.2, wording §35.2 | `scripts/path_verdict.py` on the sweep of §33.3 |
| The learned ranker (LightGBM LambdaRank) does not beat that rule, so the rule ships | §34.1 (the rule, fixed in advance), §34.2 (the verdict) | `scripts/path_verdict.py`; the ablation that pointed there is §30.3 and §32.3 |
| The model beats popularity at all 13 cut dates, 2.1x to 6.6x, each interval above 1x | §35.1, §35.2 | `ml/model/stability.py`, `scripts/compare_sweeps.py` |
| Single-split numbers: PR-AUC 0.289 against a floor of 0.079; precision@10 0.282 over 17 upgrades; nDCG@20 0.595 over 11 | §35.2 | `ml/model/train.py` |
| Releases from 28 July 2026 are held back, unseen, for the final evaluation | §22.1 (what is sealed), §22.2 (why that date) | `ml/holdout.py`, `scripts/holdout_boundary.py`, `scripts/test_holdout.py` |
| The usage scan sees module-level symbols far better than methods (12.34% vs 1.12% positive) | §11.6, contract decision 14 | `scripts/label_blindspot.py` |
| The label is a proxy: "a downstream package imports this", not "this broke a build" | §4, §21.1, §31.2 coda | — |
| Candidates come from griffe; ~1,500 downstream packages scanned; top 500 by downloads attempted, 314 produced rows | §20 | `ml/ingest/run_ingest.py`, `ml/ingest/run_usage.py` |
| Kill-date gate (20,000 labelled rows and a model beating the version-number baseline) cleared 5 Sep | §1 | — |
| "Analysed and found nothing" is a real answer: `analysed_clean`, `no_baseline`, `unknown` | §13, §14, migrations 005 to 007 | `ml/db.py` |
| Version-string and package-metadata changes are not changes | §23 (F1), §32 (F39), §33 (narrowed, and moved into the pipeline); the 15 Oct database cleanup is decision 15 onwards in `decisions.md` | `ml/features/build.py`, `ml/db.py` (the loader shares the rule) |
| One change counted once, not once per inheriting class (`inherited_by`) | §10.2 to §10.5 | `ml/ingest/api_extract.py` |
| Deprecated symbols were used downstream *more* often, not less; modules with 21+ earlier breaks are almost never used | §26.4 (current wording), §19 (first measured) | `scripts/item4_findings.py` |
| A logistic regression on the same features does nearly as well as the trees | §29.3, §30.3 | `scripts/item7_linear.py` |
| Which model the site serves, and how to check | §28.5, decision 8 | `GET /health`, `scripts/db_model_check.py` |
| The demo story (redis 7.3.0 to 7.3.1) and the failure story (cryptography 47.0.0, `PBKDF2HMAC` ranked 27th) | §35.2, §31.2 | `scripts/stories.py` |
| The pipeline never executes downloaded code | §17, §18 | `scripts/test_no_execution.py` |

## Sections by theme

- **Data pipeline and its bugs:** §2, §3, §9 (alias resolver), §10 (sdist
  layout), §12 (the analysed window), §17 to §18 (no execution).
- **Labels:** §4, §11.6, §21.1, §24 (`label_alias` ships).
- **Model and evaluation method:** §5, §11, §15, §25 (graded relevance,
  rejected), §27 (intervals), §29 to §30 (linear baseline, tuned trees),
  §34 to §35 (path length, tie handling).
- **Holdout:** §22.
- **Database and what the site serves:** §8, §13 to §14, §28; the October
  cleanup of the live table is in `decisions.md`, 15 to 25.
- **Audit and fix list:** §21 (the list), §22.4, §22.5, and the "still open"
  subsections at the end of §28 to §30.
