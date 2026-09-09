# Findings notebook

Everything BreakRank learned by measuring rather than assuming, with the
numbers that back it. Written for two readers: me in November when the
report is due, and an examiner asking "how do you know?"

Rule for this file: **no claim without the number that produced it.** If a
line here says something is true, the run that showed it is named.

Last updated after the Day-6 alias run. Dataset: **22,914 breakage rows
across 406 packages**, top 500 PyPI by download count, 6 releases each.

---

## 1. Where the project stands

Three labels now, on the same rows (§9). `label_alias` is the shipped one.

| | strict `label` | `label_scoped` | **`label_alias`** |
|---|---|---|---|
| rule | exact path match | exact, or one symbol owns the leaf | exact, or a real export path |
| basis | fact, and incomplete | **a guess** | fact, from griffe's alias graph |
| positive rows | 611 (2.67%) | 1,074 (4.69%) | **1,003 (4.38%)** |
| test positives | 98 | 195 | **198** |
| test positive rate — **the PR-AUC floor** | 0.0172 | 0.0343 | **0.0348** |
| ranker PR-AUC | 0.0981 | 0.2299 | **0.3301** |

Full measured result for the shipped model, 9 Sep run:

| | value | vs floor |
|---|---|---|
| floor — test positive rate | 0.0348 | 1.0× |
| semver baseline (the kill-date gate) | 0.0389 | 1.1× |
| best baseline — popularity | 0.1057 | 3.0× |
| **ranker PR-AUC** | **0.3301** | **9.5×** |
| precision@10 | 0.3313 | over 16 rankable pairs — see §5.1 |
| nDCG@20 | 0.6148 | over 11 rankable pairs |

**3.12× the strongest baseline**, and that baseline is popularity, not
semver. The kill-date gate only required beating the version number.

The strict and scoped PR-AUCs above come from `ablate.py`'s "everything"
row on this dataset; their baselines were not re-measured on the 9 Sep
data, so **only the alias column is a complete result**. PR-AUC is not
comparable across labels in general — its floor is the positive rate —
but scoped (0.0343) and alias (0.0348) happen to land within 0.0005 of
each other here, so those two can be read side by side.

Kill-date gate (≥20,000 labelled rows AND ranker beats the version-number
baseline on PR-AUC): **cleared 22 days early**, 5 September against a
27 September deadline. The database went live 6 September (§8), so both
halves of the project now talk through Postgres rather than through a
person.

---

## 2. Bugs we found in our own pipeline

These are the ones worth telling. Each cost real data before it was found.

### 2.1 The biased prefix (Day 2)

`breakages.extend(find_breaking_changes(...))` keeps everything the
generator yielded *before* it crashed. numpy's cyclic aliases crash it
partway through an alphabetical walk, so numpy 2.4.6 → 2.5.0 contributed
33 rows running `acos, acosh, all … block` and then nothing.

That is not a sample of numpy. It is the first 3% of the alphabet,
presented as coverage.

**Fix:** build the list first, `found = list(...)`, and discard the whole
version pair if it raises. Missing data is honest; a biased prefix that
looks like coverage is not. The same principle now governs the per-package
timeout — a package that times out is dropped whole, never half-kept,
because half a package means its *oldest* version pairs and `released_at`
is a model feature.

### 2.2 76% of pandas was its own test suite (Day 2)

pandas produced 5,072 rows. 3,813 of them (76%) were `pandas.tests.*`.
Test code is not public API and nobody downstream imports it, so every one
of those rows was a guaranteed negative padding the dataset.

**Fix:** drop symbols containing a `tests` component or a `test_` prefix
at source. pandas: 5,072 → 1,217.

### 2.3 Two runs of the same code produced different data (Day 2)

163 rows, then 165. A transient connection error dropped `packaging 25.0`,
which silently removed the whole 25.0 → 26.0 pair and two real rows.
The analysis is deterministic; the network is not.

**Fix:** `tenacity` retry on sdist downloads. Verified 13 rows × 3 runs
identical afterwards.

### 2.4 matplotlib was filed as "not a Python package"

matplotlib's sdist ships **both** `src/` (C++ extension sources —
`_backend_agg.cpp`, `_macosx.m`, not one `.py` file) and `lib/` (the
actual library). The layout check took `src/` because it existed and was
non-empty, found no Python, and logged a top-30 package as compiled-only.

**Fix:** the test is "contains Python", not "is non-empty". Recovered
matplotlib (277 rows) plus regex, tiktoken, duckdb, shapely, watchfiles,
contourpy, xxhash.

### 2.5 We threw away a package for being called `build`

`"build"` was in the folders-that-are-not-the-library list. pypa's build
tool is *distributed as* `build`, so we discarded it as a build artefact.
The same trap was set for anything published as `tools`, `tasks`, `docs`,
`scripts` or `dist`.

**Fix:** a folder whose name matches the distribution is exempt from the
noise filter. This also made the noise list safe to extend, which is how
`versioneer`, `include`, `pysrc` and `python` were added afterwards.

### 2.6 pytest was counted twice

17 rows in a 1,140-row run were exact duplicates, all pytest. pytest ships
both `pytest` and `_pytest`; we diff both, and griffe follows the aliases
and reports the same underlying change twice under the same `_pytest.*`
path.

Found because Varad's `UNIQUE` constraint would have collapsed them —
his schema caught a bug of ours.

**Fix:** deduplicate on (symbol, kind, sub_target, explanation) at source.

### 2.7 `cannot pickle '_thread.RLock' object`

Appeared six times in one run and three times across earlier runs. Always
on slow runs, always on packages that were fine next time.

A worker is a separate **process**, so when it raises, the exception is
pickled and sent to the parent. tenacity's `RetryError` carries the failed
attempt, which carries the httpx objects, which carry a thread lock. A
package that merely lost its downloads to a slow network died with a
message about locks, and the real cause was never recorded.

Reproduced exactly:

```
old: parent sees TypeError: cannot pickle '_thread.RLock' object
new: parent sees ScanError: Unpicklable: boom
```

**Fix:** flatten every worker exception to type-name plus text before it
crosses the process boundary, and `from None` so the original does not
ride along as `__cause__` and fail the same way.

### 2.8 A timeout that could be swallowed is not a timeout

The per-package alarm raised a normal `Exception`, which the download
loop's `except Exception` caught and logged as one version's failure —
and then the loop carried on with the alarm already spent, so the package
ran unbounded afterwards.

**Fix:** `PackageTimeout(BaseException)`, exactly how `KeyboardInterrupt`
and `SystemExit` solve the same problem.

### 2.9 A bug I introduced while optimising, caught before it shipped

Filtering failed downloads out of the version chain would have left 2.1.0
sitting next to 2.1.2 and diffed them as neighbours — a pair that never
existed, and a direct violation of the consecutive-pairs rule the whole
project rests on. Anything introduced in 2.1.1 would be misattributed.

**Fix:** split releases into unbroken runs and diff each run separately.

### 2.10 The metric that flattered everything

`precision@10` divided by `min(k, len(group))`. Most releases have fewer
than 10 changes, so the "top 10" was the whole release and every ordering
scored identically. A **constant score — no ranking at all — measured
0.5706**, the same as every real baseline.

**Fix:** only score version pairs with a positive **and** more than *k*
changes. On the real data that is 11 pairs at k=10 and 9 at k=20, which is
itself a finding (see §5.1).

---

## 3. What the data is actually like

### 3.1 The distribution name is not the import name

91 of the top 300 PyPI packages have a hyphen, and hyphens are not legal
in Python identifiers. `typing-extensions` → `typing_extensions`,
`pyyaml` → `yaml`, `pillow` → `PIL`, `beautifulsoup4` → `bs4`,
`python-dateutil` → `dateutil`, `attrs` → both `attrs` **and** `attr`.

Guessing the import name from the distribution name loses roughly a third
of the dataset before you start, and the failure is silent. We read the
extracted source tree instead and report what is importable.

### 3.2 griffe alias resolution must be lazy

`attrs/__init__.py` does `from attr import field`. Load `attrs` alone and
that alias points into a module griffe has never seen, so the first
attempt to follow it raises `AliasResolutionError` and the package yields
nothing — silently, as "0 changes".

Loading every top-level module of a distribution into **one**
`GriffeLoader` fixes it. Calling `resolve_aliases()` eagerly does not: it
walks everything, so one unresolvable alias anywhere kills the package.

### 3.3 Removals that do not remove anything

click removed `LazyFile` from its source, but serves it through a module
`__getattr__` deprecation shim. **7 of 13 click "removals" still work at
runtime.** Usage says "depended on", reality says "nothing broke yet".
Label 1, truth 0.

This is the clearest single example of the label being a proxy. A
`was_deprecated_before` feature is the fix and is not built yet.

### 3.4 A security fix inverted semver

jinja2 removed the sandbox's `format_string` in a **patch** release,
3.1.4 → 3.1.5, because it was a sandbox escape (CVE). The version number
promised safety and the release removed a public method.

This is the project's thesis in one example, and it is a real one.

### 3.5 Compiled and stub-only packages are legitimately out of scope

11 of the top 500 produce nothing, for two distinct and defensible reasons:

- **Compiled**: `rpds-py`, `fastuuid`, `ruamel-yaml-clib`, `pycryptodome`,
  `pycryptodomex`, `mmh3`, `orjson`, `uuid-utils`. Rust or C, no Python
  source to diff.
- **Stub-only**: `types-requests`, `types-toml`, `types-certifi`,
  `types-pyyaml`, `types-protobuf`. They ship `.pyi`, no `.py`. Excluding
  them is right for a reason worth stating: **nobody ever writes
  `import types_requests`.** They are consumed by mypy, never at runtime,
  so every row would be a guaranteed negative — the pandas.tests problem
  again.

numpy fails separately with `CyclicAliasError` on all five version pairs.
The dataset is the top 500 **minus numpy**, and we know exactly why.

---

## 4. The labels, and where they are wrong

The label is distant supervision: *does any downstream package import or
reference this exact symbol?* No human labelled anything. Being able to
say precisely where the proxy is wrong is worth more than the model.

### 4.1 36% of positives are version strings

1,369 rows are a version string changing value. They are **16.1%
positive** against an overall rate of 2.65% — six times more likely to be
labelled positive than anything else — and they account for **221 of 611
positives**.

`sqlalchemy.__version__` going 2.0.51 → 2.0.52 is labelled positive
because 17 packages read `__version__`. That is true, and it breaks
precisely nobody: it is supposed to change every release.

Strip them and the real rate is **1.80%**, roughly 390 genuine positives.

Decision: keep them, expose `is_version_string` as a feature, and report
metrics on the full set and the non-version subset. Dropping them would
hide a real property of the data; relabelling them 0 asserts something we
cannot prove.

### 4.2 The exact join misses re-exports, and it misses the best ones

Downstream code writes `from pandas import read_csv`, recorded as
`pandas.read_csv`. griffe reports the change where the function is
defined: `pandas.io.parsers.readers.read_csv`. Same function, two paths,
and `==` says no.

Measured on 23,025 rows: **462 rows are recoverable unambiguously**
(2.65% → 4.66%, the first time the rate has been inside the book's
expected 3–10% band). What they recover:

```
34 pkgs use pandas.read_csv     changed at pandas.io.parsers.readers.read_csv
29 pkgs use httpx.get           changed at httpx._api.get
28 pkgs use pandas.concat       changed at pandas.core.reshape.concat.concat
25 pkgs use pandas.to_datetime  changed at pandas.core.tools.datetimes.to_datetime
16 pkgs use litellm.completion  changed at litellm.main.completion
```

The naive `(root, leaf)` relaxation is **too loose** and we can show it:
every `google.cloud` client class carries `DEFAULT_MTLS_ENDPOINT`, so
**134 distinct changed symbols share that one leaf** and the rule would
credit all 74 users to each. Those 1,532 rows are left alone.

So the rule is scoped: relax only where exactly one changed symbol in the
package owns the leaf name. Both labels are kept in `labelled.csv`.

The correct fix is the **alias resolver** — griffe already knows
`pandas.read_csv` points at `pandas.io.parsers.readers.read_csv`, so we
can record the real export path at ingest and join on it exactly, no
name guessing. Not built; it needs a re-ingest. Varad's decision 10
anticipates it.

### 4.3 The zero-positive packages are honest

`label_check.py` flagged 16 module roots with 100+ rows and zero
positives, which looked like a broken join. It is not:

| root | changed rows | recoverable |
|---|---|---|
| sglang | 3,468 | 0 |
| databricks | 1,957 | 0 |
| sympy | 951 | 7 |
| docx | 910 | 0 |
| Cython | 818 | 7 |

These libraries churn their internals. sympy really does change 951 things
nobody imports. The join was right and the red flag was a false alarm —
worth recording, because "we checked and it is real" is a stronger claim
than never having asked.

### 4.4 Privacy flags took two corrections

The original rule flagged **any** leading underscore, which lumped
`__version__`-style dunders in with true `_internals`. Measured: all 161
private-flagged positives were dunders, and the true-private list was
empty — which also vindicated the alias resolution.

The API contract then froze the shared rule (any non-dunder `_component`,
with `__all__` membership overriding to public). Private-symbol positive
rate is now **0.26%**, down from 2.63%.

Varad estimated private symbols would be ~40% of rows. Measured with his
own rule: **15.3%**. griffe mostly only reports symbols reachable from the
public tree, so truly private things that vanish never generate a row.

---

## 5. The model

### 5.1 precision@10 rests on eleven version pairs

225 version pairs in test; 56 have a positive; **11 also have more than 10
changes**. One pair moving changes the number by nine points.

PR-AUC is computed over all 5,638 test rows and is trustworthy.
precision@10 and nDCG@20 at this sample size are anecdotes. **Never quote
them without the n.**

### 5.2 Half the baselines cannot rank within a release at all

Every change in one release shares that release's version bump and its
package's download rank. So `semver`, `popularity` and `griffe_all` are
**constant inside a version pair** and are pinned to the no-ranking floor
on the per-pair metrics.

They can tell you an upgrade is risky. They cannot tell you which of its
187 changes to read. That is the gap the product fills, stated as a
property of the problem rather than a claim about our model.

It has a second consequence: **lambdarank only compares items within a
group, so it is architecturally incapable of using those features.** The
`popularity only` ablation fits 1 tree and scores below random — not
because popularity is useless (its baseline scores 0.0884) but because
there is no within-group gradient. The ranker beats the popularity
baseline **without being able to use popularity**.

### 5.3 The ablation: the strict-label model is a path-shape heuristic

| | path shape alone | full model | ratio |
|---|---|---|---|
| strict | 0.1716 | 0.1634 | **105%** |
| scoped | 0.1639 | 0.3465 | **47%** |

> **CORRECTED 9 Sep 2026 — the reading below was wrong, and it was my
> error, not the data's.** The claim was: 105% means the model *is* the
> depth heuristic. Re-running the ablation with a fourth column showed the
> real cause. Under the strict label, on the 9 Sep dataset, EVERY subset
> beats the full model:
>
> | run | PR-AUC | vs full |
> |---|---|---|
> | everything (15 features) | 0.0981 | 1.00 |
> | no path shape | 0.1632 | 1.66 |
> | no reachability | 0.1744 | 1.78 |
> | no popularity | 0.2058 | **2.10** |
> | path shape only | 0.1720 | 1.75 |
> | per-change only | 0.1511 | 1.54 |
>
> A 15-feature model scoring below every 2-, 3- and 7-feature subset of
> itself is not a heuristic in disguise. It is an evaluation too noisy to
> answer the question: 98 test positives, early stopping at 7 trees.
> **The 105% was the same instability, read as a finding.** Nothing in the
> strict-label column supports a claim about which features matter, and
> the original text stands below only so the mistake is on the record.

Three features — `module_depth`, `name_length`, `is_top_level` — give
almost identical absolute PR-AUC under both labels. Under the strict label
that *is* the whole model and the other ten features are net-negative
decoration. Under the scoped label the model reaches twice as far, so the
same three account for under half.

The mechanism proposed at the time: under the strict label the positives
essentially **are** the shallow symbols, because deep paths are exactly
where the exact join fails (§4.2). Repair the re-export misses and depth
stops being sufficient.

**That prediction was testable, and §9.4 tested it. It was wrong** — but
wrong in an interesting direction. Fixing the join did not make depth stop
mattering; it separated *depth* from *reachability*, and reachability was
the thing carrying the signal all along.

### 5.4 The split is unstable, and the stopping rule is the worst of it

Positive rates on the 9 Sep run, `label_alias`:

| slice | rows | positive |
|---|---|---|
| train | 13,850 | 4.25% |
| **validation** | 3,381 | **6.39%** |
| test | 5,683 | 3.48% |

Validation is nearly **twice as dense in positives as test**. Early
stopping judges the model against a slice that does not resemble what it
is scored on, and it halts at 10 trees. Across the nine ablation runs the
tree count ranged from **1 to 120**.

This is no longer "the numbers are noisy". It is: *the stopping rule is
being asked a different question from the one we report on.* Every
percentage in §5.3's correction traces back here.

The fix is not another feature. It is **repeated temporal splits** — cut
at several dates, train and score at each, report the spread — so a
result has to survive more than one arbitrary date to count. That is now
the highest-value open item (§7).

Test being sparser than train makes every reported number pessimistic,
which is the right direction to be wrong in, but it should be said rather
than found.

### 5.5 The three numbers that make PR-AUC readable

PR-AUC's floor is the **positive rate**, not 0.5 the way ROC-AUC's is. The
shipped model is `lambdarank-label_alias`; its test positive rate is
**3.48%** over 5,683 rows. (§5.4's strict-label floor is 1.72% — the two
differ by 2× and are not interchangeable.)

| | value | vs floor |
|---|---|---|
| floor — positive rate | 0.0348 | 1.0× |
| semver baseline | 0.0389 | 1.1× |
| best baseline — popularity | 0.1057 | 3.0× |
| **model** | **0.3301** | **9.5×** |

**3.12× the strongest baseline, 9.5× the floor.** precision@10 is 0.3313
and nDCG@20 is 0.6148 — both up sharply from the scoped label's 0.2238
and 0.4631 — but read §5.1 before quoting either anywhere.

**Quote all three or none.** "PR-AUC 0.35" on its own is unreadable: it is
excellent at this positive rate and mediocre at 30%. Note also that the
baseline being beaten is `popularity`, not `semver` — the kill-date gate
only required beating the version-number baseline, and popularity is the
harder one, because "just sort by download count" is the objection the
project exists to answer.

The first `model_run` row written to Postgres carried no floor, and
recovering the number afterwards meant reloading `features.csv` and
re-deriving the split. `train.py` now writes `positive_rate` into `notes`.

---

## 6. Operational facts

- **griffe is 96% of the runtime; downloads are 4%.** Inside griffe,
  *parsing* is 66–97% and walking the diff is cheap. Optimising downloads
  bought 6%; parsing each version once instead of twice (a rolling window
  over the version chain) bought **31%, with byte-identical output**.
- **PyPI's speed varies by 6×.** The same 1,500-package usage scan took
  6.0 minutes once and 39.6 minutes another time, same code, more work
  done in the fast run. The `RLock` failures cluster in the slow runs
  (§2.7). Varad needs this for scheduling the nightly job.
- **Full ingest: 500 packages, 6 releases each, ~34 minutes** on an M4
  MacBook Air at `--workers 10`. Two packages hit the 600s timeout
  (transformers, libcst); both recovered at 1200s.
- **`failures.csv` is append-only**, so a retry logs a second row for a
  package that fails twice. The table is an honest history of attempts,
  not a snapshot of current state. Do not put it in a report without
  deduplicating.

- **Homebrew deleted the interpreter the venv was built on.** Mid-session,
  `python3` stopped finding pandas. `.venv/bin/python3` was still listed by
  `ls` but would not execute: it is a symlink chain ending at
  `/opt/homebrew/opt/python@3.12/bin/python3.12`, and `brew autoremove` had
  taken `python@3.12` away as an unused dependency after an upgrade to
  3.14.7. A dangling symlink is not executable, so the shell skipped it and
  fell through to Homebrew's Python — which has none of our packages.
  `zsh: no such file or directory` on a file you can see in `ls` is
  reporting the missing *target*, not the link.

  Rebuilt on 3.12 rather than 3.14 deliberately: every number in this file
  was produced on 3.12.14, and moving to 3.14 would drag pandas and numpy
  across major versions. `brew install python@3.12` explicitly (rather than
  as a dependency) is what stops autoremove from doing it again.

---

## 7. Open, and where it goes

0. **REPEATED TEMPORAL SPLITS — now the top item** (§5.4). One arbitrary
   date decides everything, validation is twice as dense in positives as
   test, early stopping lands anywhere between 1 and 120 trees, and §5.3's
   headline claim turned out to be an artefact of exactly this. Cut at
   several dates, report the spread. **No new feature is worth more than
   this**, because right now a result cannot be distinguished from a
   lucky cut.
1. ~~**Alias resolver**~~ — done, §9. Recovered 392 rows by fact and
   identified 210 heuristic claims with no import statement behind them.
2. **`was_deprecated_before`** (§3.3) — the click shim class.
3. **Version-string handling** (§4.1) — report metrics with and without.
4. ~~**`ml/db.py`**~~ — done, §8.
5. **More test positives** — 98 strict / 198 alias is thin, and only
   16 test pairs are rankable at 10. Same fix as item 0.
6. **PARAMETER_MOVED is 24% of the dataset and 0.6% positive.** Nobody has
   looked at why. Largest unexamined class.
6b. **Drop `module_depth`, `name_length`, `is_top_level`?** The alias
   ablation says removing them *improves* PR-AUC by 12.8% (§9.4). Do NOT
   act on that number: it is measured on test, and choosing features by
   test score is test-set selection — the same class of error as a random
   split. Decide it on validation, confirm once on test.
6c. **The `py_src` class of bad symbol roots** (§9.5). ~210 rows, 1.0%,
   all guaranteed negatives. Two-line fix in `_is_noise` plus a deeper
   sdist descent in `download.py`; needs a re-ingest, so batch it.
7. **Quiet releases are not recorded** (§8.3). A release we analysed and
   found clean leaves no row anywhere, so it is indistinguishable from one
   we never looked at — the exact distinction decision 1 exists to keep.
   Fix is a PyPI re-fetch into `data/releases.csv`, ~2 min for 410
   packages. Blocked on what the API wants those rows to say.
8. **Wheel-only releases are an unmeasured off-by-one** (§8.2). Rare, but
   it is the only remaining way `via_version` can name the wrong release.
9. **Yanked releases are free human labels.** A maintainer yanking a
   release is a person saying "this one shipped something bad" — an
   independent signal, not distant supervision. `charset-normalizer 3.4.8`
   is one. May be too rare to use; nobody has counted.

---

## 8. The database

Loaded 6 Sep 2026. 500 packages, 2,061 releases, 23,024 breakages, 39,154
usage rows, 23,024 predictions, one `model_run`.

### 8.1 The loader's counts measure different things

`breakage 23,024 sent, 23,030 in table` is not an error. "Sent" is what we
built from `changes.csv`; "in table" is a `SELECT count` over the whole
table. The six-row difference was **pre-existing fixture data** — Varad's
`model_run v0-fake` slice, honestly named, covering a private symbol and a
shared symbol path with two `sub_target`s so the API could be built before
the pipeline produced anything. `scripts/db_extras.py` identifies them and
proposes the deletion without running it.

`package 500` against a dry run predicting 410 was **my preview being
wrong**, not the writer. The writer loads every package in `packages.csv`
including the 90 that broke nothing, because the package table's job is to
record that we looked; the dry run counted distinct packages in
`changes.csv`. A preview that does not predict the thing it previews is
the same class of bug as the chain renderer in §8.2. Fixed.

**Model versions sort as text.** `'v0-fake' > 'lambdarank-label_scoped'`,
so `ORDER BY version DESC` picks the fixture over the real run. Future runs
get dated versions; this one cannot be renamed because 23,024 predictions
carry it as a foreign key.

### 8.2 The pairs really are consecutive

`scripts/verify_pairs.py`, written to answer Varad before loading. Every
pair moves forward in PEP 440 order; no pair skips a release another pair
covers. `boto3 1.43.84 -> 1.43.85 -> ... -> 1.43.89` — six consecutive
patch releases, each diffed.

The counts close on themselves: **1,580 pairs + 481 unbroken runs = 2,061
releases**, and **481 runs = 410 packages + 71 chain breaks**. Two scripts,
separate code paths, same 2,061.

A version can be missing from a chain for **three** reasons, not two:

1. **Never listed** — no non-yanked sdist. Leaves *no gap*; the chain steps
   over it silently. Two sub-cases: **yanked** (correct to skip — nobody
   upgraded through it; `charset-normalizer 3.4.8` shipped both wheel and
   sdist and yanked the lot) and **wheel-only** (griffe has no source, so a
   break introduced there is attributed to the next release that did ship
   source — a real off-by-one, frequency unmeasured).
2. **Listed, download failed** — leaves a gap. The chain is split so
   2.1.0 is never diffed against 2.1.2. In `failures.csv`.
3. **Listed, diffed, found nothing** — leaves a gap. Not in `failures.csv`.

The first version of the chain renderer printed each pair's *end* version
and assumed the next pair *started* there, drawing
`typing-extensions 4.13.2 -> 4.14.0 -> 4.16.0` as unbroken while flagging
the same package as broken two sections below. **A diagram that hides the
thing it exists to prove is worse than no diagram**, and this one was about
to be sent as evidence.

### 8.3 What the database cannot yet say

`changes.csv` holds only pairs that produced a change, so `release` has
2,061 rows — every release that *broke something*, plus its neighbours.
A release analysed and found clean is absent. Decision 1 wants
"analysed, found nothing" to be distinguishable from "never analysed", and
with the three cases above it is really a **three-way** distinction:
analysed-and-clean, analysis-failed, and no-source-to-analyse. If that is a
boolean in the schema it wants to be an enum. Open, §7.7.

---

## 9. The alias resolver (Day 6)

`ml/ingest/api_extract.py` — `alias_hops`, `resolve_alias_chains`,
`export_index`, `user_paths`; new `export_paths` column in `changes.csv`.

The re-export gap (§4.2) was patched with a heuristic: relax the join
where exactly one changed symbol in the package owns the leaf name. That
guess is usually right and has no way to know when it is wrong. griffe
already holds the answer — `from .api import get` is in the source — so
Day 6 replaced the guess with the fact.

### 9.1 Three things the fixture caught before the pipeline ran

Built on a hand-made package and on requests/attrs/click, deliberately
before touching the real ingest.

**`alias.target_path` needs no resolution.** It is a plain string built
from the import statement. Reading it never makes griffe go and find the
target, so none of §3.2's AliasResolutionError machinery can fire. That
is the only reason this is cheap enough to run on every version —
measured at **0.7% of parse time** on requests.

**One hop is not enough, and it fails silently.** Aliases chain:

```
fakepkg.read_csv     -> fakepkg.io.read_csv           still an alias
fakepkg.io.read_csv  -> fakepkg.io.parsers.read_csv   the real function
```

which is exactly pandas. A one-hop map joins against a middle path griffe
never reports and **does nothing at all** — no error, no crash. The
obvious implementation would have looked like "the alias idea doesn't
work". Cycles are real too (`a` imports from `b`, `b` from `a`), so the
closure needs a guard; numpy's cyclic aliases already truncate griffe's
own diff walker.

**The alias is rarely on the thing that changed.** griffe reports
`pandas.io.formats.style.Styler.where`; the re-export is on `Styler`. So
each prefix is checked and the remainder carried across —
`fakepkg.io.parsers.Frame.append` becomes `fakepkg.Frame.append`. Without
this, methods on re-exported classes are all missed, and classes are
where the methods are.

### 9.2 The 55% that was noise

First run of the new column: 55% of rows gained a user-facing path. Then:

```
packaging.utils.BuildTag  ->  packaging.metadata.utils.BuildTag
```

Backwards. `import` creates aliases in **both directions** and griffe
records both identically:

| source | alias | names | direction |
|---|---|---|---|
| `requests/__init__.py`: `from .api import get` | `requests.get` | `requests.api.get` | **shorter** — real |
| `packaging/metadata.py`: `from . import utils` | `packaging.metadata.utils` | `packaging.utils` | longer — noise |

A real re-export makes a name **shorter**. Filtering on that, plus keeping
equal depth when the root differs (`attrs.*` vs `attr.*` is a genuine
second name):

| package | before | after | truth |
|---|---|---|---|
| packaging | 55% | **0%** | 0% — it has no shorter names |
| click | 89% | 56% | four junk paths per row removed |
| attrs | 100% | 100% | was right all along |

**The most valuable line is the one that went to zero.**
`packaging.utils.BuildTag` really is the only way to import it, and a
measurement saying otherwise is inventing coverage.

Final: **12,442 rows (54.3%) have at least one shorter public name**,
mean 1.69 paths each. Highest: `databricks-sdk` 99.6% of 1,956 rows — a
generated SDK that re-exports everything. Lowest: `pygments` **0.0% of
237 rows**, which is the check that makes the rest trustworthy.

### 9.3 What the heuristic was getting wrong — 210 rows

| | rows |
|---|---|
| exact join | 611 |
| scoped recovers over exact | 463 |
| alias recovers over exact | 392 |
| **both agree** | **253** |
| **scoped claims, no import statement exists** | **210** |
| alias finds, heuristic was too cautious | 139 |

**Just under half of what the shipping label claimed beyond the exact
join has nothing in any package's source to support it.** That is Varad's
decision-10 concern, answered with a number instead of an opinion. The
ratio held at 45–46% before and after `transformers` (1,578 rows) was
added, so it is a property of the heuristic, not of the sample.

The mechanism, in one row:

```
yaml.representer.BaseRepresenter.add_representer   leaf used by 7 pkgs
```

`add_representer` genuinely is used by seven packages — as
**`yaml.add_representer`**, the module-level function. The changed symbol
is the classmethod on `BaseRepresenter`, a different object sharing a
name. The leaf heuristic cannot tell them apart. The alias graph can,
because no import statement connects the two.

### 9.4 Reachability, not depth

`ablate.py` now separates two things that were tangled:

- **path shape** — `module_depth`, `name_length`, `is_top_level`: where a
  symbol is DEFINED.
- **reachability** — `public_depth`, `has_export_path`: how short the name
  anyone can import it by is.

Under `label_alias`, every entry points the same way:

| run | PR-AUC | vs full |
|---|---|---|
| everything | 0.3301 | 1.00 |
| no reachability | 0.2057 | **0.62** |
| no path shape | 0.3725 | 1.13 |
| no popularity | 0.3322 | 1.01 |
| reachability only (2 features) | 0.3079 | **0.93** |
| path shape only (3 features) | 0.2094 | 0.63 |
| popularity only | 0.0482 | 0.15 |

Removing reachability costs **37.7%**. Removing definition-path shape
*gains* 12.8%. Popularity is neutral, consistent with §5.2.

So the answer to §5.3's question is: **the depth effect was never the
story — reachability is.** The model has learned "how short is the name a
user can import this by", not "how deep is it buried", and the two were
inseparable until the alias graph pulled them apart.

**The circularity, stated rather than buried.** The usage index records
paths as downstream code writes them, and downstream code writes short
ones. So a symbol with a short public name has both more ways to match
and matches on the kind of path the index is dense in. This is
simultaneously a real property of the ecosystem and a property of how the
label is built. Keeping the two feature groups separate is what makes it
arguable with numbers. It is not resolved.

### 9.5 A data-quality bug found on the way

3.0% of rows have a symbol root that does not resemble the package name.
Most are legitimate — `scikit-learn` → `sklearn` (294 rows), `protobuf` →
`google`, `pymongo` → `bson`. About **210 rows (1.0%) are not**:

```
tokenizers -> py_src (129)     ruff, ast-serialize -> crates (25)
cryptography -> _cffi_src (15) grpcio-* -> grpc_version (17)
dill, multiprocess -> version (10)   cython -> runtests (4)
```

`py_src.tokenizers.models.BPE.from_file` is importable by nobody; the
real path is `tokenizers.models.BPE.from_file`. Every such row is a
**guaranteed negative** no matter how used the function is.

Two causes. `_is_noise()` tests the **raw** lowercased stem against a set
containing **normalised** entries — so `py_src` sails past a filter that
literally contains `pysrc`, and the set's own `third_party` entry can
never match either. And stray top-level scripts (`version.py`,
`runtests.py`, `make_cffi.py`) are accepted by the single-file-module
branch; the fix there is that a top-level script should only count when
no package directory sits beside it, which is what keeps `six.py` working
without also admitting `runtests.py`.

Not fixed on the day, deliberately: filtering `py_src` without also
fixing the sdist descent in `download.py` would turn tokenizers into a
`NoPythonModule` failure — losing 129 rows and gaining nothing. It is 1%
of the negative class and needs a re-ingest, so it waits for a batch
(§7.6c).

### 9.6 Why the row count moved, and how we knew

The re-ingest produced 21,336 rows against the previous 23,025.
`scripts/compare_runs.py` exists to decide whether that is PyPI moving or
our own code, and the decisive test is: **for version pairs present in
both runs, do the row counts agree?**

1,540 shared pairs, 4 disagreed — all `tokenizers`, net −19 rows, three
of them **exactly halved**. Not duplicate rows (24 rows, 24 distinct
keys) but the same 12 changes recorded twice under two different wrong
roots, `bindings.*` and `py_src.*`; `--restart` cleared the stale sdists
and one copy went away. So the drop was a fix, and the rest was the
release window shifting.

The remaining gap was 8 packages that failed this run and had succeeded
before — `transformers` alone was 1,270 rows. **`done.txt` records
failures as done**, so a plain re-run does nothing; the failed packages
have to be removed from the ledger first. After that: **22,914 rows
across 406 packages**, parity with the old dataset while carrying the new
column.
