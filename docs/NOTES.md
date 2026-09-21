# Findings notebook

Everything BreakRank learned by measuring rather than assuming, with the
numbers that back it. Written for two readers: me in November when the
report is due, and an examiner asking "how do you know?"

Rule for this file: **no claim without the number that produced it.** If a
line here says something is true, the run that showed it is named.

Last updated 14 Sep 2026. Dataset: **19,121 breakage rows across 415
packages**, top 500 PyPI by download count, 6 releases each.

That row count went **down** from 22,914 and the dataset got better: the
fold in §10.2 removed 13,694 rows that were the same change counted once
per inheriting class, and §10.6 recovered 8 packages a version filter had
been hiding. Neither is a sampling change. Both are described where they
happened.

**Read §1's range table before quoting any single number from this file.**
Several sections still cite one cut date because that is what produced
them; §5.6 measured how far those numbers move when the date moves, and
the answer is: a lot.

**Sections dated before 14 Sep carry superseded figures.** They are kept
because the reasoning in them is still how the conclusion was reached,
and a findings notebook that quietly rewrites its own history is worth
less than one that shows where it was wrong. Where a number has been
replaced, the subsection says so at the top. §1 and §11 are current.

---

## 1. Where the project stands

Three labels now, on the same rows (§9). `label_alias` is the shipped one.

> **FROZEN 16 Sep 2026, tag `dataset-20260916`.** 17,014 rows, 416
> packages, 1,601 version pairs. Every number below and in the public
> write-up cites that tag.
>
> This section was rewritten four times in five days, and **not one of
> those rewrites was caused by a modelling decision.** The six-release
> window is defined relative to *today*, so each re-ingest produced a
> different dataset: the headline lift read 2.48×, 2.59×, 4.82× and
> 2.25× as releases shipped underneath it. `databricks-sdk` alone moved
> the dataset by 1,901 rows overnight.
>
> That is correct behaviour for a live site and impossible for a report
> with a submission date. So the two are now separate: re-ingests keep
> the site current and are never quoted; the report cites the frozen tag.
> §15.4 has the reasoning.

| | strict `label` | `label_scoped` | **`label_alias`** |
|---|---|---|---|
| rule | exact path match | exact, or one symbol owns the leaf | exact, or a real export path |
| basis | fact, and incomplete | **a guess** | fact, from griffe's alias graph |
| positive rows | 609 (3.58%) | 1,113 (6.54%) | **1,019 (5.99%)** |
| test positives | 132 | 292 | **278** |
| test positive rate — **the PR-AUC floor** | 0.0313 | 0.0693 | **0.0660** |
| ranker PR-AUC | 0.2262 | 0.2296 | **0.3866** |

### The headline, stated the way it survives scrutiny

**No lift this project quotes is a single number.** §5.6 refits at seven
cut dates and reports the range; §11.2 discards any cut with fewer than
10 rankable version pairs, because precision@10 over four pairs is not a
weak measurement of anything — it is not a measurement.

`label_alias`, against the strongest baseline (popularity), on its own
six usable cuts:

| | value |
|---|---|
| beats popularity | **6 of 7 cut dates** (the seventh too small to measure) |
| median lift | **1.99×** |
| worst cut | **1.45×** |
| best cut | 2.76× |

> The ranker beats the strongest baseline at **every cut date with enough
> rankable pairs to measure one** — six of seven — by a median of
> **1.99×** and never less than **1.45×**.

Say "six of seven, the other two too small to measure", never "6/6". The
excluded cuts are a limitation, not a rounding.

### Why `label_alias` and not one of the others

**On the five cuts all three labels share** — which is the only fair
comparison, and the code now enforces it (§15.1):

| label | lift median | **lift MIN** |
|---|---|---|
| strict `label` | 1.90× | 1.81× |
| `label_scoped` | 1.38× | 1.27× |
| **`label_alias`** | **2.05×** | **1.92×** |

Two numbers, two jobs, and they must not be mixed. **Which label ships**
is decided on shared cuts: 1.92×. **How good the shipped model is** comes
from `label_alias` on its own six cuts: median 1.99×, min 1.45×. That
second set is what `metrics.json` carries.

The comparison had to be fixed before it could be read. Taken over each
label's own cuts, the rule picked the strict `label` — because `label`
was skipped at one cut for having too few rankable pairs and so never
faced the hardest one. A worst case measured over different exams is not
a comparison (§15.1, and the same class of error as §11.2 and §13.2).

The provenance argument agrees with the numbers rather than carrying
them: §9.3 found 220 of scoped's positives have no import statement
behind them.

Single-split reference numbers, frozen cut (2026-07-28):

| | value | vs floor |
|---|---|---|
| floor — test positive rate | 0.0660 | 1.0× |
| semver baseline (the kill-date gate) | 0.0622 | 0.9× |
| best baseline — popularity | 0.1722 | 2.6× |
| ranker PR-AUC | 0.3866 | 5.9× |
| precision@10 | 0.2231 | over 26 rankable pairs — see §5.1 |
| nDCG@20 | 0.6289 | over 18 rankable pairs |

### The score went up. The model did not. (§11.5)

*Figures in this subsection are from the 14 Sep dataset, and stay that
way deliberately: it is a controlled experiment about the FOLD, and both
sides of it saw the same data. Re-running it on the frozen dataset would
answer a different question.*

PR-AUC was 0.3465 on 5 Sep and 0.5137 after the fold, and **that rise was
arithmetic, not skill.** Measured with the cut date held fixed and only
the duplicates differing:

| | before the fold | after |
|---|---|---|
| floor (positive rate) | 0.0112 | 0.0356 |
| PR-AUC | 0.4439 | 0.5137 |
| **lift over popularity** | **4.88×** | **4.82×** |

Lift over popularity is flat. The floor rose 3.18× because 13,694
guaranteed negatives left the dataset, and PR-AUC's floor *is* the
positive rate — so the score rose without the ranker changing.

**Never present the PR-AUC increase as an improvement.** The true
sentence is less flattering and considerably stronger: *the dataset was
30% duplicates, every duplicate was a guaranteed negative, and they were
depressing the measured score of a model that was already this good.*

Ignore vs-floor when comparing the two: it reads 39.6× → 14.4×, which
looks like a collapse and is only a smaller ratio over a bigger floor.
Lift is the one measure comparable across datasets, because the baseline
is refit inside each.

### What the model is actually made of

All 16 features, by gain, on the shipped model (§11.3 explains why the
whole table is printed and not the top eight):

`public_depth` 35.6% · `name_length` 13.4% · `package_rank` 12.2% ·
`package_churn` 10.4% · `module_depth` 7.4% · `kind` 5.1% ·
`release_size` 4.8% · `is_version_string` 3.9% · `has_sub_target` 2.9% ·
`bump` 1.4% · `in_dunder_all` 1.2% · `is_dunder` 1.0% ·
`has_export_path` 0.7% · `is_top_level` 0.1% · **`inherited_by` 0.0%** ·
**`is_private` 0.0%**

Two features at zero gain, and neither is embarrassing once stated:

- **`inherited_by`** (§10.2) is real information the model does not want.
  Passing one test and failing another is an acceptable outcome for a
  feature; pretending otherwise is not. It keeps two jobs and neither is
  "predictor": as a **database column** it belongs next to a finding on
  the site — "this change affects 3,242 inheriting classes" is the kind
  of thing a human reading a diff wants — and as a feature the model
  ignores it is the ablation's **control** (§11.3), which is the only
  reason that table knows what zero looks like. Left in the feature set
  deliberately: LightGBM never splits on it, so it costs nothing, and
  removing it would leave the noise floor resting on a single point.
- **`is_private`** is *redundant*, not irrelevant. Private symbols really
  are near-dead (0.34% positive against 3.26%), but `public_depth` and
  `has_export_path` already carry that at finer resolution. Expect the
  question — the frozen API contract's central rule having zero gain
  looks alarming until you say why.

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

### 5.6 Does it survive a different date? (12 Sep, `ml/model/stability.py`)

§5.4 said one arbitrary cut decides everything. This measures how much.
Seven cut dates across the middle of the release history; at each one the
model **and the baselines** are refitted, so lift is computed inside a
split before anything is summarised. Carrying one cut's baselines across
all seven would compare a moving model against a fixed target.

> **SUPERSEDED 14 Sep — kept as the record of what was measured on 12
> Sep, not as a current figure.** Every number in this subsection comes
> from the 21,336-row dataset, which contained roughly 3,600 inherited
> duplicate rows (§10.2, §10.5) and was missing 8 OpenTelemetry packages
> (§10.6). It also predates the rankable-pairs gate (§11.2), so its
> seven cuts include two that cannot support precision@10 at all.
> Current figures: §1. The comparison *within* this subsection — holdout
> versus cv — remains valid, because both sides saw the same data.

| label | beats popularity | median lift | worst cut | best cut | PR-AUC spread |
|---|---|---|---|---|---|
| strict `label` | 4 / 7 | 1.08× | 0.57× | 1.45× | 0.070 |
| `label_scoped` | 6 / 7 | 1.74× | 0.37× | 2.17× | 0.331 |
| **`label_alias`** | **7 / 7** | 1.82× | **1.48×** | 3.12× | 0.221 |

**Three findings, in order of how much they change what we can say.**

**1. The strict label does not beat popularity.** It reported 1.85× on one
cut; across seven it is 1.08× median and loses outright at three. Every
conclusion drawn from the strict-label column — §5.3's original claim
included — was drawn from a sample of one.

**2. `label_scoped` has a failure mode `label_alias` does not.** They are
0.08× apart on the median, which separates nothing. At the 2026-05-10 cut
scoped scores **0.37×** — three times *worse* than sorting by download
count — with early stopping firing at **1 tree**. Alias's worst cut is
1.48×. A model that is sometimes worse than the dumb baseline cannot be
shipped on the strength of its median, because you cannot tell in advance
which day you are having. **That, not the ablation and not the median, is
why `label_alias` ships.**

**3. The baseline moves too, and nobody was watching it.** Popularity's
own PR-AUC ranges **0.088 – 0.245** across the cuts. At the last two the
test half is down to ~3,200 rows and popularity roughly doubles, so those
splits are not merely noisier — they are a different problem, with a
denser, more concentrated test half. Any single-split lift silently
depends on this.

**Two things this does not fix.** `rankable10` falls to **5 pairs** at the
last cut, so precision@10 there is meaningless whatever it prints (§5.1).
And early stopping still lands anywhere from **1 to 81 trees** — the two
worst splits in the whole table are both 1-tree fits, which is now a
specific, findable cause rather than general noise.

**What the spread is not.** These splits share most of their training data
and overlap heavily in test, so they are not independent draws. The range
is a *sensitivity* — how much the answer moves when the arbitrary choice
moves — not a confidence interval, and must never be written with a `±`.

### 5.7 Fixing the stopping rule (12 Sep, `--stopping cv`)

§5.6 named the mechanism: one validation slice picks the tree count, and
the two worst results in the table were both **1-tree fits**. So stop
letting one slice decide. Four expanding-window folds *inside train* each
propose a count, the **median** wins, and the model is then refitted on
**all** of train with that number fixed — no early stopping at fit time,
because re-deciding it on a slice would reinstate the problem. Every fold
lives inside the training half; test is not involved at any point.

The median is the mechanism, not a detail: one fold collapsing to 1 tree
barely moves a median of four.

> **SUPERSEDED 14 Sep — kept as the record of what was measured on 12
> Sep, not as a current figure.** Every number in this subsection comes
> from the 21,336-row dataset, which contained roughly 3,600 inherited
> duplicate rows (§10.2, §10.5) and was missing 8 OpenTelemetry packages
> (§10.6). It also predates the rankable-pairs gate (§11.2), so its
> seven cuts include two that cannot support precision@10 at all.
> Current figures: §1. The comparison *within* this subsection — holdout
> versus cv — remains valid, because both sides saw the same data.

| | holdout (one slice) | **cv (four folds)** |
|---|---|---|
| strict `label` | 4/7, median 1.08×, worst **0.57×** | 5/7, median 1.96×, worst 0.75× |
| `label_scoped` | 6/7, median 1.74×, worst **0.37×** | **7/7**, median 1.97×, worst **1.58×** |
| `label_alias` | 7/7, median 1.82×, worst 1.48× | **7/7**, median **2.48×**, worst **1.79×** |
| trees chosen | **1 – 81** | **20 – 37** |
| alias PR-AUC spread | 0.221 | **0.129** |

**Every label improved.** That is the tell that this was a bug rather than
a tuning preference: a broken stopping rule was *suppressing* the result,
not inflating it. `label_scoped`'s worst case improved four-fold and its
collapse disappeared entirely.

Two legitimate effects are combined here and both should be stated. The
count no longer depends on one unrepresentative slice — and `fit_fixed`
trains on the whole training half, where the old path threw away a fifth
of it to build a validation set it then used badly.

**Why this changed the shipping decision's basis, not its outcome.**
Before the fix, alias was chosen because scoped had a failure mode. After
it, scoped has no failure mode either, and alias wins on the ordinary
merits instead: better worst case (1.79× vs 1.58×), better median (2.48×
vs 1.97×, a gap wider than the 0.5× this file treats as the readability
threshold), and the **smallest spread of the three**. The label-provenance
argument in §9.3 now agrees with the metrics rather than outvoting them.

**Still open.** The strict label loses to popularity at 2 of 7 dates even
with the fix, so nothing in the strict column is quotable. `rankable10`
still falls to 5–9 pairs at the late cuts (§5.1). And the CV fold count,
the 40% starting boundary and the 20-tree floor are all unexamined
choices — they are stated in `train.py` rather than tuned, which is
honest but not the same as justified.

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

0. ~~**Repeated temporal splits**~~ — done, §5.6. It cost the project
   three headline numbers and bought a defensible one. **New top item
   below.**
0b. ~~**Early stopping**~~ — done, §5.7. Every label improved; scoped's
   0.37× collapse became 1.58×. Remaining: the fold count, the 40% start
   boundary and the 20-tree floor are stated, not tuned.
0c. **ORIGINAL TEXT, kept for the record — early stopping.** Trees land anywhere from **1
   to 81** across cuts, and the two worst splits in §5.6 are both 1-tree
   fits: `label_scoped` at 0.37× and the strict label at 0.57×. That is no
   longer "the numbers are noisy" — it is a specific, findable cause with
   two named victims. The validation slice is the newest 20% of train and
   runs ~2× denser in positives than test (§5.4), so the stopping rule is
   judging against a distribution the model is never scored on. Options,
   cheapest first: a minimum tree count; stratifying validation to match
   test's positive rate; or k-fold-style validation *within* the training
   window instead of one tail slice.
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

---

## 10. The sdist layout fix, and what it uncovered (Day 8)

### 10.1 `py_src` — three bugs in one filter

`NOT_THE_LIBRARY` is a list of directory names that are never the
package: `tests`, `docs`, `examples`, `bindings`, `crates`. `tokenizers`
ships its Python under `bindings/python/py_src/tokenizers/`, and the
filter contained `pysrc` — so `py_src` sailed straight past it and the
extractor produced symbols like
`bindings.python.py_src.tokenizers.models.BPE.from_file`. Nobody can
import that. 87 such rows were in the database (NOTES §9.5).

Three separate fixes, and only the second is the one people guess:

1. **Normalise before comparing.** `re.sub(r"[-_.]", "", name).lower()`
   on both sides, so `py_src`, `py-src` and `pysrc` are one name.
2. **Descend when the top level is empty.** Some sdists nest the package
   two or three directories down. `resolve_layout` now walks down —
   **but only when the top level has nothing importable at all**, never
   merely because the directory name does not match the distribution
   name. `protobuf` is the case that rule protects: it ships
   `google/protobuf/`, and a rule that descended on a name mismatch
   would have thrown away the real package to go hunting for a directory
   called `protobuf`.
3. **Setup scripts lose to real packages.** When a directory holds both
   a real package and loose scripts, the scripts are only kept if their
   name matches the distribution.

12/12 regression checks passed, and `compare_runs.py` section 3 —
the section that exists to tell "our code changed the diff" from "PyPI
moved" — showed 24 shared pairs differing by a net **−40 rows**, every
one of them a junk root going away: `crates`, `runtests`, `grpc_version`,
`make_cffi`, `version`. The fix removed exactly what it was meant to.

### 10.2 Then one version pair produced 30% of the dataset

The re-ingest came out at **32,405 rows**, up from 22,914. Section 3 had
already exonerated the code, so the growth sat on new pairs — and almost
all of it on one:

```
transformers 5.16.1 -> 5.17.0     9,863 rows
```

9,827 under `transformers.models`, 9,797 `OBJECT_REMOVED`, 487 of 509
model subpackages implicated, 9,780 at exactly depth 5.

**Two hypotheses, both killed by measurement before anything was
changed.** Models deleted wholesale? No: 509 → 516 subpackages, net +7,
zero removed. A half-failed load on the new side? No: 2,636 vs 2,680
modules materialised against 2,637/2,681 `.py` files on disk, 74,390 vs
75,638 members — the new side has *more*. I then guessed an import
sweep, having seen griffe count `torch`, `nn` and `Callable` as members
of `modeling_roberta`, and asked for the leaf names expecting
`Optional`, `Union`, `dataclass`.

The leaf names said something else. **Eleven distinct names in 9,781
rows:**

```
invert_attention_mask                         3,243
create_extended_attention_mask_for_decoder    3,243
get_extended_attention_mask                   3,243
rot_pos_emb                                      28
fast_pos_embed_interpolate                       12
...six more, single digits
```

3 × 3,243 = **9,729 of 9,781 rows are three methods.** They are
`ModuleUtilsMixin` methods. transformers 5.17.0 removed them from that
mixin, and 3,242 model classes inherit it.

### 10.3 Why griffe reports it 3,243 times

Not a bug, and not ours. Read at the source
(`griffe/_internal/diff.py:623`):

```python
for name, old_member in old_obj.all_members.items():
```

and `all_members`, for a class, is
`{**self.inherited_members, **self.members}`. So every method a subclass
inherits is diffed as if the subclass declared it. Reproduced in a
10-line fixture — one mixin, one removed method, ten subclasses:

```
rows produced: 11
   pkg.base.Mixin.gone      Function  declared
   pkg.models.m0.M0.gone    Alias     target_path=pkg.base.Mixin.gone
   ... one per subclass
```

The defining row is a `Function`; every repeat is an `Alias` whose
`target_path` names the definition and whose own `.path` has been
re-parented onto the subclass. That `target_path` is a plain string
filled in at parse time — the same property the alias resolver leans on
(§9.1), so reading it resolves nothing and cannot raise.

### 10.4 What we did about it

Every one of those 9,729 rows is **true**. None of them is a separate
event. Left alone they would have been 30% of the training data, all
with near-identical features, all labelled 0 — one library's refactor
setting the positive rate for the entire dataset, and the site showing
one removal 3,243 times.

`fold_inherited()` collapses them onto the defining class and keeps the
count as a new column, **`inherited_by`**. Folding at extract time, not
in `labels.py`, is deliberate: a fold at label time would leave the raw
rows in `changes.csv` and in the database, so the product would still be
wrong even if the model was not.

The count is not bookkeeping. A method 3,242 classes inherit is a
different kind of break from one on a leaf class, and `inherited_by`
gives the ranker that in one number instead of 3,242 duplicate rows. It
goes into the feature set on its own ablation group (`BLAST_RADIUS`) so
it has to earn its place rather than be assumed useful.

**The guard matters more than the fold.** griffe skips private members,
so if the base class is private no row for the definition exists, and
folding onto it would silently delete a real public breakage — a public
subclass of a private mixin is exactly the case where the subclass's
name is the one users wrote. In that case the fold keeps the shallowest
inheriting path as a stand-in instead. Tie-broken on the path string,
not on iteration order: `breakage` is keyed on `symbol_path`, and a
stand-in that changed between runs would write a second row instead of
upserting the first.

`scripts/test_inherited.py` covers all three cases plus the stability
property. No network, no sdists, under a second — run it before every
ingest.

### 10.5 What this costs, and what it does not

> **CORRECTED 14 Sep.** The paragraph below claimed the Day 7 stability
> result "was never contaminated by this". **That was wrong**, and the
> measurement that disproved it is in this same section.
>
> The fold collapsed **13,694** repeats in total. transformers accounts
> for roughly 10,100 of them. The other **~3,600 were in python-docx,
> sympy, cython, pandas, matplotlib, mpmath, xlsxwriter and pyasn1** —
> every one of which was in the Day 6 dataset the Day 7 numbers came
> from. So **roughly 17% of that dataset was inherited duplicates.**
>
> transformers was the loud case, not the only one, and the claim was
> made after checking only the loud one. Duplicate rows inside a
> lambdarank group change the group structure the ranker optimises, so
> the Day 7 figures were not wrong so much as **unverified**. They have
> since been recomputed (§1): the result survived — `label_alias` still
> beats popularity at every measurable cut — but "it survived" is a
> finding, and "it was never at risk" was an assumption wearing a
> finding's clothes.
>
> The original text is kept below. The error worth remembering is not
> the number; it is checking the one package that was obviously
> implicated and generalising from it.

The transformers pair should fall from 9,863 rows to roughly 140. Every
other package with a base class and many subclasses shrinks too, by an
amount nobody has measured yet — that is the number the next run
produces, and `compare_runs.py` section 3 will light up with differences
this time, correctly, because the code genuinely did change the diff.

Two things this does **not** invalidate. The Day 7 stability result
(`label_alias` beating popularity at 7/7 cut dates, median lift 2.48×,
worst 1.79×) was computed on the 21,336-row dataset, before transformers
re-entered — so it was never contaminated by this. And the fold removes
only rows that were duplicates of a row we keep. But the positive rate
**will** move once 9,729 guaranteed negatives leave, and PR-AUC's floor
is the positive rate — so every number in §1 and §5 has to be recomputed
before it is quoted again, and the public write-up stays frozen until
then.

**What actually happened, measured:** 32,405 rows → **19,121**. The
transformers pair went 9,863 → **107**. Across 1,558 shared version
pairs, 63 changed and **every one of them fell** — zero pairs gained a
row, which is the property the fold had to have and now demonstrably
does.

### 10.6 The pre-release filter was hiding eight packages

`list_releases` skipped any version `packaging` calls a pre-release. That
rule is right almost everywhere — nobody upgrades to `2.0.0rc1`, so
diffing it would describe a change no user ever had.

`Version("0.65b0").is_prerelease` is `True`. OpenTelemetry's
instrumentation line has shipped `0.NNbM` for years and has never left
beta: **that suffix is their release.** The pipeline discarded every
version they have ever published, reported `TooFewReleases`, and moved
on. Eight packages, ranks 79 to 386, all publishing real non-yanked
sdists on every release.

The fix is a fallback, not a relaxation: take stable releases when a
package has two or more, and only then widen to include pre-releases.
A package with real releases never sees its rc builds; a package that
only ships betas stops being invisible. `is_prerelease` rides along on
every row so the fallback is visible in the data instead of inferred
from it. Dev releases (`1.2.3.dev4`) are excluded in **both** passes —
and that needs its own check, because `is_devrelease` implies
`is_prerelease`, so a single flag would have let dev builds in.

Verified on three packages before any re-ingest: `opentelemetry-util-http`
6 releases all flagged pre-release (the fix firing), `requests` 6 releases
0 flagged (the guard holding), `torch` still 0 (the fallback not papering
over a real absence).

### 10.7 What the 25 failures actually are

The question that prompted this: *transformers was excluded by a timeout —
could that be happening to others?* The answer is no as asked, and yes as
meant.

Not as asked: failures are **not** concentrated among big packages.
Median download rank of a failed package is 308 against 250 for the set;
3 in the top 100, 3 in the bottom 100.

As meant: the failure table's reasons had not earned their trust. Of the
three failures ever investigated, **two were misclassified**. transformers
was recorded as a 600s timeout and then finished in 3.1 minutes on retry —
the binding constraint was memory contention across ten parallel workers
loading 2,680-module trees, not elapsed time. `sniffio` was recorded as a
griffe `RuntimeError` and ran clean on re-run.

| | n | verdict |
|---|---|---|
| `types-*` stubs, C/Rust extensions | 8 | correct — `.pyi` or no Python at all |
| wheel-only (torch, triton, onnxruntime, playwright, psycopg-binary) | 5 | correct — griffe needs source |
| opentelemetry pre-release filter | 8 | **bug, fixed §10.6** |
| griffe cyclic-alias crash (numpy, multiprocess) | 2 | real, known |
| transformers, sniffio | 2 | misclassified, both recovered |

13 of 25 are real limits of the method and belong in the write-up as
such. The other 12 were recoverable and have been recovered. **A
distribution that ships no sdist cannot be analysed by reading source** —
that is a boundary, not a defect, and it is better stated than
discovered.

---

## 11. Measuring the measurements (Day 9)

Four bugs found on 14 Sep, none in the model, all in the things used to
judge it. A wrong model is a bad afternoon; a wrong instrument is every
number in the file.

### 11.1 The ablation was varying two things at once

Each ablation row chose its own tree count by early stopping, and the
counts came back **4, 10, 11, 27, 40, 47, 53, 59** — a fifteen-fold
range. "Removing blast radius costs 25% of PR-AUC" was then
indistinguishable from "that row happened to stop at 4 trees".

It was also still using the **holdout** stopping rule that §5.7 replaced
everywhere else, on a validation slice 2.1× denser in positives than
test.

Now every row is fitted on all of train at one CV-chosen count. The check
that it worked: the `everything` row went from 0.4365 to **0.5137**, the
shipped model's exact score. An ablation whose baseline disagrees with
the model it is ablating was never comparing anything.

### 11.2 A decision rule that flipped on 0.26% of the data

`lift_MIN` picked `label_alias` on 12 Sep and `label_scoped` on 14 Sep.
Between those runs: **50 rows added, out of 19,121.**

Both worst cases came from the `q=0.85` cut, which had **4 rankable
version pairs**. A pair is rankable at 10 only if it has more than ten
changes and at least one positive; precision@10 over four such pairs is
not a weak measurement, it is not a measurement. `MIN_TEST_POSITIVES`
already conceded that tiny cuts are invalid — it gated on the wrong
quantity.

`MIN_RANKABLE_PAIRS = 10`, checked before fitting:

| `lift_MIN` | with the 4-pair cut | without it |
|---|---|---|
| 12 Sep dataset | alias 1.94 wins | **alias 2.19 wins** |
| 14 Sep dataset | scoped 1.92 wins | **alias 2.32 wins** |

The ungated rule reverses; the gated one does not. Cut dates are
**quantiles**, so adding any rows moves all seven — two runs are not
comparable cut-by-cut, and a rule resting on the smallest cut inherits
all of that movement.

**This rule was added after seeing that it mattered, and that has to be
said out loud** — it is the same family of error as choosing features on
test PR-AUC (§5.4). Two things defend it. The justification never
references which label wins: four pairs cannot support a top-10 metric
whoever is being scored. And it is checkable against data collected
before the rule existed — the table above is exactly that check. It is
still a post-hoc choice and the viva answer is "yes, and here is why it
is not outcome-driven", not a denial.

The cost is honest: five or six usable cuts instead of seven. Say **"six
of seven, the other two too small to measure"**, never "6/6" alone.

### 11.3 An ablation table with no idea what zero looks like

`inherited_by` has **exactly zero gain** — the model was offered it and
never split on it. Dropping it moved PR-AUC by **12.6%**, because
removing a column changes LightGBM's binning and column sampling and the
fit moves whether or not the feature was used.

Meanwhile "removing path shape costs 10.4%" was being read as a result.
It is smaller than what happens when nothing is really removed.

So the ablation now fits the full model, asks which features it never
split on, and drops each as a **control** — discovered, never hardcoded,
since which features go unused changes with the label and the data. Their
cost is the noise floor, and every effect at or below it prints as
nothing.

```
NOISE FLOOR  12.6%
  path shape     10.4%   BELOW THE FLOOR — read as nothing
  reachability   26.7%   2.1x the floor
  path+reach     44.0%   3.5x the floor
  popularity     17.2%   1.4x the floor
  CONTROL  inherited_by  +12.6%
  CONTROL  is_private     +5.0%
```

**The floor is itself uncertain.** Two controls gave 5.0% and 12.6% — a
2.5× range on a two-point estimate. `max()` is the conservative choice,
but path shape at 10.4% lies *inside* that range: above one control,
below the other. The defensible sentence is **"path shape cannot be
distinguished from noise"**, not "path shape is nothing". The proper fix
is standard and not yet done: add several columns of pure random noise as
features and ablate each, giving a floor with a distribution rather than
two points.

What survives the floor: **reachability at 2.1× and path+reach at 3.5×.**
That is the finding. §5.3 was reaching for it and got there with a wrong
number ("105%"); this is the same conclusion with an instrument that
knows its own resolution.

### 11.4 The gain table was truncated, and it read as a claim

`imp.head(8)` on a 16-feature model. `inherited_by` was absent from the
printed table, which reads as "the model ignores it" — flatly
contradicting an ablation saying its removal cost PR-AUC. The
contradiction was in the *printing*.

All 16 print now, with shares, and any feature at zero gain is named
explicitly. A feature the model declined is the interesting case, not the
boring one, and it is exactly what `head(8)` hides once the feature set
passes eight.

### 11.5 Did the model improve, or did the data get honest?

Varad asked this on 15 Sep, and it is the right question: PR-AUC moved in
the same step that removed 13,694 rows, which is exactly the shape of a
number nobody should take at face value.

`scripts/fold_effect.py` answers it by holding everything still except
the duplicates — same label, same features, same CV stopping rule, and
**the same cut date**. That last part is the one that is easy to get
wrong: the normal split cuts at a *quantile of the rows*, so 32,405 rows
and 19,121 rows would cut at different dates and be scored on different
test halves. That comparison would measure the split as much as the fold.

```
                        before       after
  rows                  32,405      19,121
  floor                 0.0112      0.0356
  PR-AUC                0.4439      0.5137
  vs floor               39.6x       14.4x
  lift over popularity   4.88x       4.82x
  trees (CV median)         72          25
```

**Lift is flat. The model did not improve.** Two of those rows are traps
and both were nearly quoted:

- **PR-AUC rose 16%** — because the floor rose 3.18×. Removing guaranteed
  negatives raises the score with no change in skill.
- **vs-floor fell 39.6× → 14.4%** — and this is *not* a regression. A
  smaller floor mechanically inflates the ratio, the same way the method
  subset's 25.2× is not evidence the model prefers methods. Neither raw
  PR-AUC nor vs-floor is comparable across datasets with different
  positive rates. Lift over popularity is, because the baseline is refit
  inside each dataset and the floor cancels.

#### The line that ties this to §11.6

```
  test rows       15,182 -> 4,778
  test positives     170 ->   170
```

**10,404 test rows removed, not one of them a positive.** Every folded
row was a negative — which is precisely what the label's method
blindness predicts. One mechanism produces both findings:

> griffe reports a base-class change once per subclass → those rows are
> all methods → an import-based label cannot see methods → all 13,694
> are guaranteed negatives → the positive rate is dragged to 1.12% →
> PR-AUC reads lower than the model deserves.

The CV also chose **72 trees** on the amplified data against 25 now: a
model three times larger, to fit a dataset that was a third duplicates.

Caveat that applies to the whole subsection: the two runs share most of
their training data. This is a sensitivity check, not a significance
test. There is no p-value here and there should not be one.

### 11.6 The label cannot see methods, and that is 65% of the data

This started as Varad's question — *why does `inherited_by` have literally
zero gain, when a change hitting 3,242 classes is intuitively
high-impact?* — and the answer turned out to be much larger than the
feature.

```
  inherited_by = 0    18,604 rows    5.22% positive
  inherited_by > 0       517 rows    0.00% positive
```

Not "low". **Zero.** Against a 5.22% base rate you would expect about 27;
the odds of none by chance are around 10⁻¹². So it is structural, and the
structure is this:

```
  method on a class   12,373 rows    1.12% positive
  module-level         6,748 rows   12.34% positive
```

**Eleven times.** The usage index is built from **import statements**.
People write `from pandas import read_csv`. Nobody writes
`from pandas import DataFrame.append` — they call it on an object. So the
scanner sees module-level symbols and is close to blind to methods.

That 1.12% is not a fact about methods. `DataFrame.append` being removed
broke thousands of codebases. **It is a fact about what we can observe**,
and 65% of the dataset sits on the wrong side of it.

`inherited_by` has zero gain because it identifies a subgroup that is
100% negative *by construction*, and `public_depth` and friends already
push those rows down. The feature is fine. The label is narrow.

#### The worry that follows, and the test for it

`public_depth` is the top feature at 35.6% of gain. Module-level symbols
have low `public_depth`; methods have higher. So `public_depth` might be
encoding *"is this the kind of symbol our label can observe"* — which
would make the headline feature a proxy for our own blind spot, the same
class of error as §5.3 in a form §5.3 did not cover.

`scripts/label_blindspot.py` scores the shipped model separately on each
half — one model, trained on everything, split only at scoring time,
because training a specialist per subset answers "could a model do this?"
rather than "what is this model doing?".

| | rows | floor | PR-AUC | vs floor |
|---|---|---|---|---|
| module-level | 1,437 | 0.0905 | 0.6036 | 6.7× |
| **methods** | **3,341** | **0.0120** | **0.3021** | **25.2×** |

**It ranks inside the blind spot.** The model is ordering changes, not
sorting symbol kinds, so the result survives with the limitation stated.

Three numbers from that run that must **not** be quoted, two of them
produced by my own script before it was fixed:

- **"32× lift over popularity" among methods.** Popularity scores 0.0094
  there against a 0.0120 floor — *worse than chance*. Dividing by it
  manufactures a large number from an unstable near-zero denominator.
- **precision@10 among methods.** It rested on **3 rankable version
  pairs**, and §11.2 had just established that under 10 is not a
  measurement. The script quoted one anyway until it was fixed.
- **"25.2× beats 6.7×, so the model is better at methods."** No. A rarer
  positive class yields a larger ratio for the same real skill. Each
  number says only *within this group, far above chance* — they are not
  comparable to each other.

#### That popularity is worse than chance among methods is a finding

Big packages ship enormous numbers of methods and almost no labelled
ones, so sorting by download count **actively misleads** in exactly the
half of the data where the label is weakest. The baseline this project
exists to beat does worse than a coin flip on two-thirds of the rows.

#### What would fix it, and whose job it is

A Track B change, not a model change: the usage scanner would have to
record attribute-access call sites — seeing `df.append(...)` and knowing
`df` is a `DataFrame`. That is name resolution, not AST walking, and it
is not happening before the demo. A cheap partial version exists —
record `X.method()` wherever `X` is a name imported from a tracked
package, with no type inference at all — and is worth scoping.

Until then this is a **named limitation**, stated in the report rather
than discovered by an examiner: *our labels measure import-time usage.
Method-level breaking changes are systematically under-counted, and we
can put a number on it — 1.12% against 12.34%.*

### 11.7 What none of this changed

The result. `label_alias` beat popularity at every measurable cut before
these fixes and after them. What changed is that the numbers now come
from instruments that agree with each other: the ablation's baseline
equals the shipped model, the gain table is complete, the ablation knows
its noise floor, and the decision rule survives a 0.26% perturbation of
the data.

Worth keeping in view: **three of the four bugs in this section were in
code written to check the model, not to build it.** The instinct to
verify was right; the instruments needed verifying too — and §11.5 and
§11.6 were both written by *pointing an instrument at itself*, which is
the only technique in this file that has never yet produced a wrong
answer.

What §11.5 and §11.6 did change is the **claim**, not the result. The
ranker is as good as it was; the score it was previously given was
depressed by duplicate negatives, and the half of the dataset it is
scored on most heavily is the half our label can actually see. Both
belong in the report as stated limitations. Neither is a reason to
restate the headline in §1, which is still: *beats popularity at every
cut date large enough to measure, median 1.99×, never below 1.45×,
on the frozen dataset.*

---

## 12. What the analysed window actually contains (Day 10)

Varad's review note was fair: *"wheel-only releases are probably rare"
does not survive a review; "0.x% of releases, documented" does.*

`scripts/window_gaps.py` takes the version range actually analysed for
each of the 415 packages, asks PyPI for everything published inside it,
and accounts for every release that is **not** in `changes.csv`.

```
  releases inside our analysed windows   2,470
  in changes.csv                         2,081   84.3%

  no sdist                                   3   0.12%   the off-by-one
  yanked                                    38   1.54%   correct
  pre-release                              207   8.38%   correct (§10.6)
  dev release                              114   4.62%   correct
  left no row (analysed, nothing found)     27   1.09%   quiet releases
  unexplained                                0
```

### 12.1 The off-by-one is 0.12%, and it is one package

griffe reads source, so a release shipping only a wheel cannot be
analysed and a break introduced in it is attributed to the next release
that did publish an sdist. **Three releases in 2,470, all of them
`pure-eval`.** Real, documented, and small enough to state in a sentence
rather than hedge around.

### 12.2 The consistency check found the quiet-release gap

The `unexplained` bucket was paranoia: a release with a real sdist, not a
pre-release, not yanked, inside our window, that we did not analyse
should be **impossible**, because `list_releases` takes the last six
*eligible* releases and anything between them would have been among
those six.

It printed **27**. None has a logged failure; none appears anywhere in
`changes.csv`.

They were analysed and nothing broke. **A version pair with zero
breaking changes writes zero rows**, so a release whose *both* adjacent
pairs were clean disappears from the data entirely. `termcolor`,
`tabulate`, `toolz`, `pkginfo` — small stable libraries doing the right
thing, and the dataset cannot say so.

That is the same gap the quiet-releases question has been circling since
Day 5, arriving from a direction nobody was watching. A script written to
count wheel-only releases found it as a side effect of refusing to let a
category go unexplained.

**The 1.09% is a LOWER BOUND and must be quoted as one.** It counts
invisible *releases*, which need two clean pairs each. Clean *pairs* are
strictly more numerous and cannot be counted from outside at all.

### 12.3 Four of five enum values now have a source

Varad's `analysis_status` needs five values. After this run:

| value | source |
|---|---|
| `analysed` | `changes.csv` |
| `analysis_failed` | `failures.csv` |
| `no_source` | `window_gaps.csv` — the 3 above |
| `yanked` | `window_gaps.csv` — the 38 above |
| **`analysed_clean`** | **nothing. This is the outstanding work.** |

The fix is not a mapping. `run_ingest` has to emit a row per release it
**considered**, with a status and a change count, rather than only rows
for changes it found. Until it does, *"we checked and it is safe"* and
*"we never looked"* are the same absence — which is the one answer a
dependency tool should be able to give confidently and currently cannot.

---

## 13. The audit that nearly deleted 591 true findings (Day 11)

Migration 005 landed, the database reloaded against it, and then the
clean-up nearly did more damage than the mess it was cleaning.

### 13.1 A guard that fired on "unusual", not on "known bad"

`db_prune.py` refuses to delete more than 5% of the table. After the
reload it wanted **18.3%**, and refused.

It had no idea what was wrong. It only knew the number was far outside
normal, and that made a human look — which is the entire value of the
guard. Had the bug affected 4% of rows instead, the delete would have
gone through, ~900 true findings would have vanished, and **nothing
downstream would ever have complained**: a missing breakage row looks
exactly like a change that never happened.

**A guard that only fires on conditions you already thought of catches
nothing you have not already imagined.** This one fired on a shape.

### 13.2 The bug was a meaning mismatch, not a logic error

Both audit scripts decided whether a row was *superseded* — "we
re-analysed that release and did not produce this row" — using:

```python
live_rel = set(zip(package, version_to)) | set(zip(package, version_from))
```

That reads naturally: *is this release still in our data?* It is the
wrong question.

**Every breakage row hangs off `version_to`.** `breakage_rows()` attaches
each row via `release_id[(package, version_to)]`, so a row for release V
was produced by the pair `previous -> V`. "Superseded" can therefore only
mean *we re-ran that pair and it did not produce this row* — which
requires V to be a **version_to** today.

When the six-release window rolls forward, V becomes the OLDEST version
in it and appears only as a `version_from`. The release is still "live",
but the pair that produced its rows was never re-analysed. We know
nothing about those rows — and the script called them superseded.

Caught by reading the sample rows, not the summary. The breakdown said
*"4,198 superseded"* and looked entirely plausible. The twelve example
rows above it said `anthropic.AI_PROMPT: Public object was removed` — a
public, module-level removal, which a fold that only touches inherited
methods has no way to supersede. **Same failure as the Day 5 chain
renderer: a summary that was internally consistent and wrong, with only
the raw rows disagreeing.**

Fixing it moved **591 rows** from *delete* to *keep*.

| | before fix | after |
|---|---|---|
| superseded (delete) | 4,198 | 3,607 |
| aged out (keep) | 198 | **789** |

### 13.3 Proving the remaining 3,607 really were superseded

"Close to what I expected" is how three bugs survived this week, so the
3,607 got their own test rather than a nod.

If the fold caused a row to disappear, **the same change is still in
today's data** — same package, same release, same kind, same leaf name —
recorded on the defining class instead of the inheriting one. The leaf
survives; the path moves.

```
superseded rows: 3,607
  same package+version+kind+LEAF under a different path today:  3,586  (99.4%)
  no counterpart at all:                                            21
```

And the 21 are not mysterious either:

```
cython 3.3.0         runtests.TAG_EXCLUDERS
zstandard 0.24.0     make_cffi.HEADERS
grpcio-tools 1.82.1  grpc_version.PROTOBUF_VERSION
python-dateutil      updatezinfo.main
```

`runtests`, `make_cffi`, `grpc_version`, `protoc_lib_deps`,
`updatezinfo` — build and test scripts, removed by the Day 8 layout fix
(§10.1), which `compare_runs` section 3 had already named as its intended
removals. A second deliberate change, showing up in a different audit.

Every one of the 3,745 deleted rows is accounted for by a fix we made on
purpose. Final state: **19,909 rows — 19,120 current plus 789 aged-out**,
0 superseded, 0 un-importable, Varad's `breakrank-fixture` rows untouched.

### 13.4 The same confusion, from two directions, in one day

Varad rejected `analysed_clean` for the oldest release in a window and
added `no_baseline` instead, because *"recording it as analysed_clean
would have the site say 'safe to upgrade' about a release we never
compared"*.

That is **this bug**, arrived at independently from the schema side. A
release is not a unit of analysis; **a pair is.** A release with no
predecessor has nothing to be clean about, and a release that is only
ever a `version_from` has no rows of its own to supersede.

Two people found the same distinction on the same day from opposite ends,
which is the strongest argument available for keying `n_changes` on the
pair rather than the release.

### 13.5 Three loader fixes that shipped with the reload

- **`inherited_by` and `positive_rate` are written**, detected rather
  than assumed: `check_schema` returns a capability dict, and any optional
  column the database lacks is dropped from **both** the column list and
  the row dicts. A bound parameter with no column to land in fails a
  transaction exactly as hard as the reverse.
- **`positive_rate` is a field in `metrics.json`**, not a phrase inside
  the notes string. A floor recoverable only by parsing prose is half a
  result. A fallback still parses the old string, because a run produced
  by an older `train.py` is still a valid run.
- **`trained_at` is no longer set to `now()` on conflict.** Re-scoring an
  *older* model version would have stamped it newest, and the API picks
  the current model with `ORDER BY trained_at DESC LIMIT 1`. This is the
  exact mirror of the `DEFAULT now()` bug Varad found in his seed, where
  the fixture silently became newest on every reseed. Same bug, opposite
  direction, both fixed while there is still only one real model.

---

## 14. "We checked and nothing broke" is now an answer (Day 11)

`data/releases.csv` — one row per release **considered**, not one per change
found. This is the file that closes the gap §12.2 found from the outside
and Varad's `analysis_status` enum was designed around.

```
analysed        1,601  ┐ 2,405 pairs actually diffed
analysed_clean    804  ┘
no_baseline       487    one per segment: nothing to compare against
analysis_failed    59
pre_release       262  ┐
dev_release       148  │ 499 filtered before analysis, reason kept
yanked             54  │
no_source          35  ┘
                -----
                3,450  release rows, 3,450 distinct — the ledger closes
```

### 14.1 The number that did not exist

**804 clean pairs.** `scripts/window_gaps.py` could see only **27** from
the outside and said so explicitly: a release only becomes invisible when
*both* its adjacent pairs are clean, so the count of clean releases is a
lower bound on the count of clean pairs. It is 30× larger.

That caveat was written before the number was knowable. Being right about
the *direction* of an unmeasurable quantity is worth more than a guess at
its size, and stating it as a bound is what made the 804 verifiable
rather than surprising.

### 14.2 Two independent paths, one number

```
n_changes sum   17,014
changes.csv     17,014
```

`n_changes` is accumulated per pair inside the ingest; `changes.csv` is
written row by row. Nothing links them except both being correct. An
exact match is the check worth having before a homepage ranks releases by
that column.

### 14.3 `no_baseline` is per SEGMENT, not per package

The oldest release in a chain has no predecessor, so it is not clean —
nothing was compared. But a failed download *splits* the chain (§2.1),
and the release immediately after a gap has no comparable predecessor
either. Both get `no_baseline`.

487 of them across 500 packages: more than one per package, which is the
failed downloads showing up exactly where they should.

Varad rejected `analysed_clean` for this case before seeing any of the
data, on the grounds that it *"would have the site say 'safe to upgrade'
about a release we never compared"*. That is the same pair-not-release
distinction as the `live_rel` bug in §13.2, caught from the schema side
instead of the audit side.

### 14.4 The bug in shipping it, and the check that should have caught it

Changing `_process_package` to return four values instead of three, I
updated three of its five return paths. The two I missed were the
early exits for "every download failed" and "no importable module" — so
8 packages died with `ValueError: not enough values to unpack` and were
recorded as `pipeline / UnexpectedError` instead of
`resolve_module / NoPythonModule`.

The run still completed, because `main` catches per-package exceptions by
design. **A pipeline built to survive one package exploding will happily
survive its own author breaking every path that explodes.**

Fixed, and verified with an AST walk over the function rather than by
testing one path:

```python
sizes = {len(n.value.elts) for n in ast.walk(fn)
         if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)}
assert sizes == {4}
```

Arity is a structural property, so check it structurally. Running the
happy path proves nothing about the five error paths, and the error paths
are where a resumable pipeline spends its interesting moments.

---

## 15. The comparison that graded on different exams (Day 11)

`lift_MIN` has now produced a misleading answer **three times**, and the
third one finally showed what the other two had in common.

### 15.1 What happened

On the 17,014-row dataset the rule chose the strict `label`:

```
       label  splits  lift_MIN
       label       5      1.81   <- "winner"
label_scoped       6      1.27
 label_alias       6      1.45
```

**Five splits against six.** `label` was skipped at q=0.85 for having 9
rankable pairs, so it never faced the 2026-08-22 cut — which is exactly
where `label_alias` recorded its worst case of 1.45×.

`lift_MIN` is a **worst case**. Comparing worst cases across different
sets of cuts rewards whichever label was spared the hardest one. On the
five cuts all three labels actually share:

| | 05-10 | 06-05 | 07-08 | 07-28 | 08-10 | min |
|---|---|---|---|---|---|---|
| `label` | 1.86 | 1.81 | 2.13 | 1.90 | 2.23 | 1.81 |
| `label_scoped` | 1.58 | 1.27 | 1.38 | 1.30 | 1.44 | 1.27 |
| **`label_alias`** | 1.94 | 1.92 | 2.05 | 2.25 | 2.76 | **1.92** |

`label_alias` wins, as it has on every dataset once the comparison is
made fairly. The code now restricts the label-vs-label table to shared
cuts and says so when they differ; per-label tables still use every cut
that label could use, because that is the honest picture of each one.

**Two numbers, two jobs, do not mix them.** Which label ships is decided
on shared cuts (1.92×). How good the shipped model is comes from
`label_alias` on its own six cuts (6/6, median 1.99×, min 1.45×) — that
is what `metrics.json` carries and what the report quotes.

### 15.2 What all three failures had in common

| | §11.2 | §13.2 | §15.1 |
|---|---|---|---|
| where | stability | db audit | stability |
| looked like | a model comparison | a row classification | a model comparison |
| actually was | a measurement of which cut dates existed | a measurement of which releases were re-analysed | a measurement of which cuts each label could use |

Every time, **a number that appeared to measure the models was partly
measuring which rows each model was given.** That is one bug wearing
three costumes, and it is worth naming as a class rather than fixing
three times: *before comparing two things, check they were given the same
question.*

### 15.3 Still open: the tree count is being clamped, not chosen

```
CV folds chose [5, 7, 1, 3] trees -> median 20
```

The median of `[5, 7, 1, 3]` is **4**. `MIN_TREES = 20` clamped it, and
the printed line hides that. The folds wanted a four-tree model, which is
not a tree-count decision — it is the validation signal being too weak to
make one.

Every ablation row inherits that clamp, which is why the 16-feature model
now loses to two of its own subsets again (`no path shape` 0.4441 and
`no blast radius` 0.4292 against 0.3866) and why the noise floor reads
**11%** off a single control feature. Nothing in that table below 11% is
readable, which currently includes popularity at 8.5%.

Not fixed. Named, so it is not rediscovered as a finding.

### 15.4 The dataset moves faster than the write-up can

Four ingests in five days. The headline lift has read **2.48×, 2.59×,
4.82×, 2.25×**, and not one of those changes came from a modelling
decision — they came from PyPI publishing releases. `databricks-sdk`
alone swung the dataset by 1,901 rows overnight when its six-release
window rolled forward.

The window is defined relative to *today*, so the dataset is not a fixed
object. That is correct for a live site and impossible for a report with
a submission date: an examiner re-running the pipeline in November gets a
different answer, and every figure in §1 is stale within days of being
written.

**So the dataset is frozen at 17,014 rows / 416 packages, 16 Sep 2026,
tagged `dataset-20260916`.** Every number in the report and the public
write-up cites that tag. Re-ingests continue for the live site and are
never quoted. Separating "the dataset the report is about" from "the
dataset the site serves" is the only way both can be honest.

## 16. The two features that need more than one release (Day 12)

Every feature in the model so far can be read off a single diff. These
two cannot. `was_deprecated_before` needs the release *before* the break,
and `prior_breaks_in_module` needs every release before that. The project
book assigns them to week 5 and calls the first one "probably your single
strongest feature."

Both are written at **extract time**, in `api_extract.diff_series`, and
that is not a style choice. Once a pair is diffed the two sdists are
deleted; the deprecation marker is gone with them. Nothing in
`changes.csv` can reconstruct it later. So the feature had to exist
before the re-ingest, not after — otherwise the re-ingest happens twice.

### 16.1 griffe's `deprecated` flag is None for real deprecations

The obvious implementation is `obj.deprecated`. Measured on griffe 2.2.0,
against a two-line fixture:

| marker in source | `griffe.deprecated` |
|---|---|
| `.. deprecated:: 1.0` in the docstring | `None` |
| `@deprecated('use plain instead')` decorator | `None` |

The first is defensible — prose is not metadata. The second is not
obvious at all, and it is the one that matters, because a decorator is
the machine-readable form a maintainer is *supposed* to use.

A feature reading only that flag would have been `False` on all 17,014
rows. It would have reported zero gain, the ablation would have shown
removing it costs nothing, and the honest-sounding conclusion —
"deprecation warnings do not predict which breaks bite" — would have been
a fact about a broken reader.

So `was_deprecated()` reads three sources: the flag, the decorator list,
and the docstring. Decorators come back as **strings** on this version
(`decorators=["deprecated('use plain instead')"]`), so they are matched
as text against the stem `deprecat`, which covers "deprecated",
"deprecation", "DeprecationWarning" and `.. deprecated::` at once.

### 16.2 The version that put the signal on the wrong rows

Aliases must be skipped: reading `.docstring` on one raises
`AliasResolutionError` — the same trap §9.1 documents, and the first
probe crashed on it exactly as predicted.

Skipping them naively gave this:

```
pkg.core.a   True      <- the definition path
pkg.core.b   True
pkg.a        False     <- the path users actually import
pkg.b        False
```

Correct by the letter. Also useless: the signal landed on the rows almost
nobody writes, and was withheld from the rows the feature exists for.

Fixed with `deprecated_paths()` — one walk per version collecting every
*definition* path that carries a marker, consulted by alias rows through
`alias.target_path`, which is a plain string and needs no resolution
(§9.1 again, used constructively this time).

Worth recording *when* an alias row exists at all: griffe reports the
re-exported name as its own breakage only if the package declares it in
`__all__`. Measured — `from .core import a` yields `['pkg.core.a']`;
adding `__all__ = ['a']` yields `['pkg.a', 'pkg.core.a']`. The first
fixture for this test had no `__all__`, produced no alias rows, and would
have passed by having nothing to check.

### 16.3 `prior_breaks_in_module` must not see the future

The counter accumulates oldest-first inside `diff_series`, so at pair *k*
it holds pairs 0..*k*−1 only. Building it from a completed run and
reading it back per row would leak later releases into earlier ones — the
same class of error as a random split, and just as invisible in the
output. A feature that can see the future scores beautifully and predicts
nothing.

`scripts/test_history.py` pins this: 13 checks, no network, under a
second. It includes the controls, which are the point — an unmarked
symbol that stays `False`, and a function *named* `deprecated` carrying
no marker, which stays `False` too. Without those, a function returning
`True` unconditionally passes every other check in the file.

### 16.4 A summary that disagreed with its own table

Adding the `HISTORY` group to `ablate.py` printed this:

```
history only     0.4003   ...   0.60
...
No single group reaches 85% of the full model (best is 'per-change only' at 35%)
```

Both lines came from the same script. The table iterates the `runs` dict;
the sentence underneath held its own hand-typed list of group names, and
the new group was in one and not the other. A summary that can contradict
the table above it is worse than no summary — the table is skimmed, the
sentence is quoted.

Both the ablation loop and the "best solo group" line now derive their
names from `runs` itself. Same fix, one more time, as §11/§13/§15: a
number that looked like it described the model partly described which
rows or names the code happened to be given.

Also fixed while there: the noise-floor multiple printed `infx the floor`
when every control came back exactly flat. There is no multiple of zero;
it says so now.

### 16.5 What went to the database, and what did not

Nothing. `BREAKAGE_COLS` is unchanged, there is no migration, and Varad's
contract ("do not add, rename or drop columns without telling me first")
is untouched.

One thing was added inside the existing `detail` JSON column:
`deprecated_before: true`, and **only when true**. A key written on every
row would be a claim on every row, and a `changes.csv` from before 16 Sep
cannot tell "we looked and found nothing" apart from "we never looked".
Writing it only when the answer is yes makes its absence mean nothing,
which is honest for both files — and gives the site a sentence worth
showing: *the maintainer marked this deprecated in the previous release.*

### 16.6 The version of this feature I nearly shipped was half noise

`was_deprecated()` originally matched one stem — `deprecat`, case
insensitive — anywhere in the decorator text and anywhere in the
docstring. The reasoning written next to it was that authors spell the
word a dozen ways, so a stem is thorough rather than lazy.

It is thorough. It is also answering a different question: *does this
symbol mention deprecation*, when the feature needs *is this symbol
deprecated*. Those are not close.

**Decorators.** Surveyed across sqlalchemy, pydantic, numpy, django and
pandas: 297 decorators contained the stem.

| decorator | hits | what it means |
|---|---:|---|
| `@pytest.mark.filterwarnings("ignore::DeprecationWarning")` | 158 | a test *silencing* deprecation noise |
| `@deprecate_nonkeyword_arguments` / `@deprecate_kwarg` / `@deprecate_posargs` | 47 | a calling convention is going, the function is not |
| `@pytest.mark.parametrize(..., DeprecationWarning)` and friends | 16 | the stem is in an *argument* |
| `@deprecated`, `@util.deprecated`, `@typing_extensions.deprecated` | 64 | the real thing |
| `@util.deprecated_params` | 9 | a parameter, not the symbol |

Roughly three noise hits for every real one, and the largest single
category means close to the *opposite* of what the feature records.

The fix is not a list of exceptions. It is reading the decorator's NAME —
the part before the paren — instead of the whole call. `pytest.mark.
filterwarnings` does not contain the word; only its argument did. All 158
disappear without a rule that mentions pytest.

**Docstrings.** Across 41,914 symbols in twelve packages the stem fired on
symbols like these:

| symbol | matching text | what it is |
|---|---|---|
| `sqlalchemy.exc.SADeprecationWarning` | "Issued for usage of deprecated APIs." | the warning class itself |
| `click.core.Command` | ":param deprecated: ..." | documents a parameter *named* deprecated |
| `numpy.linalg.qr` | "'economic' mode is deprecated" | a mode, not the function |
| `boto3.compat.filter_python_deprecation_warnings` | the name and docstring both | machinery for handling deprecations |

So `_DEPRECATED_DOC` matches declaration shapes instead: the Sphinx
`.. deprecated::` directive, the reST `:deprecated:` field, a
`Deprecated:` section header, an ALL-CAPS shout, MkDocs' `!!! warning
"Deprecated"`, "deprecated since/in/as of", and plain self-scoped prose —
"This class is deprecated" counts, "This parameter is deprecated" does
not.

**Result, measured over the same 41,914 symbols: 262 hits under the old
rule, 147 under the new one. 44% of what the feature would have reported
was noise.**

Two smaller things, both mine:

*The ALL-CAPS branch was not all-caps.* Written `^\s*DEPRECATED\b` under a
pattern-wide `(?i)`, it matched any line merely beginning with the
lowercase word — which a wrapped `:param:` description does the moment
"deprecated" lands at the start of a continuation line. That is exactly
the `click.core.Command` false positive, reproduced by a case-
insensitivity I had added myself. `(?-i:DEPRECATED)` scopes the flag off
for that branch alone.

*The first fixture for the noise cases used `from .core import *`.*
griffe records a star-import as a single pseudo-member named `pkg/core/*`,
so the pair produced one row for the wildcard instead of one per symbol,
and all six checks came back `None` rather than `False`. A test whose
subject does not exist does not fail — it returns nothing, and `None`
is not `False` only because `check()` compares exactly.

None of this would have shown up in training. A feature that is 44% noise
still correlates with the label, still reports gain, still survives an
ablation. It would have gone into the report as "deprecation is the
strongest signal we have", and roughly half of what it was reading would
have been libraries talking about deprecation rather than doing it.

`scripts/test_history.py` case 4 now pins every shape in the tables
above, with one genuine `@deprecated` in the same fixture so the case
cannot pass by the function simply returning False.

### 16.7 What is still wrong, measured and left alone

Two residuals, both known, both quantified rather than discovered later.

**A directive that scopes to a parameter.** attrs writes
`.. deprecated:: 24.1.0 *hash*` inside `define`'s docstring — the Sphinx
directive is symbol-level everywhere else, and here it names a parameter.
`attr._next_gen.define` and `attr._make.attrs` therefore read True while
being perfectly current API.

Measured across the same twelve packages: **3 of 147 flagged symbols, all
three in attrs.** It is one library's house style, not a pattern. A regex
branch for it would be tuning the extractor to one package's docstrings
to move 2% of one column — so it is written down instead. If it ever
matters, the rule is "a `.. deprecated::` directive followed immediately
by an emphasised single identifier is about that identifier", and it
would only ever turn True into False.

**The feature is rare.** Across six packages at ten versions each — 158
breakage rows — exactly **3 carried a prior deprecation marker**, all in
attrs (`attr.validators.provides` and `attr._make.attrib`, plus the false
positive above).

That is worth saying before the model runs, because the project book
calls `was_deprecated_before` "probably your single strongest feature"
and a ~2% positive rate is not what a strongest feature usually looks
like. Two readings, and the run will separate them:

  - **It is rare but sharp.** When a maintainer does warn, the removal
    really is less likely to break anyone, and LightGBM can split on a
    2% feature if the split is clean.
  - **It is rare and therefore weak.** A feature true on ~400 of 20,000
    rows cannot move a ranking metric much no matter how right it is.

Either way it should be reported as *what the data supports*, not as
what the book predicted. The `HISTORY` ablation group exists to answer
this with a number: "history only" against "popularity only" on the same
fixed tree count.

Also unchanged, and named in `was_deprecated`'s docstring: a bare
`warnings.warn(..., DeprecationWarning)` in a function body with nothing
in the docstring is not detected. That needs the function's source, not
its signature. It is a false negative in every case, so the feature
under-claims rather than over-claims — which, after §16.6, is the
direction to be wrong in.

## 17. The pipeline had permission to run the code it downloaded (Day 12)

macOS raised **"Malicious Script Blocked — a script was blocked because
it contains malware. This script did not harm your Mac"** during two
consecutive ingests over the top 500. Both times it appeared while the
same cluster of packages was in flight (`datasets`, `pypdf`, `sympy`,
`transformers` and neighbours), and the first of those runs died at
package 256 with no crash report and no jetsam event.

The cause was one missing argument in this repo.

### 17.1 What griffe does by default

griffe reads a package two ways. **Static** analysis parses the `.py`
files as text and never runs them. **Dynamic** analysis imports the
module and inspects the live objects — and importing a module runs every
line at its top level.

`GriffeLoader.__init__` defaults `allow_inspection` to **True**.
`load_all_modules` did not override it. So from the first ingest until
16 Sep 2026, the pipeline had standing permission to import code it had
downloaded from PyPI seconds earlier.

`loader.py:600-606` is the whole decision:

```python
elif module_path.suffix in {".py", ".pyi"}:
    module = self._visit_module(...)      # parse the text
elif self.allow_inspection:
    module = self._inspect_module(...)    # IMPORT the binary
else:
    raise LoadingError("Cannot load compiled module without inspection")
```

So the trigger is **a module whose file is not `.py` or `.pyi`** — a
compiled extension. Reproduced on a fixture: a package containing
`ext.cpython-311-darwin.so`, loaded with the default, calls
`_inspect_module` on that path. That is a `dlopen` of a native binary
extracted from a downloaded archive, which is exactly the shape of thing
XProtect's behavioural monitor exists to stop.

sdists are not supposed to contain binaries. They are tarballs, and
plenty do: vendored libraries, prebuilt artifacts, test fixtures.

### 17.2 Three wrong fixtures before the right one

Worth recording, because the first three all "passed":

| fixture | reaches the import fallback? |
|---|---|
| a package that parses cleanly | no — with the flag on or off |
| a module body with a side effect | no — it parses, so it is never inspected |
| a **SyntaxError** | no — griffe raises `LoadingError` instead |
| a **compiled extension** | **yes** |

The intuition "unparseable means it falls back to importing" is wrong on
griffe 2.2.0, and a test built on it passes while proving nothing. The
same trap as the star-import fixture in §16.6: a check whose subject does
not exist does not fail, it returns nothing.

`scripts/test_no_execution.py` therefore ends with a check that the test
can fail — it runs the same fixture through griffe's own default and
asserts that the import IS attempted. A guard that cannot be seen to fail
is not a guard.

### 17.3 What this cost, and what it did not

No evidence anything was harmed. macOS said it blocked the script, and
the machine kept working. But the honest statement is the one about
permissions, not outcomes: **for four ingests the pipeline was allowed to
execute third-party code, and whether it ever did is not something the
data can now answer.** Absence of a crash is not absence of execution.

The fix is one argument in two places, plus the standing rule that it is
never removed "temporarily" to debug a package that will not load. The
cost of turning it off is known and small: a compiled module produces no
rows instead of inspected ones. A gap in coverage is a different category
of thing from an executed binary, and no dataset is worth the trade.

### 17.4 The diagnosis that was wrong three times first

For the record, because the wrong answers were each stated with more
confidence than they deserved:

1. **"It's macOS scanning the extracted files."** Plausible, unverified.
2. **"It's a memory kill."** Disproved by the user's own logs — no jetsam
   event, no crash report. Asserted before checking.
3. **"It's Brave's download protection, not macOS."** Wrong, and the
   worst of the three: it told someone a security warning was not from
   their operating system when it was. The screenshot settled it in one
   glance — system alert styling, `?` help button, XProtect's exact
   wording.

Only the fourth attempt started from "read our own code and find out what
it is permitted to do", which is where it should have started, because
that is the one question we control the answer to. The user's machine
was running the unfixed pipeline through all three wrong answers.

## 18. "Malicious Script Blocked" was transformers' remote-code feature (Day 12)

Every silent death of the top-500 run on the Mac — at package 256, then
232, then again under `screen` — was macOS XProtect blocking what it calls
`ScriptedMalware` and killing the process. The kill is a SIGKILL to the
process group, which is why it left no Python traceback, no jetsam event
and no crash report: the exact fingerprint we kept failing to explain.

The false trails, recorded because each was stated with more confidence
than it earned: a memory kill (disproved — no jetsam for python), Brave's
downloader (wrong — the alert is macOS's own, `type=ScriptedMalware` in
`syspolicyd`), and a bundled EICAR test file (a scan of ranks 225-275
found none).

### 18.1 Finding it: one package at a time

`scripts/find_culprit.py` does what the parallel run cannot — processes
the danger-zone packages **one at a time**, in rank order, writing the
package name to disk *before* touching it. With no parallelism there is no
ambiguity about which package was in flight when the kill lands.

It died on **transformers (rank 225)**, every earlier package clean. The
completion counters from the full runs (256, 232) were a red herring all
along: transformers is huge and slow, so by the time a worker finished
loading its 2,680-module tree and tripped XProtect, 200-plus smaller
packages had already completed. Same trigger every time, different
bystander count.

### 18.2 What macOS actually objects to

transformers 5.17.0's sdist has 2,714 files and **not one** carries a
suspicious extension — no shell scripts, no binaries. The trigger is a
`.py` file's *contents*: `src/transformers/dynamic_module_utils.py`,
whose job is the `trust_remote_code` feature. Line 308:

```python
module_spec.loader.exec_module(module)
```

preceded by `get_cached_module_file` pulling a `.py` from the HuggingFace
Hub into `HF_MODULES_CACHE`. Download remote code, then execute it — which
is, byte for byte, the behaviour a malware heuristic exists to catch. It
is not malware; it is one of the most-downloaded libraries on PyPI doing
something it documents and warns about. XProtect cannot tell the
difference from a static signature, and neither could any scanner.

This is a hypothesis, not a proof — XProtect's rules are private and its
detection cannot be reproduced off a Mac. But it is specific, it fits
every observation (why transformers, why "scripted", why "did not harm
your Mac", why the identical block each time), and it is a far better
answer than "a false positive on the run's pattern".

The irony worth stating: this project exists to read other people's code
without running it, and it was killed by a library whose feature is to
run other people's code. We were never in danger — `allow_inspection=False`
(§17) and `ast.parse` mean we only ever read `dynamic_module_utils.py` as
text. macOS flagged the file at rest, on write, not anything we did with it.

### 18.3 The fix, and honest accounting

`run_ingest.py` gains an `XPROTECT_BLOCKED = {"transformers"}` set. Blocked
packages are excluded on macOS, **recorded as a `skipped` failure row**
(never a silent gap), and marked done so a resume does not re-hit them.
`BREAKRANK_NO_SKIP=1` disables the list — which is how Linux runs, where
XProtect does not exist and transformers processes normally.

So transformers is not lost. Its rows are backfilled from the Linux
ingest (the cloud sandbox), where the same code reads the same file
without a scanner in the way, and concatenated into the dataset. The
skip is a macOS-only workaround, not a data decision, and the released
dataset contains transformers exactly as if the Mac had never balked.

## 19. The history features, measured (Day 13)

Dataset: 17,802 rows, 250 packages, one code path (fixed griffe, history
features, transformers skipped on macOS per §18). Strict label: 788
positives (4.43%). Temporal split at 2026-04-13. All numbers below are
from this run and are not comparable cut-by-cut to the frozen
`dataset-20260916` — different rows, different floor (§11.2, §15.4).

The model holds: PR-AUC 0.2375 (range 0.216–0.265 across 7 cut dates),
**1.68× lift over popularity, beating it at 7 of 7 cuts** and beating
the semver kill-date gate at 7 of 7. That is the robust claim.

### 19.1 `was_deprecated_before` — the book's "single strongest feature" — is a dud

| | |
|---|---|
| rows where True | 199 of 17,802 (1.1%) |
| gain share | **0.1%** — 15th of 18 features, above only `is_private` and the two zero-gain columns |
| positive rate when True | 5.03% |
| positive rate when False | 4.42% |

That last pair is the finding. A deprecated-then-removed symbol is used
downstream at the **same rate** as everything else — 5.0% against 4.4%,
on 199 rows, is no difference at all. The book's intuition was that a
maintainer's warning gives users time to migrate, so the removal breaks
fewer people. In this data it does not show up. Either people do not
act on deprecation warnings, or the ones who would have been broken
migrated *and are still counted as users* because the usage index reads
the latest version of downstream code (§4). Both are plausible; neither
rescues the feature.

It is not a broken reader. §16.6 tightened the matcher against 41,914
symbols and it fires correctly on real deprecations (`attr.validators.
provides`, urllib3's `format_header_param`). The feature works. The
*signal* is not there.

Reported as a negative result, which is worth more in a viva than a
confirmation: the feature the book was most confident about was built,
tested honestly, and found to contribute nothing. That is what testing a
hypothesis looks like.

### 19.2 `prior_breaks_in_module` — the one nobody bet on — works, and non-monotonically

Gain share 4.2%, 8th of 18. Modest, and real. The reason it earns a slot
is the shape of it:

| prior breaks in the module | positive rate | n |
|---|---|---|
| 0 | 3.38% | 10,162 |
| 1–5 | 8.86% | 2,970 |
| 6–20 | **10.13%** | 1,717 |
| 21+ | **0.27%** | 2,953 |

A first break in a quiet module is unremarkable. A break in a module
that has been *moderately* churning is **three times** as likely to hit
something people use — those are the actively-developed public
surfaces. And a break in a module with 21+ prior breaks is almost never
used by anyone: that is the giant-refactor signature, a library
rewriting its internals on symbols nobody imports (the same population
as the inherited-member repeats in §10.2 and `pandas.tests` in §4).

That U-shape is exactly what a tree model can exploit and a linear one
cannot, and it is a real property of how libraries evolve, not of how
the label is built.

### 19.3 What the model is actually made of

| feature | share |
|---|---|
| is_version_string | 20.2% |
| name_length | 14.4% |
| package_churn | 14.3% |
| module_depth | 10.2% |
| package_rank | 10.0% |
| kind | 8.0% |
| release_size | 7.9% |
| prior_breaks_in_module | 4.2% |
| … | |
| was_deprecated_before | 0.1% |
| inherited_by, bump | 0.0% |

Path shape and popularity, with the version-string flag on top. The
history idea contributed one modest, interpretable feature and one
null result.

### 19.4 The ablation is unreadable below 12.2%, and it says so

`ablate.py` reports a **noise floor of 12.2%**: removing `inherited_by`,
a feature the model never split on, moved PR-AUC by 12.2%. Every
per-group effect except two sits under that line, and the two that
clear it (`no history` −14.6%, `no popularity` −13.5%) clear it by a
hair. This is the `MIN_TREES = 20` clamp named in §15.3, still not
fixed: at 20 trees the fit is unstable enough that the ablation cannot
resolve anything smaller than a tenth of the score. The gain table and
the positive-rate breakdowns above are the evidence; the ablation
confirms only that history is not load-bearing.

One line in it deserves a note rather than a headline: "path shape
only" (3 features) scores 156% of the full model. Read with the noise
floor in mind, that is a real signal that 18 features at 20 trees is
over-featured for this dataset size, and a reason to revisit the tree
count — not a reason to ship a three-feature model.

## 20. The full top-500 ingest, and everything re-measured on it (Day 14)

The sandbox fill finished: all 500 packages attempted, `done.txt` = 500.
314 produced rows. **Correction (Day 15):** this section originally said
the other 186 were "wheels-only, namespace or pure-data packages with no
public Python API." That was an inference, never checked, and it is
wrong. The true accounting, from `failures.csv` and `done.txt`:

| outcome | packages | evidence |
|---|---|---|
| produced rows | 314 | `changes.csv` |
| analysed, zero breaking changes in window | 27 | in `done.txt`, absent from both other files (certifi, mdurl, …) |
| worker process killed (`BrokenProcessPool`) | 149 | all 149 `pipeline` failures; tracebacks in `ingest_fill.log`; memory pressure on the 2-core container — includes torch, transformers, reportlab; **retryable** |
| no importable module (compiled-only / not Python) | 4 | `resolve_module` |
| griffe error (numpy `CyclicAliasError` + 3 others) | 4 | `griffe`; numpy is NOT fixable by memory — needs a griffe workaround |
| fewer than 2 source releases | 2 | `list_releases` |

The 149 crashed packages are tail-biased (median download rank 426 vs
172 for successes; none of the top-50), so head coverage is intact. The
final dataset is **23,268 rows / 314 packages**, against 39,154 usage
symbols — §19 was measured on 281 / 21,285, so every headline number
was re-run rather than trusted.

Pipeline, in order, on the complete `changes.csv`: `labels.py` →
`build.py` (18 features, temporal split at 2026-06-13, 2,036 version
pairs) → `baselines.py` → `train.py` → `stability.py --label label` →
`ablate.py`. One trap worth recording: `train.py` prints a "across 6 cut
dates" line by READING the previous `stability_label_cv.csv`, so run in
this order it quoted the stale 281-set lift (1.45×). The number below is
from the fresh stability run, which is the only honest source for it.

### 20.1 The result got stronger, not weaker

Medians across the 6 usable cut dates (q=0.85 skipped, 6 rankable pairs):

| | 281 pkgs (§19) | 314 pkgs (full) |
|---|---|---|
| model PR-AUC | 0.206 | **0.236** (range 0.168–0.271) |
| popularity | 0.145 | 0.135 |
| semver gate | 0.083 | 0.058 |
| floor (positive rate) | 0.045 | 0.048 |
| lift vs popularity (median of per-cut ratios) | 1.45× | **1.78×** (1.16–1.97) |
| beats popularity / semver | 6/6, 6/6 | **6/6, 6/6** |
| precision@10 / nDCG@20 | — | 0.172 / 0.560 |

The semver gate now sits barely above the random floor — the version
bump is constant inside a release and cannot order anything, which is
the project's thesis stated as a number. Overall positive rate over all
rows is 3.69% (test-slice floor 4.8%; the two differ because the
temporal split concentrates positives unevenly — both are reported,
neither is "the" base rate without saying which).

### 20.2 Both findings hold, and sharpen

`was_deprecated_before`: 278 / 23,268 rows carry the marker. Used
downstream 3.60% when deprecated vs 3.69% when not — now marginally
LOWER, i.e. zero signal. Gain share 0.1%, the lowest of every feature
the model split on (`inherited_by` and `bump` are at 0.0% and were
never split on, so "dead last of the real features" is the exact claim).

`prior_breaks_in_module`: gain share **4.7% → 6.3%**, 7th of 18. The
U-shape is unchanged in form: 0 → 2.58% (n=14,387), 1–5 → 8.21%
(3,617), 6–20 → 8.69% (2,107), 21+ → 0.25% (3,157). Middle buckets are
3.2–3.4× the quiet bucket. A feature that grows with the dataset while
the rival shrinks to nothing is the cleanest version of §19's story.

Gain table, full set: is_version_string 22.6, package_churn 17.1,
module_depth 12.5, name_length 11.0, kind 8.0, package_rank 6.5,
prior_breaks_in_module 6.3, release_size 6.1, public_depth 4.3,
is_dunder 2.6, then the tail under 1%. module_depth and name_length
swapped places vs §19; nothing else moved order.

### 20.3 The ablation is still unreadable, and is not quoted

Noise floor at the primary cut rose to **16.8%** (control: dropping
`bump`, zero-gain, moved PR-AUC +16.8%). "path shape only" reports 116%
of the full model. Same diagnosis as §19.4 and §15.3: 20 trees is too
few for group ablation to resolve anything. The review deck, report and
Q&A deliberately quote only the cross-cut medians and the gain shares,
and the report's limitations now say so in one line so a panelist who
runs `ablate.py` is not surprised.

### 20.4 What changed in the deliverables

Deck rebuilt to the supervisor's required outline (Title, Problem
statement, Objectives, Introduction, Literature review, Methodology,
Block diagram, Work done till date, Weekly plan), 15 slides. Literature
slide cites Raemaekers et al. (Maven, 2014–17), Xavier et al. (SANER
2017), Decan & Mens (IEEE TSE 2019) and griffe — all four verified
against their publisher pages before being written in. Weekly plan no
longer lists "finish the ingest" (done); week 4 became the label-variant
experiment `labels.py` has been asking for since Day 3. Every number in
deck, report and Q&A is now the §20.1–20.2 figure.

## 21. Audit findings and the fix list (Day 15)

Two audit passes over the code and data after the mid-semester review
materials were built. Every number below was computed against
`features.csv` / `labelled.csv` as they stand, and every experiment was
run then reverted, so the artifacts in the repo are unchanged. The
findings are ordered by how much they change a headline claim. Items
marked **[better]** were tested and IMPROVE the result when fixed.

### 21.1 Label validity — the deepest problems

**F1. 45% of positives are `__version__` incrementing. [better]**
1,760 of the 1,769 `is_version_string` rows are `ATTRIBUTE_CHANGED_VALUE`
on `pkg.__version__` / `VERSION`. Downstream code references the
constant, so usage > 0 and it is labelled positive — but a version
constant changing value breaks nobody. Version strings: 21.65% positive;
everything else: 2.21%. The model's top feature (22.6% gain) is
detecting this artifact. Re-run with those rows dropped: 21,499 rows,
476 positives, base rate 2.2%, PR-AUC median 0.177, **lift vs popularity
1.78× → 2.88×**, still 6/6 on both gates. Popularity was propped up by
version strings more than the model was.
*Fix:* drop `is_version_string == 1` rows in `labels.py` (or `build.py`)
before anything else; delete the feature. One filter line.

**F2. "Impact" means "one of 1,500 packages imports it."**
`user_count` among the 859 positives: 55% have exactly 1 user, 68% have
≤ 2, only 9 rows (1%) exceed 50. The median "high-impact" change has one
user. Binarising at `user_count > 0` discards the ordering signal that
LambdaRank natively consumes.
*Fix:* graded relevance — pass `np.log1p(user_count)` (clipped to an
integer grade 0–4) as the lambdarank label instead of 0/1. Keep the
binary label for PR-AUC. This is the single biggest modelling change
available and it is ~10 lines in `train.py`.

**F3. The usage join is sparse and its failure rate is unmeasured.**
Only 430 of 12,489 distinct changed symbols (3.4%) appear in the usage
index at all. Some of that is genuine (internal symbols), but
`ablate.py`'s own docstring warns the strict join fails on deep
definition paths. `label_alias` finds 62% more positives (1,395 vs 859)
and has never been evaluated.
*Fix:* run `baselines.py`, `stability.py`, `train.py` with
`--label label_alias`; compare lift over the SAME-label baseline (PR-AUC
is not comparable across labels — floors differ). Ship whichever wins.

**F4. Usage is a Sept-2026 snapshot applied to changes up to 14 years old.**
17.3% of rows are > 2 years old; 4.8% > 5 years. Old rows have a HIGHER
positive rate (5.9% at 5+ y vs 3.2% at < 1 y) — survivorship, not signal.
*Fix (cheap):* restrict the ingest window to versions released within
24 months of the usage scan, and state the snapshot date in the report's
limitations. *Fix (proper):* scan usage at multiple historical dates —
out of scope this semester.

### 21.2 Feature leakage and the model

**F5. `package_churn` leaks the future. [better]**
`build.py` computes it as `groupby("package").transform("size")` over
the WHOLE frame before `temporal_split`. Single constant per package; a
median 33% of it comes from releases after the cut. Not knowable at
serving time. #2 feature at 17.1% gain. Re-run with a past-only count
(`rank(method="min") - 1` over `released_at` within package): PR-AUC
median 0.236 → 0.265, **lift 1.78× → 2.02×**, min lift across cuts
1.16× → 1.73×.
*Fix:* replace the transform with the past-only cumulative count in
`add_features`. `release_size` is fine (within one pair — contemporaneous).

**F6. `package_rank` is a package identifier.**
314 distinct values for 314 packages. With `package_churn` constant per
package, the model can memorise "package X has positives." Test-set
split: PR-AUC **0.392 on packages seen in train vs 0.197 on unseen**
(lift over floor 6.2× vs 5.0×). Mitigations already present: 70% of test
rows are from unseen packages, and within-upgrade precision@10 is
identical (0.192 vs 0.188) — the memorisation is entirely in
cross-package ordering.
*Fix:* report both regimes separately, always. For the product the
"seen" regime is arguably the relevant one (new release of a known
package); say so.

**F7. Model complexity is set by a constant, not the data.**
CV folds chose [3, 10, 54, 6] trees; median 8; clamped to
`MIN_TREES = 20`. This is why the ablation noise floor is 17% and why
"path shape only" scores 116% of the full model. Nothing about the
ensemble size is tuned.
*Fix:* tune `n_estimators`, `num_leaves`, `learning_rate`,
`min_child_samples` on the time-ordered validation slice; remove the
clamp; re-run ablation and expect the noise floor to drop.

**F8. "Trees exploit the U-shape a linear model can't" is asserted, not tested.**
*Fix:* add a logistic-regression baseline in `baselines.py` on the same
18 features. If it matches LightGBM, the claim goes; if not, it is now
evidence.

### 21.3 Statistical honesty

**F9. Finding 1 (deprecation) is underpowered, not "no signal."**
278 deprecated rows, 10 positives. 95% CI on 3.60% is [1.4%, 5.8%]; the
comparison 3.69% sits inside it. The data cannot distinguish "no effect"
from a ±2-point effect either way.
*Fix:* rewrite §19.1, the report, and the deck note as "no detectable
effect at n = 278; the test cannot resolve effects smaller than ~2
points." A larger deprecated sample (a wider version window) is the only
way to actually answer the book's question.

**F10. Finding 2's U-shape is mostly a version-string artifact.**
With version strings: 2.58 → 8.21 → 8.69 → 0.25%. Without:
2.30 → 3.92 → 2.76 → 0.03%. The middle peak ("3× more likely") largely
evaporates; the feature keeps real gain (6.3% → 5.4%). What survives
robustly is the collapse: modules with 21+ prior breaks are used
downstream 0.03% of the time.
*Fix:* restate the finding as "mass-refactor modules are effectively
never used downstream," not "moderate churn triples risk." Re-measure
after F1.

**F11. No confidence intervals anywhere.**
Every number in the deck and report is a point estimate on one cut or a
median of six.
*Fix:* bootstrap over version pairs (resample groups, not rows — see
F12) for PR-AUC and lift; report 95% intervals. ~30 lines in
`stability.py`. This would have caught F9 automatically.

**F12. Row counts overstate the sample; rows are not independent.**
23,268 rows collapse to 15,139 distinct (pair, symbol) changes. 8,129
rows are parameter-level entries sharing a symbol-level label (e.g.
`redis.client.Redis.__init__` with 49 "moved" parameters → 49 rows, one
label). 859 positive rows = 814 distinct positive changes. The 20,000-row
target is met on rows, not on independent observations.
*Fix:* state "15,139 distinct symbol-changes" alongside the row count;
bootstrap by group (F11); consider collapsing parameter rows into their
symbol row with a `n_param_changes` feature.

**F13. No untouched holdout.**
Every experiment — feature additions, ablations, stability, both
findings — evaluated on the same test split. The headline is optimistic
by an unknown amount.
*Fix:* freeze the last cut (`released_at >= 2026-08-04`) NOW as a
holdout; never evaluate on it until the final report; report it once.

### 21.4 Metric reliability and the product

**F14. The product-facing metrics stand on 96 of 2,036 upgrades (4.7%).**
precision@10 / nDCG@20 are (correctly) restricted to pairs with ≥ 1
positive and > 10 rows. Per cut that is 12–38 groups; the latest cut's
precision@10 is a mean over 12 upgrades. PR-AUC (the headline) is global
across packages and does not match the product's within-upgrade question.
*Fix:* report the group count next to every p@10 / nDCG; give intervals
(F11); lead with within-upgrade metrics only where n ≥ 30 groups.

**F15. For 95% of upgrades there is nothing to rank — and that is a product.**
1,530 of 2,036 pairs have no positive; most of the rest have ≤ 10
changes. The most valuable output for a typical upgrade is "no changed
symbol in this release is imported by anyone in the top-1,500," which
the pipeline can already assert.
*Fix (Varad):* the live API should return an explicit `all_clear` state
with the count of scanned downstream packages, before any ranking.

### 21.5 Data quality

**F16. `PARAMETER_MOVED` is 30% of the data at 0.32% positive, and 44% are echoes.**
6,906 rows. 43.6% share a (pair, symbol) with an `OBJECT_REMOVED` /
`PARAMETER_REMOVED` / `PARAMETER_ADDED_REQUIRED` row — griffe reporting
the shift of every parameter after the one that actually changed.
*Fix:* drop `PARAMETER_MOVED` rows that co-occur with a sibling
removal/addition on the same symbol; keep the rest with a flag.

**F17. 10.7% of rows involve a prerelease / dev / post version.**
e.g. `opentelemetry-semantic-conventions 0.59b0 → 0.60b0`. Legitimate for
beta-only packages; noise otherwise.
*Fix:* add `is_prerelease_pair` as a flag; evaluate with and without.

**F18. numpy is absent, and memory will not bring it back.**
Failed with griffe `CyclicAliasError`, not `BrokenProcessPool`.
*Fix:* reproduce on the numpy sdist alone; try `griffe` with
`resolve_aliases=False` / a newer griffe; if unfixable, file it upstream
and note numpy's absence explicitly in the report.

**F19. The 149 `BrokenProcessPool` packages are retryable.**
Tail-biased (median rank 426), so head coverage is intact, but 149 is
149. *Fix:* re-run those packages only, `--workers 1`, on a machine with
≥ 8 GB free, then re-run §20's pipeline.

### 21.6 Reproducibility

**F20. griffe is pinned with `>=`.**
The dataset is a function of griffe's diff semantics; a 2.3 release could
silently change what counts as a breaking change.
*Fix:* `griffe==2.2.0` in `requirements.txt`; pin lightgbm and pandas
likewise; commit a `pip freeze` as `requirements.lock`.

**F21. `metrics.py` has no tests.**
It is correct (read line by line: tie-breaking is seeded, the rankable
filter is right, nDCG's ideal is computed properly) — but nothing proves
it stays correct.
*Fix:* `tests/test_metrics.py` with a 6-row hand-computed case for each
of the three metrics, plus the constant-score case that bit before.

### 21.6b Gaps against the project book (Day 15, after re-reading it)

Compared the built system against the Project Book, ignoring dates.
Kill-gate conditions (>= 20,000 rows; beat semver on PR-AUC) are both
met, and the book's harder bar ("if you can't beat popularity you don't
have a project") is met 6/6. All eight ML-side components exist. The
project is AHEAD on ranking (Phase 4) and rolling evaluation (Phase 5).
What the book asked for that was never built:

**F22. `symbol_age_in_releases` — the third "clever" history feature.**
Book §4.6: "old, stable symbols have accumulated more users." Never
built. F4 found old rows have a HIGHER positive rate, which is exactly
this signal. Likely the strongest feature not yet in the model.
*Fix:* count releases (in the same chain) in which the symbol existed
before the current pair; past-only, from griffe, no leak.

**F23. Changelog text features — Phase 3 week 5, component 6.**
Book §4.6: TF-IDF over the changelog line for a change ("BREAKING" is
rare and informative), sentence embeddings later. None exist. The
largest single scheduled item that was skipped.
*Fix:* fetch CHANGELOG / GitHub release notes per version_to; match a
line to a symbol by name; TF-IDF (start) → all-MiniLM-L6-v2 (later).

**F24. `days_since_prev_release` — in the book's feature list.** Never
built. Cheap: diff of consecutive `released_at` within package.

**F25. The classifier → ranker comparison was never made.**
Book §4.7 sequences LGBMClassifier (weeks 3–6) THEN LGBMRanker (7–8)
and says compare them. `train.py --objective binary` exists; no
recorded comparison. *Fix:* run it once, record in NOTES; ship whichever
wins on nDCG@20.

**F26. "One story of a real release it got right" — 30 Oct DoD.**
Not produced. One hour with `features.csv` + the model's scores.

**F27. Provenance error, corrected Day 15:** the deck/report/study
guide called `prior_breaks_in_module` "added on a hunch / the feature
nobody bet on." The book lists it explicitly as
`previous_breakages_in_this_module`, beside `was_deprecated_before`.
Corrected everywhere to: the book named three history features and bet
on the wrong one.

**F28. Frontend is a placeholder skeleton (verified live, Day 15).**
`break-rank.vercel.app` loads over HTTPS with the BreakRank heading, but
the tab title is the Next.js default "Create Next App", the only metric
reads "187 changes analysed this week (placeholder)" — 187 being the
book's own example number — and it hangs on "checking API…". No
requirements.txt upload, no ranked list, no navigation. Below the book's
week-4 DoD ("plain HTML page showing the data"), well short of week 6
("real data from the database") and week 8 ("paste requirements.txt,
get an ordered list"). Owner: Varad.
*Fix:* set the title; replace the placeholder with a live count from
Neon (moves it to week-4 DoD in an hour); then the upload → rank flow.

**F29. API is on Render's free tier — the book's warning §3.6 #2, verbatim.**
`breakrank.onrender.com` is a real FastAPI service (`/health` → 200,
`/docs` serves Swagger) but measured cold start was **32.1 s**; the book
says 30–60 s and "do not use Render's free web service for the model."
Database is on Neon (correct — avoids warning #1). Owner: Varad.
*Fix:* move the API to Hugging Face Spaces (Docker SDK, port 7860) per
book Part 8, or at minimum add the 6-hourly GitHub Actions keep-alive
ping. Until then: warm `/health` two minutes before any demo.

**Direction summary after verifying the web half:** ML track at book
Phase 4–5; web track at Phase 1–2 with a working but sleeping API. The
book's second gate ("if the site is not live and working, stop adding
features") applies to the web half now.

**Previously not assessable from the ML side:** the book's Phase 3 / second gate
("if the site is not live, stop adding features") and week-6 DoD (a
public HTTPS link showing real data). Owner: Varad. This is the largest
open direction risk if it is not met.

### 21.7 What the audits cleared (do not "fix" these)

- Duplicates: a first check without `sub_target` in the key showed 7,221;
  with it, **1** redundant row. Not a problem.
- Sampling bias of the crashes runs the RIGHT way (tail, not head).
- `released_at` is complete (0 missing) and is the correct boundary date.
- `release_size` is contemporaneous — not a leak.
- The model generalises to unseen packages at 5× floor. It is not
  memorising everything.
- `metrics.py` is correct.

### 21.8 Order of work

The eight-week plan in the deck should be replaced by this, in order.
Each of the first four both fixes a defect AND raises the headline.

1. F13 — freeze the holdout. Ten minutes; must precede everything else.
2. F1 + F5 — drop version strings, leak-free churn. Re-run §20. [better ×2]
3. F2 — graded relevance. Re-run.
4. F9 + F10 — rewrite both findings honestly against the new numbers.
5. F11 + F12 — bootstrap by group; report intervals and the 15,139 count.
6. F3 — evaluate the alias label; pick one.
7. F7 + F8 — tune the ensemble; add the linear baseline.
8. F16 + F17 — collapse echoes, flag prereleases; re-run.
9. F19 + F18 — retry the 149; chase numpy.
10. F20 + F21 — pin deps; test the metrics.
11. F6 + F14 — reporting: two regimes, group counts everywhere.
12. F15 — hand the `all_clear` state to Varad's API. Then integrate.
13. F22 + F24 — symbol age, days-since-previous-release. Re-run. (Likely [better].)
14. F23 — changelog TF-IDF. The book's biggest skipped item.
15. F25 + F26 — the classifier comparison; one real-release story.
16. F28 + F29 (Varad) — real count on the frontend; move the API off Render or add keep-alive. Before any live demo.

Items 1–5 are about a week and turn the project from "good student
work" into something that survives a hostile reader.
