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
