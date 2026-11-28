"""
F26 (the 30 Oct definition of done) and the project book's §11.9: one
release the ranker got right, and one it got wrong, both from releases
the model never trained on.

    python scripts/stories.py
    python scripts/stories.py --min-changes 30 --include-private
    python scripts/stories.py --ranker model

Reads  data/features.csv, and with --ranker model artifacts/ranker.txt.
       Writes nothing.

WHICH RANKER. The one the site serves: from 11 Oct path length alone,
shortest public import path first, nothing fitted (NOTES §36.2,
ml/model/ship.py), scored as the loader scores it for the site.
--ranker model tells the stories of the model, kept as the fallback.

WHERE THE STORIES COME FROM. The test half of the development data:
upgrades released after the single split's cut and before the holdout
(28 Jul 2026). train.py fitted ranker.txt on the train half only, so the
model has never seen these releases; path length is fitted on nothing.
The site ranks them with the same scores, so a story here can be shown
live. The holdout is not read: picking a story from it would be looking
at its labels.

WHAT IS RANKED. The list the site shows by default: one release's
changes, private symbols left out (the API's include_private=false),
sorted by score and, inside a tie, by name, as the site lists a tie once
user_count stops ordering it (F45's fix). --include-private ranks them
all.

TIES ARE COUNTED BY THEIR AVERAGE. Path length gives every change at one
depth the same score, and the order inside a tie means nothing (§36.1),
so a count over a list takes its average over every order of the ties
(metrics._expected, F44), and a change's place is the middle of the
places its tie spans. The name order is only how the list is printed.

HOW THE TWO ARE CHOSEN, by rule rather than by eye:

  candidates  upgrades with at least one used change and at least
              --min-changes changes, so ranking has something to do
  hits@10     used changes in the ranker's top 10, averaged over the
              order of its ties
  expected    what a random order would put there: used x 10 / changes.
              Popularity and semver score every change of a release the
              same, so inside one release they ARE a random order.
  success     the most hits@10 above expected; a tie goes to the release
              whose found changes are used by the most packages
  failure     the used change with the most users whose place is in the
              bottom half of its list: the change people most needed to
              see, buried. A change tied with the whole of its list was
              not buried but not ranked, and does not count. With none in
              a bottom half, the used change placed lowest relative to its
              list.

Every candidate's result is printed too, so the two stories are read
against the rest and not instead of them. The book's advice holds: the
failure story matters more than the success story.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.features.build import BOOLEAN, CATEGORICAL, NUMERIC  # noqa: E402
from ml.holdout import GROUP, assert_no_holdout  # noqa: E402
from ml.model.metrics import _expected  # noqa: E402
from ml.model.ship import RANKERS, SHIPPED_RANKER, path_scores  # noqa: E402

DATA = pathlib.Path("data")
FEATURES = DATA / "features.csv"
RANKER = pathlib.Path("artifacts") / "ranker.txt"
LABEL = "label_alias"
COUNT = "alias_user_count"
API = "https://breakrank.onrender.com"
TOP = 10
NAME = {"path": "path length", "model": "the model"}
# Shown beside a buried change, so the failure story can say why.
WHY = ["kind", "public_depth", "module_depth", "name_length",
       "in_dunder_all", "has_export_path", "prior_breaks_in_module",
       "release_size", "package_rank"]


def score(df: pd.DataFrame, ranker: str = SHIPPED_RANKER) -> np.ndarray:
    """The scores the site serves for this ranker, made exactly as
    ml/db.py makes them: path length's from public_depth, a change with
    no depth the deepest of its release; the model's from ranker.txt."""
    if ranker == "path":
        return path_scores(df, within=GROUP).to_numpy(float)
    import lightgbm as lgb
    x = df[NUMERIC + BOOLEAN + CATEGORICAL].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype("category")
    raw = lgb.Booster(model_file=str(RANKER)).predict(x)
    return raw[:, 1] if getattr(raw, "ndim", 1) > 1 else raw


def story_rows(df: pd.DataFrame, include_private: bool) -> pd.DataFrame:
    """The rows a story may come from: the test half only, and without
    private symbols unless asked, as the site lists them."""
    test = df[df["split"] == "test"].copy()
    if not include_private:
        test = test[~test["is_private"].astype(bool)]
    if "sub_target" not in test:
        test["sub_target"] = ""
    return test


def ranked(group: pd.DataFrame) -> pd.DataFrame:
    """One release's changes as the site lists them: by score, a tie by
    name, so the order is the same each run. `rank` is the row's place in
    that list; `first` and `last` are the places its tie spans, and
    `place` their middle, where the change sits on average over every
    order of the tie. An untied change's three are its rank."""
    order = group.sort_values(["score", "symbol", "sub_target"],
                              ascending=[False, True, True],
                              na_position="last", kind="mergesort")
    rank = np.arange(1, len(order) + 1)
    s = pd.Series(order["score"].to_numpy(float)).fillna(-np.inf)
    run = (s != s.shift()).cumsum()
    span = pd.Series(rank).groupby(run.to_numpy())
    first = span.transform("min").to_numpy()
    last = span.transform("max").to_numpy()
    return order.assign(rank=rank, first=first, last=last,
                        place=(first + last) / 2)


def at(x) -> str:
    """A change's place, as a span when it is tied."""
    return (f"{int(x['first'])}-{int(x['last'])}" if x["last"] > x["first"]
            else f"{int(x['rank'])}")


def tally(df: pd.DataFrame, min_changes: int) -> pd.DataFrame:
    """One row per candidate upgrade: its size, used changes, hits@10 and
    what a random order would expect. hits@10, and the users of the
    changes it finds, are averaged over every order of the ties
    (_expected): a tie that straddles the 10th place counts by how many
    of its used changes the top 10 holds on average."""
    rows = []
    for key, g in df.groupby(GROUP, sort=False):
        used = int(g[LABEL].sum())
        n = len(g)
        if used == 0 or n < min_changes:
            continue
        s = g["score"].to_numpy(float)
        hit = g[LABEL].to_numpy(float)
        users = (g[COUNT].fillna(0).to_numpy(float) * hit if COUNT in g
                 else hit)
        rows.append({
            "package": key[0], "version_from": key[1], "version_to": key[2],
            "changes": n, "used": used,
            "hits10": float(_expected(s, hit, TOP).sum()),
            "expected10": used * min(TOP, n) / n,
            "found_users": float(_expected(s, users, TOP).sum()),
        })
    t = pd.DataFrame(rows)
    if len(t):
        # Rounded so two releases level in fact are level here too, and
        # the success rule's tie-break decides between them.
        t["above"] = (t["hits10"] - t["expected10"]).round(9)
        t["found_users"] = t["found_users"].round(9)
    return t


def pick_success(t: pd.DataFrame) -> pd.Series:
    best = t.sort_values(["above", "found_users", "package"],
                         ascending=[False, False, True], kind="mergesort")
    return best.iloc[0]


def pick_failure(df: pd.DataFrame, t: pd.DataFrame) -> pd.Series:
    """The buried used change: most users among those whose place is in
    the bottom half of their list; failing that, the lowest relative to
    its list. A tied change's place is the middle of its tie, so a change
    tied with its whole list sits at exactly the half: the ranker made no
    order there, which is not a burial, so it does not count as one."""
    rows = []
    for _, c in t.iterrows():
        g = df[(df["package"] == c["package"])
               & (df["version_from"] == c["version_from"])
               & (df["version_to"] == c["version_to"])]
        r = ranked(g)
        n = len(r)
        for _, x in r[r[LABEL] == 1].iterrows():
            rows.append({**x.to_dict(), "changes": n,
                         "depth": (x["place"] - 1) / max(n - 1, 1)})
    p = pd.DataFrame(rows)
    users = p[COUNT] if COUNT in p else pd.Series(0, index=p.index)
    p = p.assign(_users=users.fillna(0))
    low = p[(p["depth"] >= 0.5) & (p["first"] > 1)]
    if len(low):
        return low.sort_values(["_users", "depth", "symbol"],
                               ascending=[False, False, True],
                               kind="mergesort").iloc[0]
    return p.sort_values(["depth", "_users", "symbol"],
                         ascending=[False, False, True],
                         kind="mergesort").iloc[0]


def users_text(x) -> str:
    if x.get(LABEL, 0) != 1:
        return "-"
    n = x.get(COUNT)
    if n is None or pd.isna(n):
        return "used"
    return f"{int(n):,} pkg" + ("" if int(n) == 1 else "s")


def show_list(g: pd.DataFrame, mark: str | None = None) -> None:
    """The site's first 10, then every used change below them. A place
    printed as a span is a tie: the site lists it by name, and no order
    inside it is claimed."""
    r = ranked(g)
    print(f"    {'place':>7}  {'used by':>9}  {'kind':<24} symbol")
    for _, x in pd.concat([r[r["rank"] <= TOP],
                           r[(r["rank"] > TOP) & (r[LABEL] == 1)]]).iterrows():
        sub = "" if pd.isna(x.get("sub_target")) or not x.get("sub_target") \
            else f" ({x['sub_target']})"
        flag = "  <-" if mark is not None and x["symbol"] == mark else ""
        print(f"    {at(x):>7}  {users_text(x):>9}  "
              f"{str(x['kind'])[:24]:<24} {x['symbol']}{sub}{flag}")
    if (r["last"] > r["first"]).any():
        print("    (a span is a tie: listed by name, in no particular "
              "order)")


def links(package: str, version: str, symbol: str) -> None:
    print(f"    on the site:   {API}/packages/{package}/releases/"
          f"{version}/breakages")
    q = symbol.replace('"', "")
    print(f"    issues after:  https://github.com/search?type=issues&q="
          f"%22{q}%22  (look for ones opened after the release)")


def main() -> None:
    ap = argparse.ArgumentParser(description="F26: one release the ranker "
                                 "got right, one it got wrong.")
    ap.add_argument("--min-changes", type=int, default=20,
                    help="smallest release worth a story (default 20)")
    ap.add_argument("--include-private", action="store_true",
                    help="rank private symbols too; the site hides them")
    ap.add_argument("--ranker", choices=RANKERS, default=SHIPPED_RANKER,
                    help="path, path length alone, the ranker the site "
                         "serves (NOTES §36.2); or model, the fallback")
    args = ap.parse_args()
    who = NAME[args.ranker]

    need = (FEATURES,) if args.ranker == "path" else (FEATURES, RANKER)
    for p in need:
        if not p.exists():
            sys.exit(f"{p} not found. Run ml/features/build.py and "
                     "ml/model/train.py first.")
    df = pd.read_csv(FEATURES, dtype={"version_from": str,
                                      "version_to": str})
    assert_no_holdout(df, "stories.py")
    df["score"] = score(df, args.ranker)
    if pd.isna(df["score"]).any():
        sys.exit(f"{int(pd.isna(df['score']).sum())} changes have no score "
                 "from this ranker; every list needs one.")
    test = story_rows(df, args.include_private)

    t = tally(test, args.min_changes)
    if t.empty:
        sys.exit(f"No release in the test half has a used change and "
                 f"{args.min_changes}+ changes. Try --min-changes 10.")

    shown = "all changes" if args.include_private else \
        "public changes, as the site shows them"
    print(f"\nTHE TEST HALF: releases the model never trained on "
          f"({shown}), ranked by {who}")
    print(f"  {len(t)} releases have a used change and "
          f"{args.min_changes}+ changes.")
    beat = int((t["above"] > 0).sum())
    tied = int((t["above"] == 0).sum())
    print(f"  {who}'s top {TOP}, averaged over the order of its ties, holds "
          f"more used changes than\n  a random order would at {beat} of "
          f"them, as many at {tied}, fewer at {len(t) - beat - tied}")
    print(f"  used changes found in the top {TOP}: {t['hits10'].sum():.1f}"
          f" of {int(t['used'].sum())}; a random order would expect "
          f"{t['expected10'].sum():.1f}")
    print(f"  'used by' below: how many scanned downstream packages import "
          f"the change, by\n  {LABEL}'s count ({COUNT}); '-' means none "
          "does.")

    s = pick_success(t)
    g = test[(test["package"] == s["package"])
             & (test["version_from"] == s["version_from"])
             & (test["version_to"] == s["version_to"])]
    print(f"\nTHE SUCCESS STORY: {s['package']} {s['version_from']} -> "
          f"{s['version_to']}")
    when = str(g["released_at"].iloc[0])[:10] if "released_at" in g else ""
    print(f"  released {when}: {s['changes']} changes, {s['used']} of them "
          f"used downstream")
    print(f"  {who}'s top {TOP} holds {s['hits10']:.1f} of the used changes,"
          f" on average over the order of\n  its ties; a random order, which "
          f"is what popularity and semver are inside one\n  release, would "
          f"expect {s['expected10']:.1f}")
    show_list(g)
    found = ranked(g)
    found = found[(found["first"] <= TOP) & (found[LABEL] == 1)]
    if len(found):
        lead = (found.sort_values(COUNT, ascending=False).iloc[0]
                if COUNT in found else found.iloc[0])
        links(s["package"], s["version_to"], lead["symbol"])

    f = pick_failure(test, t)
    g = test[(test["package"] == f["package"])
             & (test["version_from"] == f["version_from"])
             & (test["version_to"] == f["version_to"])]
    print(f"\nTHE FAILURE STORY: {f['package']} {f['version_from']} -> "
          f"{f['version_to']}")
    print(f"  {f['symbol']} is used by {users_text(f)}, and {who} put it "
          f"at {at(f)} of {int(f['changes'])}")
    print("  its features, for the why:")
    print("    " + "   ".join(f"{c} {f[c]}" for c in WHY if c in f))
    show_list(g, mark=f["symbol"])
    links(f["package"], f["version_to"], f["symbol"])

    print("\nRead both against the summary above, and quote that summary "
          "beside them:\na story shows what the numbers mean, it is not "
          "more evidence than they are.")


if __name__ == "__main__":
    main()
