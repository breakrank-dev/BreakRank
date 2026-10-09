"""
F26: does scripts/stories.py pick its two stories by its stated rules,
from releases the model never trained on, and from nothing sealed?

    python scripts/test_stories.py

No network and no real data; about a minute. Hand-built frames for the
rules, then the real script on test_relevance.py's fixture in a temp
directory. Nothing in your data/ or artifacts/ is read or touched.

Five cases:

  1. The tally: hits@10 counts used changes in the model's top 10, and
     "expected" is what a random order would put there. Releases with no
     used change, or fewer than --min-changes changes, are not stories.
  2. The success story: the most hits above expected; a tie goes to the
     release whose found changes more packages use.
  3. The failure story: the most-used change the model put in the
     bottom half of its list, not merely the lowest-ranked one; with
     none in a bottom half, the one ranked lowest relative to its list.
  4. Where stories come from: the test half only, never the train half
     the model learned on; private symbols out unless asked, as on the
     site; and a features.csv carrying a holdout row is refused.
  5. End to end on the fixture: both stories printed, with their links,
     each from the test half.
"""

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.holdout import HOLDOUT_START  # noqa: E402

PASS, FAIL = "  ok  ", "  FAIL"
failures = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{PASS if ok else FAIL}  {name}")
    if not ok:
        print(f"          got  {got!r}")
        print(f"          want {want!r}")
        failures.append(name)


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load("stories")


def upgrade(name: str, n: int, used_at: dict[int, int],
            split: str = "test", private: tuple = ()) -> pd.DataFrame:
    """One release of n changes, scored so the model's rank is the row
    number (1 = top). used_at maps a rank to how many packages use the
    change there; every other change is unused."""
    ranks = np.arange(1, n + 1)
    return pd.DataFrame({
        "package": name, "version_from": "1.0", "version_to": "2.0",
        "symbol": [f"{name}.f{r:03d}" for r in ranks],
        "sub_target": "", "kind": "OBJECT_REMOVED",
        "score": -ranks.astype(float),
        "label_alias": [int(r in used_at) for r in ranks],
        "alias_user_count": [used_at.get(r, 0) for r in ranks],
        "split": split,
        "is_private": [r in private for r in ranks]})


def case_tally() -> None:
    print("\n1. THE TALLY")
    df = pd.concat([
        upgrade("a", 20, {2: 5, 7: 3}),             # 2 used, both in top 10
        upgrade("b", 20, {1: 1, 4: 1, 12: 1, 15: 50}),
        upgrade("c", 25, {25: 90}),                  # its one used change last
        upgrade("small", 10, {1: 9}),                # under 20 changes
        upgrade("quiet", 30, {}),                    # nothing used
    ], ignore_index=True)
    t = S.tally(df, 20).set_index("package")
    check("only releases with a used change and 20+ changes are candidates",
          sorted(t.index), ["a", "b", "c"])
    check("hits@10 counts used changes in the top 10",
          t["hits10"].to_dict(), {"a": 2, "b": 2, "c": 0})
    check("expected is used x 10 / changes: 1.0, 2.0, 0.4",
          {k: round(v, 3) for k, v in t["expected10"].items()},
          {"a": 1.0, "b": 2.0, "c": 0.4})
    tiny = S.tally(upgrade("tiny", 8, {3: 1, 8: 1}), 5).iloc[0]
    check("under 10 changes every change is in the top 10, so a random "
          "order expects all its used ones",
          (int(tiny["hits10"]), round(float(tiny["expected10"]), 3)),
          (2, 2.0))


def case_success() -> None:
    print("\n2. THE SUCCESS STORY")
    df = pd.concat([upgrade("a", 20, {2: 5, 7: 3}),
                    upgrade("b", 20, {1: 1, 4: 1, 12: 1, 15: 50}),
                    upgrade("c", 25, {25: 90})], ignore_index=True)
    check("the most hits above expected wins",
          S.pick_success(S.tally(df, 20))["package"], "a")
    tie = pd.concat([upgrade("a", 20, {2: 5, 7: 3}),
                     upgrade("z", 20, {1: 40, 3: 2})], ignore_index=True)
    check("on a tie, the release whose found changes more packages use",
          S.pick_success(S.tally(tie, 20))["package"], "z")


def case_failure() -> None:
    print("\n3. THE FAILURE STORY")
    df = pd.concat([upgrade("a", 20, {2: 5, 7: 3}),
                    upgrade("b", 20, {1: 1, 4: 1, 12: 1, 15: 50}),
                    upgrade("c", 25, {25: 90})], ignore_index=True)
    f = S.pick_failure(df, S.tally(df, 20))
    check("the most-used change in a bottom half: c's, used by 90, last of 25",
          (f["package"], int(f["alias_user_count"]), int(f["rank"])),
          ("c", 90, 25))
    both = upgrade("f", 20, {12: 60, 20: 2})
    f = S.pick_failure(both, S.tally(both, 20))
    check("in the bottom half the most used wins, not the lowest ranked",
          (int(f["alias_user_count"]), int(f["rank"])), (60, 12))
    top = pd.concat([upgrade("d", 20, {1: 500, 3: 2}),
                     upgrade("e", 20, {2: 1, 9: 7})], ignore_index=True)
    f = S.pick_failure(top, S.tally(top, 20))
    check("with none in a bottom half, the lowest relative to its list, "
          "not the most used", (f["package"], int(f["rank"])), ("e", 9))


def case_rows() -> None:
    print("\n4. WHERE STORIES COME FROM")
    df = pd.concat([upgrade("trained", 20, {1: 99}, split="train"),
                    upgrade("unseen", 20, {3: 4}, private=(1, 2))],
                   ignore_index=True)
    rows = S.story_rows(df, include_private=False)
    check("a release in the train half is never a story",
          sorted(rows["package"].unique()), ["unseen"])
    check("private symbols are out by default, as on the site",
          bool(rows["is_private"].any()), False)
    check("and in with --include-private",
          len(S.story_rows(df, include_private=True)), 20)


def build(tmp: pathlib.Path) -> bool:
    (tmp / "data").mkdir()
    load("test_relevance").labelled().to_csv(tmp / "data" / "labelled.csv",
                                             index=False)
    for script in ("ml/features/build.py", "ml/model/train.py"):
        p = subprocess.run([sys.executable, str(ROOT / script)], cwd=tmp,
                           capture_output=True, text=True, timeout=900)
        if p.returncode:
            print((p.stdout + p.stderr)[-1500:])
            return False
    return True


def run(tmp: pathlib.Path, *args: str):
    p = subprocess.run([sys.executable, str(ROOT / "scripts/stories.py"),
                        *args], cwd=tmp, capture_output=True, text=True,
                       timeout=600)
    return p.returncode, p.stdout + p.stderr


def case_end_to_end(tmp: pathlib.Path) -> None:
    print("\n5. END TO END")
    ok = build(tmp)
    check("the fixture builds and the model trains", ok, True)
    if not ok:
        return
    code, out = run(tmp)
    check("stories.py runs and tells both stories, with links",
          (code, "THE SUCCESS STORY" in out, "THE FAILURE STORY" in out,
           out.count("on the site:")), (0, True, True, 2))
    feats = pd.read_csv(tmp / "data" / "features.csv",
                        dtype={"version_from": str, "version_to": str})
    test = feats[feats["split"] == "test"]
    told = []
    for line in out.splitlines():
        if line.startswith(("THE SUCCESS STORY: ", "THE FAILURE STORY: ")):
            name, rest = line.split(": ", 1)[1].split(" ", 1)
            v_from, v_to = rest.split(" -> ")
            told.append(((test["package"] == name)
                         & (test["version_from"] == v_from)
                         & (test["version_to"] == v_to)).any())
    check("both are releases from the test half", told, [True, True])

    held = feats.head(1).assign(
        released_at=str((HOLDOUT_START + pd.Timedelta(days=3)).date()))
    pd.concat([feats, held]).to_csv(tmp / "data" / "features.csv",
                                    index=False)
    code, out = run(tmp)
    check("a features.csv carrying a holdout row is refused",
          (code != 0, "HOLDOUT ROWS" in out), (True, True))


def main() -> None:
    case_tally()
    case_success()
    case_failure()
    case_rows()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="breakrank-stories-"))
    try:
        case_end_to_end(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed. Both stories are picked by rule, from")
    print("releases the model never trained on, and nothing sealed is read.")


if __name__ == "__main__":
    main()
