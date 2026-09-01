"""
A readable view of data/day1_changes.csv.

Spreadsheets are the wrong tool for this file — the `explanation` column is
full of long file paths and it pushes everything off screen. This prints the
same data in a way you can actually read.

    python scripts/explore.py              overview: what is in the file
    python scripts/explore.py removed      just the OBJECT_REMOVED rows
    python scripts/explore.py click        everything for one package
    python scripts/explore.py row 13       one row, fully explained

Nothing here changes any data. It only looks.
"""

import pathlib
import sys

import pandas as pd

CSV = pathlib.Path("data/day1_changes.csv")

# Plain-English meaning of each column, printed by the `row` view.
COLUMNS = {
    "package": "Which library this change is in",
    "version_from": "The version being upgraded FROM",
    "version_to": "The version being upgraded TO",
    "symbol": "The exact thing that changed, as a dotted path",
    "kind": "What sort of change griffe detected",
    "is_private": "Does any part of the name start with _ (private by convention)",
    "module_depth": "How many dots in the path — deeper usually means more obscure",
    "is_top_level": "Can you reach it directly as package.thing",
    "name_length": "Characters in the final part of the name",
    "released_at": "When version_to was published",
    "explanation": "griffe's own description of the change",
}

KIND_MEANING = {
    "OBJECT_REMOVED": "A function or class is GONE. The most serious kind.",
    "PARAMETER_REMOVED": "An argument disappeared. Calls passing it now fail.",
    "PARAMETER_ADDED_REQUIRED": "A new COMPULSORY argument. Existing calls fail.",
    "PARAMETER_CHANGED_REQUIRED": "An optional argument became compulsory.",
    "PARAMETER_CHANGED_DEFAULT": "A default value changed. Silent behaviour change.",
    "PARAMETER_CHANGED_KIND": "Positional became keyword-only, or vice versa.",
    "PARAMETER_MOVED": "Arguments swapped order. Positional calls break.",
    "ATTRIBUTE_CHANGED_VALUE": "A constant or type hint was rewritten. Usually harmless.",
    "ATTRIBUTE_CHANGED_TYPE": "A class attribute's declared type changed.",
    "RETURN_CHANGED_TYPE": "What the function gives back changed type.",
    "OBJECT_CHANGED_KIND": "A function became a class, or similar.",
}


def load() -> pd.DataFrame:
    if not CSV.exists():
        sys.exit(f"{CSV} not found. Run:  python scripts/day1_demo.py")
    return pd.read_csv(CSV)


def overview(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {len(df)} rows in {CSV}")
    print(f"{'=' * 66}")

    print("\nONE ROW = one thing that changed between two versions of one library.")
    print("So 'click 8.4.2 -> 8.5.0 removed click.utils.LazyFile' is one row.\n")

    print("-" * 66)
    print("BY PACKAGE")
    print("-" * 66)
    for pkg, n in df["package"].value_counts().items():
        pairs = df[df.package == pkg].groupby(["version_from", "version_to"]).ngroups
        print(f"  {pkg:<14} {n:>4} changes across {pairs} version pairs")

    print("\n" + "-" * 66)
    print("BY KIND OF CHANGE  (this is the important table)")
    print("-" * 66)
    for kind, n in df["kind"].value_counts().items():
        share = n / len(df)
        bar = "#" * max(1, round(share * 34))
        print(f"\n  {kind}")
        print(f"  {n:>4} rows ({share:>4.0%})  {bar}")
        print(f"        {KIND_MEANING.get(kind, '')}")

    serious = df[df.kind.isin(["OBJECT_REMOVED", "PARAMETER_REMOVED",
                               "PARAMETER_ADDED_REQUIRED"])]
    print("\n" + "-" * 66)
    print(f"Rows that could plausibly break somebody: {len(serious)} out of {len(df)}")
    print(f"Everything else — {len(df) - len(serious)} rows — is mostly noise.")
    print("Separating those two groups automatically IS this project.")
    print("-" * 66)
    print("\nNext:  python scripts/explore.py removed")


def removed(df: pd.DataFrame) -> None:
    sub = df[df.kind == "OBJECT_REMOVED"]
    if sub.empty:
        print("No OBJECT_REMOVED rows in this file.")
        return

    print(f"\n{'=' * 66}")
    print(f"  {len(sub)} things that were REMOVED")
    print(f"{'=' * 66}")
    print("\nFor each one ask: would anyone outside that library have called this?")
    print("That instinct is what you are going to teach a model.\n")

    for pkg, group in sub.groupby("package"):
        print(f"\n{pkg}")
        print("-" * 66)
        for _, r in group.iterrows():
            flag = "private" if r.is_private else "public "
            print(f"  [{flag}] {r.symbol}")
            print(f"            {r.version_from} -> {r.version_to}, "
                  f"depth {r.module_depth}")


def package(df: pd.DataFrame, name: str) -> None:
    sub = df[df.package.str.lower() == name.lower()]
    if sub.empty:
        opts = ", ".join(sorted(df["package"].unique()))
        sys.exit(f"No rows for '{name}'. Available: {opts}")

    print(f"\n{'=' * 66}")
    print(f"  {name} — {len(sub)} changes")
    print(f"{'=' * 66}")

    for (vf, vt), group in sub.groupby(["version_from", "version_to"]):
        print(f"\n  {vf} -> {vt}   ({len(group)} changes)")
        print("  " + "-" * 62)
        for _, r in group.iterrows():
            mark = "!" if r.kind == "OBJECT_REMOVED" else " "
            print(f"  {mark} {r.kind:<26} {r.symbol}")


def one_row(df: pd.DataFrame, idx: int) -> None:
    if idx not in df.index:
        sys.exit(f"No row {idx}. Valid range: 0 to {len(df) - 1}")
    r = df.loc[idx]

    print(f"\n{'=' * 66}")
    print(f"  ROW {idx}, column by column")
    print(f"{'=' * 66}\n")

    for col, meaning in COLUMNS.items():
        value = str(r[col])
        if len(value) > 60:
            value = value[:57] + "..."
        print(f"  {col:<14} {value}")
        print(f"  {'':<14} \033[2m{meaning}\033[0m\n")

    print("-" * 66)
    print("In one sentence:")
    print(f"  In {r.package}, upgrading from {r.version_from} to {r.version_to}")
    print(f"  changed {r.symbol}")
    print(f"  ({KIND_MEANING.get(r.kind, r.kind)})")
    print("-" * 66)


if __name__ == "__main__":
    data = load()
    arg = sys.argv[1] if len(sys.argv) > 1 else "overview"

    if arg == "overview":
        overview(data)
    elif arg == "removed":
        removed(data)
    elif arg == "row":
        one_row(data, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        package(data, arg)
