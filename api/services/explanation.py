"""Decision 4: the API composes the display sentence.

The pipeline stores structured facts in breakage.detail and the parameter or
base name in breakage.sub_target. Wording lives here so changing it is a
two-minute redeploy rather than re-running ingestion over thousands of releases.
"""

TEMPLATES = {
    "OBJECT_REMOVED": "{symbol} was removed.",
    "PARAMETER_REMOVED": "The {parameter} parameter was removed from {symbol}.",
    "PARAMETER_ADDED_REQUIRED": "{symbol} now requires a {parameter} argument.",
    "PARAMETER_CHANGED_REQUIRED": (
        "The {parameter} parameter of {symbol} is now required."
    ),
    "PARAMETER_CHANGED_DEFAULT": (
        "The default for {parameter} in {symbol} changed "
        "from {old_value} to {new_value}."
    ),
    "PARAMETER_CHANGED_KIND": "The {parameter} parameter of {symbol} changed kind.",
    "PARAMETER_MOVED": "Parameters of {symbol} were reordered.",
    "RETURN_CHANGED_TYPE": "{symbol} now returns a different type.",
    "ATTRIBUTE_CHANGED_TYPE": "The type of {symbol} changed.",
    "ATTRIBUTE_CHANGED_VALUE": "The value of {symbol} changed.",
    "CLASS_REMOVED_BASE": "{symbol} no longer inherits from {base}.",
    "OBJECT_CHANGED_KIND": "{symbol} changed from one kind of object to another.",
}

FALLBACK = "{symbol} changed in this release."


def render(
    kind: str,
    symbol_path: str,
    user_count: int,
    detail: dict,
    sub_target: str = "",
    inherited_by: int = 0,
) -> str:
    """Build the display sentence for one breakage."""
    template = TEMPLATES.get(kind, FALLBACK)

    # detail first so the explicit keys below win. sub_target is a real
    # column, so it is the source of truth over anything duplicated into
    # detail.
    fields = {
        **detail,
        "symbol": symbol_path,
        "parameter": sub_target,
        "base": sub_target,
    }

    try:
        sentence = template.format(**fields)
    except (KeyError, IndexError):
        # detail is missing a key the template wanted, or contains stray
        # braces. Degrade to the generic sentence rather than 500 — the user
        # came for a ranked list, not a stack trace.
        sentence = FALLBACK.format(symbol=symbol_path)

    if user_count > 0:
        plural = "package" if user_count == 1 else "packages"
        sentence += f" {user_count} {plural} in the ecosystem call it."

    if inherited_by > 0:
        sentence += f" It affects {inherited_by:,} inheriting classes."

    if deprecated := detail.get("was_deprecated_in"):
        sentence += f" It was deprecated in {deprecated}."

    if moved := detail.get("moved_to"):
        sentence += f" It appears to have moved to {moved}."

    return sentence