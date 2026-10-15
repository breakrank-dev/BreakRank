// Shared by the home page and the analyse page. The leading underscore on the
// folder tells Next.js this is not a page of its own.

export type Breakage = {
  symbol: string;
  kind: string;
  sub_target: string;
  score: number | null;
  user_count: number;
  inherited_by: number;
  explanation: string;
  is_private: boolean;
};

const KIND_LABELS: Record<string, string> = {
  OBJECT_REMOVED: "Removed",
  PARAMETER_REMOVED: "Parameter removed",
  PARAMETER_ADDED_REQUIRED: "Required parameter added",
  PARAMETER_CHANGED_REQUIRED: "Parameter now required",
  PARAMETER_CHANGED_DEFAULT: "Default changed",
  PARAMETER_CHANGED_KIND: "Parameter kind changed",
  PARAMETER_MOVED: "Parameters reordered",
  RETURN_CHANGED_TYPE: "Return type changed",
  ATTRIBUTE_CHANGED_TYPE: "Attribute type changed",
  ATTRIBUTE_CHANGED_VALUE: "Value changed",
  CLASS_REMOVED_BASE: "Base class removed",
  OBJECT_CHANGED_KIND: "Kind changed",
};

export const STATUS_NOTES: Record<string, string> = {
  analysed_clean: "We analysed this release and found no breaking changes.",
  analysis_failed: "We tried to analyse this release and could not.",
  no_source: "This release shipped no source code, so it could not be analysed.",
  no_baseline: "This is the oldest release we hold, so there is nothing to compare it against.",
  yanked: "This release was withdrawn by its maintainers.",
  unknown: "We hold this release, but we can't confirm it was compared with the one before it.",
};

export function statusNote(status: string): string {
  return STATUS_NOTES[status] ?? "No changes to show for this release.";
}

function kindLabel(kind: string) {
  return KIND_LABELS[kind] ?? kind.replaceAll("_", " ").toLowerCase();
}

export function BreakageList({ items }: { items: Breakage[] }) {
  return (
    <ol className="space-y-3">
      {items.map((b, i) => (
        <li
          key={`${b.symbol}-${b.kind}-${b.sub_target}`}
          className="flex gap-4 rounded-lg border border-slate-800 bg-slate-900/40 p-4
                     transition-colors hover:border-slate-700"
        >
          <div className="w-8 shrink-0 pt-0.5 text-right font-mono text-sm text-slate-600">
            {i + 1}
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <code className="break-all text-sm text-slate-100">{b.symbol}</code>
              <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
                {kindLabel(b.kind)}
                {b.sub_target && <span className="text-slate-500"> · {b.sub_target}</span>}
              </span>
            </div>
            <p className="mt-1.5 break-words text-sm text-slate-400">{b.explanation}</p>
          </div>

          {b.user_count > 0 && (
            <div className="w-20 shrink-0 text-right">
              <div className="text-lg font-semibold text-slate-100">
                {b.user_count.toLocaleString()}
              </div>
              <div className="text-[11px] leading-tight text-slate-500">
                {b.user_count === 1 ? "package" : "packages"}
                <br />
                import this
              </div>
            </div>
          )}
        </li>
      ))}
    </ol>
  );
}

// Shown when a release has changes but none of them is imported by any scanned
// package. Deliberately not called "all clear": the scanner reads import
// statements, so it misses methods called on objects.
export function NoneImportedNote({ total }: { total: number }) {
  return (
    <p className="mb-4 rounded-lg border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-300">
      None of these {total.toLocaleString()} {total === 1 ? "change is" : "changes are"} imported by
      any of the ~1,500 packages we scanned. Our scanner reads import statements, so it can miss
      methods called on objects — check method changes yourself.
    </p>
  );
}

export function RankingBadge({ ranking }: { ranking: string }) {
  if (ranking !== "model") {
    return (
      <span className="rounded-full bg-slate-800 px-2.5 py-1 text-xs text-slate-400">
        ranked by usage (no score)
      </span>
    );
  }
  return (
    <span className="inline-flex flex-col items-end gap-1 text-right">
      <span className="rounded-full bg-emerald-950/50 px-2.5 py-1 text-xs text-emerald-300">
        shortest import path first
      </span>
      <span className="text-[11px] text-slate-500">
        Changes at one depth are in no particular order.
      </span>
    </span>
  );
}

export function HonestFooter() {
  return (
    <div className="mt-10 space-y-3 border-t border-slate-800 pt-4 text-xs leading-relaxed text-slate-600">
      <p>
        <span className="text-slate-400">How is this ordered?</span> BreakRank lists a
        release&apos;s changes by how short their public import path is, shortest first. On
        releases published before 28 July 2026 that rule beats a popularity baseline at every
        evaluation date, by 2.5x to 4.8x; a learned ranker did not beat the rule, so the rule
        ships.
      </p>
      
      <p>
        The order is a ranking within each release, not a probability that your code breaks. Usage
        counts come from a static scan of ~1,500 downstream packages and count how many of them
        import each symbol. A count of zero means the symbol was not observed in that scan, not that
        nobody uses it — the scanner reads import statements, so it sees module-level symbols far
        better than methods called on objects. Private symbols are excluded.
      </p>
    </div>
  );
}
