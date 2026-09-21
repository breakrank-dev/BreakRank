"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL;

type Breakage = {
  symbol: string;
  kind: string;
  sub_target: string;
  score: number | null;
  user_count: number;
  inherited_by: number;
  explanation: string;
  is_private: boolean;
};

type Result = {
  package: string;
  version: string;
  model_version: string | null;
  ranking: string;
  analysis_status: string;
  total: number;
  breakages: Breakage[];
};

// Edit these to packages you have verified. Preset buttons mean you never
// type a package name on stage.
const PRESETS = [
  { pkg: "typing-extensions", version: "4.14.0" },
  { pkg: "pyyaml", version: "6.0" },
  { pkg: "sqlalchemy", version: "2.0.48" },
];

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

const STATUS_NOTES: Record<string, string> = {
  analysed_clean: "We analysed this release and found no breaking changes.",
  analysis_failed: "We tried to analyse this release and could not.",
  no_source: "This release shipped no source code, so it could not be analysed.",
  no_baseline: "This is the oldest release we hold, so there is nothing to compare it against.",
  yanked: "This release was withdrawn by its maintainers.",
};

function kindLabel(kind: string) {
  return KIND_LABELS[kind] ?? kind.replaceAll("_", " ").toLowerCase();
}

export default function Home() {
  const [pkg, setPkg] = useState(PRESETS[0].pkg);
  const [version, setVersion] = useState(PRESETS[0].version);
  const [data, setData] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function analyse(p: string, v: string) {
    setPkg(p);
    setVersion(v);
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch(
        `${API}/packages/${encodeURIComponent(p)}/releases/${encodeURIComponent(v)}/breakages?limit=20`
      );

      if (res.status === 404) {
        const body = await res.json().catch(() => null);
        setError(body?.detail?.message ?? `BreakRank has no record of ${p}==${v}.`);
        return;
      }
      if (!res.ok) {
        setError(`The API returned an error (${res.status}).`);
        return;
      }

      setData((await res.json()) as Result);
    } catch {
      setError(
        "Could not reach the API. It sleeps after 15 minutes idle on the free tier — give it a minute and try again."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-4xl px-6 py-14">
        {/* Header */}
        <header className="mb-10">
          <h1 className="text-4xl font-bold tracking-tight">BreakRank</h1>
          <p className="mt-2 text-slate-400">
            Ranking Python dependency breaking changes by real-world impact
          </p>
        </header>

        {/* Controls */}
        <section className="mb-8 rounded-lg border border-slate-800 bg-slate-900/40 p-5">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              analyse(pkg, version);
            }}
            className="flex flex-col gap-3 sm:flex-row"
          >
            <input
              value={pkg}
              onChange={(e) => setPkg(e.target.value)}
              placeholder="package"
              className="flex-1 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm
                         outline-none focus:border-slate-500"
            />
            <input
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              placeholder="version"
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm
                         outline-none focus:border-slate-500 sm:w-40"
            />
            <button
              type="submit"
              disabled={loading}
              className="rounded-md bg-slate-100 px-5 py-2 text-sm font-medium text-slate-900
                         hover:bg-white disabled:opacity-50"
            >
              {loading ? "Analysing…" : "Analyse"}
            </button>
          </form>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500">Try:</span>
            {PRESETS.map((p) => (
              <button
                key={`${p.pkg}-${p.version}`}
                onClick={() => analyse(p.pkg, p.version)}
                className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300
                           hover:border-slate-500 hover:text-slate-100"
              >
                {p.pkg} {p.version}
              </button>
            ))}
          </div>
        </section>

        {/* Loading */}
        {loading && (
          <p className="text-sm text-slate-400">
            Analysing… the API sleeps on the free tier, so the first request can take up to a minute.
          </p>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 p-5">
            <p className="text-sm text-amber-200">{error}</p>
            <p className="mt-2 text-xs text-amber-200/60">
              We return this rather than an empty list, because &ldquo;we don&rsquo;t track this
              package&rdquo; and &ldquo;this package has no breaking changes&rdquo; are different answers.
            </p>
          </div>
        )}

        {/* Results */}
        {data && (
          <section>
            {/* Summary bar */}
            <div className="mb-6 flex flex-wrap items-baseline justify-between gap-3 border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-semibold">
                  {data.package}{" "}
                  <span className="font-mono text-slate-400">{data.version}</span>
                </h2>
                {data.total > 0 && (
                  <p className="mt-1 text-sm text-slate-400">
                    Showing the top{" "}
                    <span className="font-semibold text-slate-100">
                      {data.breakages.length}
                    </span>{" "}
                    of{" "}
                    <span className="font-semibold text-slate-100">
                      {data.total.toLocaleString()}
                    </span>{" "}
                    detected changes
                  </p>
                )}
              </div>

              <div className="flex flex-col items-end gap-1 text-xs">
                <span
                  className={`rounded-full px-2.5 py-1 ${
                    data.ranking === "model"
                      ? "bg-emerald-950/50 text-emerald-300"
                      : "bg-slate-800 text-slate-400"
                  }`}
                >
                  {data.ranking === "model"
                    ? "ranked by model"
                    : "ranked by usage (no model score)"}
                </span>
                {data.model_version && (
                  <span className="font-mono text-slate-600">{data.model_version}</span>
                )}
              </div>
            </div>

            {/* Empty, with the reason */}
            {data.breakages.length === 0 && (
              <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
                <p className="text-slate-300">
                  {STATUS_NOTES[data.analysis_status] ??
                    "No changes to show for this release."}
                </p>
                <p className="mt-2 font-mono text-xs text-slate-600">
                  analysis_status: {data.analysis_status}
                </p>
              </div>
            )}

            {/* The ranked list */}
            <ol className="space-y-3">
              {data.breakages.map((b, i) => (
                <li
                  key={`${b.symbol}-${b.kind}-${b.sub_target}-${i}`}
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
                        {b.sub_target && (
                          <span className="text-slate-500"> · {b.sub_target}</span>
                        )}
                      </span>
                    </div>

                    <p className="mt-1.5 text-sm text-slate-400">{b.explanation}</p>
                  </div>

                  {b.user_count > 0 && (
                    <div className="w-20 shrink-0 text-right">
                      <div className="text-lg font-semibold text-slate-100">
                        {b.user_count.toLocaleString()}
                      </div>
                      <div className="text-[11px] leading-tight text-slate-500">
                        packages
                        <br />
                        use this
                      </div>
                    </div>
                  )}
                </li>
              ))}
            </ol>

            {/* Honest footer */}
            {data.breakages.length > 0 && (
              <p className="mt-8 border-t border-slate-800 pt-4 text-xs leading-relaxed text-slate-600">
                Usage counts come from a static scan of ~1,500 downstream packages and measure how
                many of them reference each symbol. A count of zero means the symbol was not observed
                in that scan, not that nobody uses it — our scanner reads import statements, so it
                sees module-level symbols far better than methods called on objects. Private symbols
                are excluded by default.
              </p>
            )}
          </section>
        )}

        {/* First load, nothing requested yet */}
        {!data && !error && !loading && (
          <p className="text-sm text-slate-500">
            Pick a package above to see its breaking changes, ranked.
          </p>
        )}
      </div>
    </main>
  );
}