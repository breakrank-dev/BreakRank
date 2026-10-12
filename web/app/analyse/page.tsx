"use client";

import Link from "next/link";
import { useState } from "react";
import {
  BreakageList,
  HonestFooter,
  NoneImportedNote,
  RankingBadge,
  statusNote,
  type Breakage,
} from "../_components/breakages";
import { API, API_MISSING_MESSAGE, API_UNREACHABLE_MESSAGE, errorMessage } from "../_lib/api";

type ReleaseResult = {
  version: string;
  analysis_status: string;
  ranking: string;
  total: number;
  n_imported: number;
  breakages: Breakage[];
};

type PackageResult = {
  package: string;
  current: string;
  latest: string;
  releases: ReleaseResult[];
  older_releases_not_shown: number;
};

type AnalyzeResult = {
  model_version: string | null;
  packages: PackageResult[];
  up_to_date: { package: string; version: string; latest: string }[];
  not_tracked: string[];
  skipped: { line_number: number; line: string; reason: string }[];
};

const MAX_CHARS = 20_000; // the API rejects anything longer
const EXAMPLE = `cryptography==46.0.7
flask>=2.0   # not pinned, so BreakRank will skip it and say why
`;

export default function Analyse() {
  const [text, setText] = useState("");
  const [data, setData] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(API ? null : API_MISSING_MESSAGE);
  const [loading, setLoading] = useState(false);

  async function analyse() {
    if (!API) {
      setError(API_MISSING_MESSAGE);
      return;
    }
    if (!text.trim()) {
      setError("Paste a requirements.txt first.");
      return;
    }
    if (text.length > MAX_CHARS) {
      setError(`That is ${text.length.toLocaleString()} characters; the limit is ${MAX_CHARS.toLocaleString()}.`);
      return;
    }
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requirements: text }),
      });
      if (!res.ok) {
        setError(await errorMessage(res, `The API returned an error (${res.status}).`));
        return;
      }
      setData((await res.json()) as AnalyzeResult);
    } catch {
      setError(API_UNREACHABLE_MESSAGE);
    } finally {
      setLoading(false);
    }
  }

  const pinnedCount = data
    ? data.packages.length + data.up_to_date.length + data.not_tracked.length
    : 0;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-4xl px-4 py-14 sm:px-6">
        <header className="mb-10">
          <Link href="/" className="text-sm text-slate-500 hover:text-slate-300">
            ← Look up one release
          </Link>
          <h1 className="mt-3 text-4xl font-bold tracking-tight">Check a requirements.txt</h1>
          <p className="mt-2 text-slate-400">
            For each package pinned with <code className="text-slate-300">==</code>, see what changed
            between your version and the newest one we hold, ranked within each release.
          </p>
        </header>

        <section className="mb-8 rounded-lg border border-slate-800 bg-slate-900/40 p-5">
          <label htmlFor="requirements" className="sr-only">
            requirements.txt
          </label>
          <textarea
            id="requirements"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            spellCheck={false}
            placeholder={"cryptography==46.0.7\nrequests==2.32.3"}
            className="w-full rounded-md border border-slate-700 bg-slate-950 p-3 font-mono text-sm
                       outline-none focus:border-slate-500"
          />
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <button
                onClick={analyse}
                disabled={loading}
                className="rounded-md bg-slate-100 px-5 py-2 text-sm font-medium text-slate-900
                           hover:bg-white disabled:opacity-50"
              >
                {loading ? "Analysing…" : "Analyse"}
              </button>
              <button
                onClick={() => setText(EXAMPLE)}
                className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300
                           hover:border-slate-500 hover:text-slate-100"
              >
                Use an example
              </button>
            </div>
            <span className={`text-xs ${text.length > MAX_CHARS ? "text-amber-300" : "text-slate-600"}`}>
              {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
          </div>
        </section>

        {loading && (
          <p className="text-sm text-slate-400">
            Analysing… the API sleeps on the free tier, so the first request can take up to a minute.
          </p>
        )}

        {error && (
          <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 p-5">
            <p className="text-sm text-amber-200">{error}</p>
          </div>
        )}

        {data && (
          <section className="space-y-10">
            <p className="text-sm text-slate-400">
              {pinnedCount} pinned {pinnedCount === 1 ? "package" : "packages"}:{" "}
              <span className="text-slate-200">{data.packages.length} with newer releases</span>,{" "}
              {data.up_to_date.length} up to date, {data.not_tracked.length} not tracked.
              {data.skipped.length > 0 &&
                ` ${data.skipped.length} ${data.skipped.length === 1 ? "line" : "lines"} skipped.`}
            </p>

            {data.packages.map((p) => (
              <article key={p.package} className="rounded-lg border border-slate-800 p-5">
                <h2 className="text-xl font-semibold">
                  {p.package}{" "}
                  <span className="font-mono text-base text-slate-400">
                    {p.current} → {p.latest}
                  </span>
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  {p.releases.length} newer {p.releases.length === 1 ? "release" : "releases"}, newest
                  first. Each list is ranked within its own release.
                  {p.older_releases_not_shown > 0 &&
                    ` ${p.older_releases_not_shown} older ${
                      p.older_releases_not_shown === 1 ? "release is" : "releases are"
                    } not shown.`}
                </p>

                <div className="mt-6 space-y-8">
                  {p.releases.map((r) => (
                    <section key={r.version}>
                      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                        <h3 className="font-mono text-slate-200">{r.version}</h3>
                        <div className="flex items-center gap-3 text-xs text-slate-500">
                          {r.total > 0 && (
                            <span>
                              top {r.breakages.length} of {r.total.toLocaleString()}
                            </span>
                          )}
                          {r.total > 0 && <RankingBadge ranking={r.ranking} />}
                        </div>
                      </div>

                      {r.total === 0 ? (
                        <p className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-300">
                          {statusNote(r.analysis_status)}
                        </p>
                      ) : (
                        <>
                          {r.n_imported === 0 && <NoneImportedNote total={r.total} />}
                          <BreakageList items={r.breakages} />
                        </>
                      )}
                    </section>
                  ))}
                </div>
              </article>
            ))}

            {data.up_to_date.length > 0 && (
              <section>
                <h2 className="mb-2 text-sm font-semibold text-slate-300">Already on the newest release we hold</h2>
                <p className="text-sm text-slate-400">
                  {data.up_to_date.map((u) => `${u.package} ${u.version}`).join(", ")}
                </p>
              </section>
            )}

            {data.not_tracked.length > 0 && (
              <section>
                <h2 className="mb-2 text-sm font-semibold text-slate-300">Not tracked by BreakRank</h2>
                <p className="text-sm text-slate-400">{data.not_tracked.join(", ")}</p>
              </section>
            )}

            {data.skipped.length > 0 && (
              <section>
                <h2 className="mb-2 text-sm font-semibold text-slate-300">Lines we could not use</h2>
                <ul className="space-y-1 text-sm">
                  {data.skipped.map((s) => (
                    <li key={s.line_number} className="text-slate-400">
                      <span className="font-mono text-slate-600">line {s.line_number}:</span>{" "}
                      <code className="break-all text-slate-300">{s.line}</code> — {s.reason}
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </section>
        )}

        <HonestFooter />
      </div>
    </main>
  );
}
