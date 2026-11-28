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
} from "./_components/breakages";
import { API, API_MISSING_MESSAGE, API_UNREACHABLE_MESSAGE, errorMessage } from "./_lib/api";

type Result = {
  package: string;
  version: string;
  model_version: string | null;
  ranking: string;
  analysis_status: string;
  total: number;
  n_imported?: number;
  breakages: Breakage[];
};

// Only releases you have opened on the live site yourself. Preset buttons mean
// you never type a package name on stage.
const PRESETS = [
  { pkg: "cryptography", version: "47.0.0" },
  { pkg: "sglang", version: "0.5.10" },
  { pkg: "cryptography", version: "50.0.1" },
];

export default function Home() {
  const [pkg, setPkg] = useState(PRESETS[0].pkg);
  const [version, setVersion] = useState(PRESETS[0].version);
  const [data, setData] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(API ? null : API_MISSING_MESSAGE);
  const [loading, setLoading] = useState(false);

  async function analyse(p: string, v: string) {
    setPkg(p);
    setVersion(v);
    if (!API) {
      setError(API_MISSING_MESSAGE);
      return;
    }
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch(
        `${API}/packages/${encodeURIComponent(p)}/releases/${encodeURIComponent(v)}/breakages?limit=20`
      );
      if (res.status === 404) {
        setError(await errorMessage(res, `BreakRank has no record of ${p}==${v}.`));
        return;
      }
      if (!res.ok) {
        setError(await errorMessage(res, `The API returned an error (${res.status}).`));
        return;
      }
      setData((await res.json()) as Result);
    } catch {
      setError(API_UNREACHABLE_MESSAGE);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-4xl px-4 py-14 sm:px-6">
        <header className="mb-10 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold tracking-tight">BreakRank</h1>
            <p className="mt-2 text-slate-400">
              Ranking Python dependency breaking changes by real-world impact
            </p>
          </div>
          <Link
            href="/analyse"
            className="rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-300
                       hover:border-slate-500 hover:text-slate-100"
          >
            Check a requirements.txt →
          </Link>
        </header>

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
              aria-label="Package name"
              className="flex-1 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm
                         outline-none focus:border-slate-500"
            />
            <input
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              placeholder="version"
              aria-label="Version"
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

        {loading && (
          <p className="text-sm text-slate-400">
            Analysing… the API sleeps on the free tier, so the first request can take up to a minute.
          </p>
        )}

        {error && (
          <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 p-5">
            <p className="text-sm text-amber-200">{error}</p>
            <p className="mt-2 text-xs text-amber-200/60">
              We return this rather than an empty list, because &ldquo;we don&rsquo;t track this
              package&rdquo; and &ldquo;this package has no breaking changes&rdquo; are different answers.
            </p>
          </div>
        )}

        {data && (
          <section>
            <div className="mb-6 flex flex-wrap items-baseline justify-between gap-3 border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-semibold">
                  {data.package} <span className="font-mono text-slate-400">{data.version}</span>
                </h2>
                {data.total > 0 && (
                  <p className="mt-1 text-sm text-slate-400">
                    Showing the top{" "}
                    <span className="font-semibold text-slate-100">{data.breakages.length}</span> of{" "}
                    <span className="font-semibold text-slate-100">{data.total.toLocaleString()}</span>{" "}
                    detected changes
                  </p>
                )}
              </div>
              <div className="flex flex-col items-end gap-1 text-xs">
                <RankingBadge ranking={data.ranking} />
                {data.model_version && (
                  <span className="font-mono text-slate-600">{data.model_version}</span>
                )}
              </div>
            </div>

            {data.breakages.length === 0 && (
              <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
                <p className="text-slate-300">{statusNote(data.analysis_status)}</p>
                <p className="mt-2 font-mono text-xs text-slate-600">
                  analysis_status: {data.analysis_status}
                </p>
              </div>
            )}

            {data.total > 0 && data.n_imported === 0 && (
                          <NoneImportedNote total={data.total} />
            )}

            <BreakageList items={data.breakages} />
          </section>
        )}

        {!data && !error && !loading && (
          <p className="text-sm text-slate-500">
            Pick a package above to see its breaking changes, ranked.
          </p>
        )}

        <HonestFooter />
      </div>
    </main>
  );
}
