"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL;

export default function Home() {
  const [status, setStatus] = useState("checking API...");

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then((d) => setStatus(d.ok ? `API connected (${d.model_version})` : "API error"))
      .catch(() => setStatus("API unreachable"));
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-5xl font-bold mb-4">BreakRank</h1>
        <p className="text-slate-400 mb-8">
          Ranking Python dependency breaking changes by real-world impact
        </p>
        <div className="border border-slate-800 rounded-lg p-8 inline-block">
          <div className="text-4xl font-bold">187</div>
          <div className="text-sm text-slate-500 mt-1">
            changes analysed this week (placeholder)
          </div>
        </div>
        <p className="text-xs text-slate-600 mt-8">{status}</p>
      </div>
    </main>
  );
}