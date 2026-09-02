export default function Home() {
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
      </div>
    </main>
  );
}