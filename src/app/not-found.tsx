import Link from 'next/link'

export default function NotFound() {
  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center">
      <div className="max-w-2xl mx-auto px-6 text-center">
        <h1 className="text-6xl font-bold text-emerald-400 mb-4">404</h1>
        
        <h2 className="text-2xl font-semibold mb-6">
          Looking for the 23.1% Raw Inference Logs?
        </h2>
        
        <p className="text-gray-400 text-lg mb-8 leading-relaxed">
          Detailed inference logs, DSL architecture specs, per-task failure analysis, 
          and rule distribution data are reserved for our research partners and{' '}
          <a 
            href="https://github.com/sponsors/Ag3497120" 
            className="text-emerald-400 hover:text-emerald-300 underline"
            target="_blank"
          >
            GitHub Sponsors
          </a>.
        </p>

        <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 mb-8 text-left">
          <h3 className="text-emerald-400 font-mono text-sm mb-3">$ cat score.json</h3>
          <pre className="text-gray-300 text-sm font-mono overflow-x-auto">
{`{
  "benchmark": "ARC-AGI-2",
  "score": "231/1000 (23.1%)",
  "method": "Pure symbolic — zero neural networks",
  "avg_solve_time": "0.66s/task",
  "sponsor_data": "https://github.com/sponsors/Ag3497120"
}`}
          </pre>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 text-left">
          <div className="bg-gray-900 border border-gray-800 rounded p-4">
            <div className="text-emerald-400 font-bold">$5/mo</div>
            <div className="text-gray-400 text-sm mt-1">Score updates + failure categories</div>
          </div>
          <div className="bg-gray-900 border border-emerald-800 rounded p-4">
            <div className="text-emerald-400 font-bold">$20/mo</div>
            <div className="text-gray-400 text-sm mt-1">Full inference logs + rule distribution</div>
          </div>
          <div className="bg-gray-900 border border-emerald-600 rounded p-4">
            <div className="text-emerald-400 font-bold">$50/mo</div>
            <div className="text-gray-400 text-sm mt-1">Raw eval data + DSL drafts + architecture</div>
          </div>
        </div>

        <div className="flex gap-4 justify-center">
          <a
            href="https://github.com/sponsors/Ag3497120"
            className="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-3 rounded-lg font-semibold transition"
            target="_blank"
          >
            Become a Sponsor
          </a>
          <Link
            href="/"
            className="border border-gray-600 hover:border-gray-400 text-gray-300 px-6 py-3 rounded-lg font-semibold transition"
          >
            Back to Verantyx
          </Link>
        </div>

        <p className="text-gray-600 text-xs mt-8">
          Public data available at{' '}
          <a href="/score.json" className="text-gray-500 hover:text-gray-400">/score.json</a>
          {' '}·{' '}
          <a href="https://github.com/Ag3497120/verantyx-v6" className="text-gray-500 hover:text-gray-400" target="_blank">
            GitHub
          </a>
        </p>
      </div>
    </div>
  )
}
