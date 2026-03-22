import Link from 'next/link'

export default function NotFound() {
  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center relative">
      {/* Vignette */}
      <div className="fixed inset-0 pointer-events-none" style={{ background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.4) 100%)' }} />

      <div className="max-w-2xl mx-auto px-6 text-center relative z-10">
        {/* 404 with cinematic blur flash */}
        <h1
          className="text-8xl font-black mb-4 tracking-tight"
          style={{
            background: 'linear-gradient(135deg, #0EA5E9, #7C3AED)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          404
        </h1>

        {/* Decorative line */}
        <div className="mx-auto mb-8 h-px w-24" style={{ background: 'linear-gradient(90deg, transparent, rgba(14,165,233,0.4), transparent)' }} />

        <h2 className="text-xl font-semibold mb-6 text-gray-300">
          Looking for the 23.1% Raw Inference Logs?
        </h2>

        <p className="text-gray-500 text-base mb-8 leading-relaxed">
          Detailed inference logs, DSL architecture specs, per-task failure analysis,
          and rule distribution data are reserved for our research partners and{' '}
          <a
            href="https://github.com/sponsors/Ag3497120"
            className="transition-colors duration-300"
            style={{ color: 'rgba(14, 165, 233, 0.7)' }}
            target="_blank"
          >
            GitHub Sponsors
          </a>.
        </p>

        <div
          className="rounded-2xl p-6 mb-8 text-left"
          style={{
            background: 'rgba(10, 10, 20, 0.6)',
            border: '1px solid rgba(14, 165, 233, 0.08)',
          }}
        >
          <h3 className="font-mono text-xs mb-3 tracking-wider" style={{ color: 'rgba(14, 165, 233, 0.6)' }}>$ cat score.json</h3>
          <pre className="text-gray-400 text-sm font-mono overflow-x-auto">
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
          {[
            { price: '$5/mo', desc: 'Score updates + failure categories', border: 'rgba(14,165,233,0.08)' },
            { price: '$20/mo', desc: 'Full inference logs + rule distribution', border: 'rgba(14,165,233,0.2)' },
            { price: '$50/mo', desc: 'Raw eval data + DSL drafts + architecture', border: 'rgba(14,165,233,0.35)' },
          ].map((tier, i) => (
            <div
              key={i}
              className="rounded-xl p-4"
              style={{
                background: 'rgba(10, 10, 20, 0.6)',
                border: `1px solid ${tier.border}`,
              }}
            >
              <div className="font-bold text-sm" style={{ color: 'rgba(14, 165, 233, 0.8)' }}>{tier.price}</div>
              <div className="text-gray-500 text-xs mt-1">{tier.desc}</div>
            </div>
          ))}
        </div>

        <div className="flex gap-4 justify-center">
          <a
            href="https://github.com/sponsors/Ag3497120"
            className="px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-300"
            style={{
              background: 'rgba(14, 165, 233, 0.1)',
              border: '1px solid rgba(14, 165, 233, 0.3)',
              color: '#e2e8f0',
            }}
            target="_blank"
          >
            Become a Sponsor
          </a>
          <Link
            href="/"
            className="px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-300 text-gray-400"
            style={{
              border: '1px solid rgba(107, 114, 128, 0.2)',
            }}
          >
            Back to Verantyx
          </Link>
        </div>

        <p className="text-gray-700 text-xs mt-8">
          Public data available at{' '}
          <a href="/score.json" className="text-gray-600 hover:text-gray-400 transition-colors">/score.json</a>
          {' '}·{' '}
          <a href="https://github.com/Ag3497120/verantyx-v6" className="text-gray-600 hover:text-gray-400 transition-colors" target="_blank">
            GitHub
          </a>
        </p>
      </div>
    </div>
  )
}
