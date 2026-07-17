# Verantyx Website

Cutting-edge website for **Verantyx** — an LLM-free symbolic reasoning engine.

## Features

- **Next.js 15** with App Router and Edge Runtime
- **TypeScript** for type safety
- **Tailwind CSS v4** for styling
- **Framer Motion** for smooth animations
- **Pure SVG charts** — zero chart libraries
- **Canvas-based particle background** with constellation effects
- **Responsive design** — looks great on all devices
- **60fps animations** throughout

## Performance Highlights

- **ARC-AGI-2**: 20.7% (207/1000)
- **Humanity's Last Exam**: 4.6%
- **Zero neural networks** — pure symbolic reasoning
- **Every solution is verifiable**

## Architecture

Seven-phase solving pipeline:
1. Cross DSL (Neighborhood Rules)
2. Standalone Primitives
3. Stamp/Pattern Fill
4. Composite Chains
5. Iterative Cross
6. Puzzle Reasoning Language
7. ProgramTree Synthesis

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Tech Stack

- Next.js 15.1.0
- React 19.0.0
- TypeScript 5.7.2
- Tailwind CSS 4.0.0
- Framer Motion 11.11.17

## Repository

[View on GitHub](https://github.com/Ag3497120/verantyx-v6)

## Built By

kofdai

## Apple Music API proxy (friends)

Catalog-only edge proxy for friend apps:

`Friend app → https://verantyx.ai/api/apple-music/* → Apple Music API`

Implemented as **Cloudflare Pages Functions** under `functions/api/apple-music/` (Next.js API routes do not run on this static export). Docs: [/apple-music-api/](https://verantyx.ai/apple-music-api/).

| Endpoint | Auth | Notes |
|----------|------|--------|
| `GET /api/apple-music/search?term=&types=&storefront=` | `x-api-key` | Catalog search |
| `GET /api/apple-music/health` | none | No secrets in response |

**Secrets** (Cloudflare Pages → Settings → Environment variables; encrypt secrets):

- `APPLE_TEAM_ID`, `APPLE_MUSIC_KEY_ID`, `APPLE_MUSIC_PRIVATE_KEY` (PEM — never commit `.p8`)
- `FRIEND_API_KEYS` (comma-separated)

Rate limit: **60 req / 60s / API key** (in-memory per CF isolate). See `.env.example` and the docs page for curl + local `wrangler pages dev` steps.

## Cloudflare Pages

This site is a **Next.js static export**. Production must publish the contents of `out/` as the site root (not the repo root).

`wrangler.toml` sets `pages_build_output_dir = "out"` so Pages does not publish the Git root (which made `/` 404 while the real site lived under `/out/`). `out/` is **committed** so a skipped build step still has something to publish. Functions in `/functions` deploy with the Pages project.

| Setting | Value |
|--------|--------|
| Framework preset | None (or "Next.js (Static HTML Export)") |
| Build command | `npm run build` (preferred; or `npm run cf-build`) |
| Build output directory | `out` (also pinned in `wrangler.toml`) |
| Root directory | `/` |
| Node version | 20 (see `.nvmrc`) |

Do **not** set the output directory to `public`, `.next`, or leave it blank (blank/`/` publishes the repo root). Prefer regenerating `out/` with `npm run build` before release.

If a deploy “succeeds” but `/` 404s while `/out/` works: output dir is wrong — fix dashboard + keep `wrangler.toml`.

If a deploy succeeds but the site still shows old homepage copy (e.g. "Explore Our Projects" instead of "Flagship: Verantyx-CLI"):

1. Pages → Settings → Builds → **Clear build cache**
2. Retry the deployment (or push an empty commit)
3. Caching → **Purge Everything** for `verantyx.ai` / `www.verantyx.ai`

---

**No LLMs. No neural networks. No pretrained models.**
