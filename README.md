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

## Cloudflare Pages

This site is a **Next.js static export**. Production must build into `out/`.

`wrangler.toml` is intentionally **not** used: when present with `pages_build_output_dir`, Cloudflare can leave the dashboard Build command empty and skip `npm run build`, then fail with `Output directory "out" not found`. Configure builds in the Pages dashboard only.

| Setting | Value |
|--------|--------|
| Framework preset | None (or "Next.js (Static HTML Export)") |
| Build command | `npm run build` (or `npm ci && npm run build` / `npm run cf-build`) — or leave empty if publishing committed `out/` |
| Build output directory | `out` |
| Root directory | `/` |
| Node version | 20+ (see `.nvmrc`) |

Do **not** set the output directory to `public` or `.next`. `public/` is only for static assets copied into `out/`.

`out/` **is committed** as a safety net: production previously 404'd every in-site route when Pages skipped the build and published an empty/missing `out/`. Prefer regenerating with `npm run build` before release; the committed export keeps the site online even if the remote build step is misconfigured.

If the build log says `No build command specified. Skipping build step`, either set **Build command** to `npm run build` under Settings → Builds, or rely on the committed `out/` (and clear build cache if needed).

If a deploy succeeds but the site still shows old homepage copy (e.g. "Explore Our Projects" instead of "Flagship: Verantyx-CLI"):

1. Pages → Settings → Builds → **Clear build cache**
2. Retry the deployment (or push an empty commit)
3. Caching → **Purge Everything** for `verantyx.ai` / `www.verantyx.ai`

---

**No LLMs. No neural networks. No pretrained models.**
