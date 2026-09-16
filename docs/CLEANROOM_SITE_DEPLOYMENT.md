# verantyx.ai publication

The current Cloudflare Pages integration publishes the tracked `out/` directory. A successful deployment does not prove that `src/` was rebuilt. On 2026-09-16, the public homepage hash matched the old tracked `out/index.html` even after source commits deployed successfully.

## Release procedure

1. Update the Next.js source and public assets.
2. Run `npm run build` to regenerate the static export.
3. Include the regenerated `out/` files and deletions in the release commit. Do not include `.next/`, local credentials, `.env` files, private notebooks, or profiles.
4. Wait for Cloudflare Pages to complete.
5. Check the actual custom domain, retained routes, removed routes, and light/dark control. Do not use the deployment badge alone as evidence.

Home, Vera and Apps remain in navigation. App detail pages, author information, legal/support pages and existing unlisted utilities remain. The retired .jcross, API documentation, catalogue, 3D, demo, CLI and install pages must not remain in the exported artifact.

If the Cloudflare project is later configured to run `npm run cf-build` before publishing `out`, this prebuilt-artifact release procedure can be replaced after confirming the real deployment behavior. Do not change domain bindings or secrets merely to work around a stale export.
