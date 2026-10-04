// Put the Cleanroom workspace (built from github.com/Ag3497120/cleanroom with SITE_ROLE unset)
// at /cleanroom/ of verantyx.ai, after the Next build of this site. Other pages (/apps, /author,
// /vera, ...), Pages Functions and this site's _headers/_redirects/_routes.json are kept.
// Cleanroom の workspace は /cleanroom/ に置く。/ は Next のページ（Vera のホーム）。
// index.html / index.txt を out/ 直下に置くとホームを上書きするので、出力先を out/cleanroom/ に変えた。
import { cpSync, existsSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const source = 'cleanroom-export', target = 'out';
const keep = new Set(['_headers', '_redirects', '_routes.json', 'api', '404.html', 'robots.txt', 'sitemap.xml', 'app-ads.txt', 'apps', 'author', 'vera', 'privacy', 'support', 'terms', 'writing', 'compute']);
const moved = ['index.html', 'index.txt'];
const movedTo = join(target, 'cleanroom');
if (!existsSync(source)) throw new Error('cleanroom-export is missing; build Cleanroom and copy dist/pages here');
if (!existsSync(target)) throw new Error('run the Next build first');
let copied = 0;
for (const name of readdirSync(source)) {
  if (moved.includes(name)) continue;
  if (keep.has(name) && existsSync(join(target, name)) && statSync(join(target, name)).isDirectory() !== false && name !== 'vera') continue;
  if (name === 'vera' && existsSync(join(target, 'vera'))) {
    // /vera/ is this site's page; Cleanroom's reader files live in /vera/*.onnx|json|ort — merge them in.
    cpSync(join(source, 'vera'), join(target, 'vera'), { recursive: true, force: false, errorOnExist: false });
    copied++; continue;
  }
  cpSync(join(source, name), join(target, name), { recursive: true, force: true });
  copied++;
}
if (existsSync(join(source, 'cleanroom', 'index.html'))) {
  throw new Error('cleanroom-export/cleanroom/index.html exists; it would collide with the workspace moved to /cleanroom/');
}
mkdirSync(movedTo, { recursive: true });
for (const name of moved) {
  if (!existsSync(join(source, name))) continue;
  cpSync(join(source, name), join(movedTo, name), { force: true });
  copied++;
}
console.log(`overlay-cleanroom: copied ${copied} top-level entries from ${source}; ${moved.join(', ')} → ${movedTo}/`);
