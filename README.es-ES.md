

# Sitio Web de Verantyx

Sitio web de vanguardia para **Verantyx** — un motor de razonamiento simbólico sin LLM.

## Características

- **Next.js 15** con App Router y Edge Runtime
- **TypeScript** para seguridad de tipos
- **Tailwind CSS v4** para estilos
- **Framer Motion** para animaciones fluidas
- **Gráficos SVG puros** — cero librerías de gráficos
- **Fondo de partículas basado en Canvas** con efectos de constelaciones
- **Diseño responsivo** — se ve genial en todos los dispositivos
- **Animaciones a 60 fps** en todo el sitio

## Destacados de Rendimiento

- **ARC-AGI-2**: 20.7% (207/1000)
- **Humanity's Last Exam**: 4.6%
- **Cero redes neuronales** — razonamiento simbólico puro
- **Cada solución es verificable**

## Arquitectura

Pipeline de resolución de siete fases:
1. Cross DSL (Neighborhood Rules)
2. Standalone Primitives
3. Stamp/Pattern Fill
4. Composite Chains
5. Iterative Cross
6. Puzzle Reasoning Language
7. ProgramTree Synthesis

## Desarrollo

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

## Stack Tecnológico

- Next.js 15.1.0
- React 19.0.0
- TypeScript 5.7.2
- Tailwind CSS 4.0.0
- Framer Motion 11.11.17

## Repositorio

[Ver en GitHub](https://github.com/Ag3497120/verantyx-v6)

## Creado por

kofdai

## Proxy de la API de Apple Music (para amigos)

Proxy de borde solo de catálogo para aplicaciones amigas:

`Friend app → https://verantyx.ai/api/apple-music/* → Apple Music API`

Implementado como **Cloudflare Pages Functions** bajo `functions/api/apple-music/` (las rutas API de Next.js no se ejecutan en esta exportación estática). Documentación: [/apple-music-api/](https://verantyx.ai/apple-music-api/).

| Endpoint | Autenticación | Notas |
|----------|------|--------|
| `GET /api/apple-music/search?term=&types=&storefront=` | `x-api-key` | Búsqueda de catálogo |
| `GET /api/apple-music/health` | ninguna | Sin secretos en la respuesta |

**Secretos** (Cloudflare Pages → Configuración → Variables de entorno; cifrar secretos):

- `APPLE_TEAM_ID`, `APPLE_MUSIC_KEY_ID`, `APPLE_MUSIC_PRIVATE_KEY` (PEM — nunca confirmar/commit `.p8`)
- `FRIEND_API_KEYS` (separados por comas)

Límite de tasa: **60 req / 60s / clave API** (en memoria por aislado de CF). Consulte `.env.example` y la página de documentación para los pasos de curl y `wrangler pages dev` local.

## Cloudflare Pages

Este sitio es una **exportación estática de Next.js**. En producción, se deben publicar los contenidos de `out/` como la raíz del sitio (no la raíz del repositorio).

`wrangler.toml` establece `pages_build_output_dir = "out"` para que Pages no publique la raíz de Git (lo que hacía que `/` diera 404 mientras el sitio real vivía bajo `/out/`). `out/` está **confirmado en el repositorio** para que, si se omite un paso de compilación, aún haya algo que publicar. Las funciones en `/functions` se despliegan junto con el proyecto Pages.

| Configuración | Valor |
|--------|--------|
| Preset de framework | Ninguno (o "Next.js (Static HTML Export)") |
| Comando de compilación | `npm run build` (recomendado; o `npm run cf-build`) |
| Directorio de salida de compilación | `out` (también fijado en `wrangler.toml`) |
| Directorio raíz | `/` |
| Versión de Node | 20 (consulte `.nvmrc`) |

No configure el directorio de salida como `public`, `.next`, ni lo deje en blanco (en blanco/`/` publica la raíz del repositorio). Es preferible regenerar `out/` con `npm run build` antes de liberar.

Si un despliegue "tiene éxito" pero `/` da error 404 mientras `/out/` funciona: el directorio de salida es incorrecto — corrija el panel de control y mantenga `wrangler.toml`.

Si un despliegue tiene éxito pero el sitio sigue mostrando el texto antiguo de la página de inicio (por ejemplo, "Explore Our Projects" en lugar de "Flagship: Verantyx-CLI"):

1. Pages → Settings → Builds → **Clear build cache**
2. Reintente el despliegue (o publique un commit vacío)
3. Caching → **Purge Everything** para `verantyx.ai` / `www.verantyx.ai`

---

**Sin LLMs. Sin redes neuronales. Sin modelos preentrenados.**
