/* The stereo cross — the mark, and the structure the engine is named for.
 *
 * Three mutually perpendicular bars through one origin, drawn in isometric
 * projection: six arms, each split along its own axis so the two visible
 * faces of the prism read as light and shadow. The left-hand faces take the
 * teal, the right-hand faces the blue, and the point where all three axes
 * meet carries the light — which is exactly where a cross store puts a core.
 *
 * Inline SVG rather than an image file: it stays sharp at every size, it
 * costs no request, and it can be recoloured by the theme.
 */

export default function Logo({
  size = 22,
  glow = true,
}: {
  size?: number | string;
  glow?: boolean;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      style={{ display: 'block', flexShrink: 0 }}
    >
      <defs>
        <linearGradient id="vx-teal" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#5FD3D0" />
          <stop offset="100%" stopColor="#2FA8B8" />
        </linearGradient>
        <linearGradient id="vx-blue" x1="1" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2B7FC7" />
          <stop offset="100%" stopColor="#1B5C9E" />
        </linearGradient>
        <linearGradient id="vx-mid" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#49BFC6" />
          <stop offset="100%" stopColor="#2570B4" />
        </linearGradient>
        <radialGradient id="vx-core" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%" stopColor="#FFFFFF" stopOpacity="1" />
          <stop offset="45%" stopColor="#BDF0FF" stopOpacity="0.75" />
          <stop offset="100%" stopColor="#7FE3FF" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Left half of the figure — the lit faces */}
      <polygon points="100.0,100.0 23.8,56.0 37.3,32.6 113.5,76.6" fill="url(#vx-teal)" />
      <polygon points="100.0,100.0 23.8,56.0 10.3,79.4 86.5,123.4" fill="url(#vx-mid)" />
      <polygon points="100.0,100.0 23.8,144.0 10.3,120.6 86.5,76.6" fill="url(#vx-teal)" />
      <polygon points="100.0,100.0 23.8,144.0 37.3,167.4 113.5,123.4" fill="url(#vx-mid)" />

      {/* Vertical bar — one face to each side, which is what makes it read
          as a solid rather than as a flat asterisk */}
      <polygon points="100.0,100.0 100.0,12.0 73.0,12.0 73.0,100.0" fill="url(#vx-teal)" />
      <polygon points="100.0,100.0 100.0,12.0 127.0,12.0 127.0,100.0" fill="url(#vx-blue)" />
      <polygon points="100.0,100.0 100.0,188.0 73.0,188.0 73.0,100.0" fill="url(#vx-mid)" />
      <polygon points="100.0,100.0 100.0,188.0 127.0,188.0 127.0,100.0" fill="url(#vx-blue)" />

      {/* Right half — the shaded faces */}
      <polygon points="100.0,100.0 176.2,56.0 162.7,32.6 86.5,76.6" fill="url(#vx-blue)" />
      <polygon points="100.0,100.0 176.2,56.0 189.7,79.4 113.5,123.4" fill="url(#vx-mid)" />
      <polygon points="100.0,100.0 176.2,144.0 189.7,120.6 113.5,76.6" fill="url(#vx-blue)" />
      <polygon points="100.0,100.0 176.2,144.0 162.7,167.4 86.5,123.4" fill="url(#vx-mid)" />

      {glow && <circle cx="100" cy="100" r="52" fill="url(#vx-core)" />}
    </svg>
  );
}
