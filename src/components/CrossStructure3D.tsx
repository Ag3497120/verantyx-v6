'use client';

import { useEffect, useRef } from 'react';

export default function CrossStructure3D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animFrame = 0;
    let mouseX = 0.5;
    let mouseY = 0.5;
    let time = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const handleMouse = (e: MouseEvent) => {
      mouseX = e.clientX / window.innerWidth;
      mouseY = e.clientY / window.innerHeight;
    };
    window.addEventListener('mousemove', handleMouse);

    // 3D Cross vertices (Von Neumann: center + 4 cardinal arms)
    // Each arm has depth (z) for 3D effect
    interface Point3D { x: number; y: number; z: number; }

    function project(p: Point3D, cx: number, cy: number, fov: number, rotX: number, rotY: number): { x: number; y: number; scale: number } {
      // Rotate around Y axis
      let x = p.x * Math.cos(rotY) - p.z * Math.sin(rotY);
      let z = p.x * Math.sin(rotY) + p.z * Math.cos(rotY);
      let y = p.y;

      // Rotate around X axis
      const y2 = y * Math.cos(rotX) - z * Math.sin(rotX);
      const z2 = y * Math.sin(rotX) + z * Math.cos(rotX);
      y = y2;
      z = z2;

      const scale = fov / (fov + z + 400);
      return { x: cx + x * scale, y: cy + y * scale, scale };
    }

    function drawCross3D(
      ctx: CanvasRenderingContext2D,
      cx: number, cy: number,
      size: number, depth: number,
      rotX: number, rotY: number, rotZ: number,
      color: string, alpha: number, fov: number,
      pulse: number
    ) {
      const armLength = size;
      const armWidth = size * 0.3;
      const armDepth = depth;

      // Define the 6 faces of each arm as quads in 3D
      // Cross has 5 parts: center cube + 4 arms
      const parts: Point3D[][] = [];

      // Center cube
      const cw = armWidth;
      parts.push([
        { x: -cw, y: -cw, z: -cw }, { x: cw, y: -cw, z: -cw },
        { x: cw, y: cw, z: -cw }, { x: -cw, y: cw, z: -cw },
      ]);
      parts.push([
        { x: -cw, y: -cw, z: cw }, { x: cw, y: -cw, z: cw },
        { x: cw, y: cw, z: cw }, { x: -cw, y: cw, z: cw },
      ]);

      // Right arm (+X)
      parts.push([
        { x: cw, y: -cw, z: -cw }, { x: armLength, y: -cw, z: -cw },
        { x: armLength, y: cw, z: -cw }, { x: cw, y: cw, z: -cw },
      ]);
      parts.push([
        { x: cw, y: -cw, z: cw }, { x: armLength, y: -cw, z: cw },
        { x: armLength, y: cw, z: cw }, { x: cw, y: cw, z: cw },
      ]);
      parts.push([
        { x: cw, y: -cw, z: -cw }, { x: armLength, y: -cw, z: -cw },
        { x: armLength, y: -cw, z: cw }, { x: cw, y: -cw, z: cw },
      ]);
      parts.push([
        { x: cw, y: cw, z: -cw }, { x: armLength, y: cw, z: -cw },
        { x: armLength, y: cw, z: cw }, { x: cw, y: cw, z: cw },
      ]);
      // End cap
      parts.push([
        { x: armLength, y: -cw, z: -cw }, { x: armLength, y: cw, z: -cw },
        { x: armLength, y: cw, z: cw }, { x: armLength, y: -cw, z: cw },
      ]);

      // Left arm (-X)
      parts.push([
        { x: -cw, y: -cw, z: -cw }, { x: -armLength, y: -cw, z: -cw },
        { x: -armLength, y: cw, z: -cw }, { x: -cw, y: cw, z: -cw },
      ]);
      parts.push([
        { x: -cw, y: -cw, z: cw }, { x: -armLength, y: -cw, z: cw },
        { x: -armLength, y: cw, z: cw }, { x: -cw, y: cw, z: cw },
      ]);
      parts.push([
        { x: -armLength, y: -cw, z: -cw }, { x: -armLength, y: cw, z: -cw },
        { x: -armLength, y: cw, z: cw }, { x: -armLength, y: -cw, z: cw },
      ]);

      // Up arm (-Y)
      parts.push([
        { x: -cw, y: -cw, z: -cw }, { x: cw, y: -cw, z: -cw },
        { x: cw, y: -armLength, z: -cw }, { x: -cw, y: -armLength, z: -cw },
      ]);
      parts.push([
        { x: -cw, y: -cw, z: cw }, { x: cw, y: -cw, z: cw },
        { x: cw, y: -armLength, z: cw }, { x: -cw, y: -armLength, z: cw },
      ]);
      parts.push([
        { x: -cw, y: -armLength, z: -cw }, { x: cw, y: -armLength, z: -cw },
        { x: cw, y: -armLength, z: cw }, { x: -cw, y: -armLength, z: cw },
      ]);

      // Down arm (+Y)
      parts.push([
        { x: -cw, y: cw, z: -cw }, { x: cw, y: cw, z: -cw },
        { x: cw, y: armLength, z: -cw }, { x: -cw, y: armLength, z: -cw },
      ]);
      parts.push([
        { x: -cw, y: cw, z: cw }, { x: cw, y: cw, z: cw },
        { x: cw, y: armLength, z: cw }, { x: -cw, y: armLength, z: cw },
      ]);
      parts.push([
        { x: -cw, y: armLength, z: -cw }, { x: cw, y: armLength, z: -cw },
        { x: cw, y: armLength, z: cw }, { x: -cw, y: armLength, z: cw },
      ]);

      // Apply rotZ to all points
      const cosZ = Math.cos(rotZ);
      const sinZ = Math.sin(rotZ);

      // Sort faces by average z (painter's algorithm)
      const projected = parts.map(face => {
        const pts = face.map(p => {
          const rx = p.x * cosZ - p.y * sinZ;
          const ry = p.x * sinZ + p.y * cosZ;
          return project({ x: rx, y: ry, z: p.z }, cx, cy, fov, rotX, rotY);
        });
        const avgZ = face.reduce((s, p) => {
          const rx = p.x * cosZ - p.y * sinZ;
          const ry = p.x * sinZ + p.y * cosZ;
          const z = rx * Math.sin(rotY) + p.z * Math.cos(rotY);
          const z2 = ry * Math.sin(rotX) + z * Math.cos(rotX);
          return s + z2;
        }, 0) / face.length;
        return { pts, avgZ };
      });

      projected.sort((a, b) => a.avgZ - b.avgZ);

      for (const { pts } of projected) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();

        const pulseAlpha = alpha * (0.5 + 0.5 * Math.sin(pulse));
        ctx.fillStyle = color.replace(')', `,${pulseAlpha * 0.3})`).replace('rgb', 'rgba');
        ctx.fill();
        ctx.strokeStyle = color.replace(')', `,${pulseAlpha * 0.8})`).replace('rgb', 'rgba');
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    function drawElectricPulse(ctx: CanvasRenderingContext2D, cx: number, cy: number, size: number, t: number) {
      // Electric pulses traveling along the cross arms
      const armLength = size;
      const directions = [
        { dx: 1, dy: 0 }, { dx: -1, dy: 0 },
        { dx: 0, dy: -1 }, { dx: 0, dy: 1 },
      ];

      for (const dir of directions) {
        const progress = ((t * 2) % 1);
        const px = cx + dir.dx * armLength * progress;
        const py = cy + dir.dy * armLength * progress;

        const gradient = ctx.createRadialGradient(px, py, 0, px, py, 15);
        gradient.addColorStop(0, 'rgba(14, 165, 233, 0.8)');
        gradient.addColorStop(0.5, 'rgba(14, 165, 233, 0.3)');
        gradient.addColorStop(1, 'rgba(14, 165, 233, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(px - 15, py - 15, 30, 30);
      }
    }

    function animate() {
      if (!canvas || !ctx) return;
      time += 0.005;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const baseRotX = (mouseY - 0.5) * 0.4 + Math.sin(time * 0.3) * 0.1;
      const baseRotY = (mouseX - 0.5) * 0.4 + time * 0.2;
      const baseRotZ = Math.sin(time * 0.15) * 0.05;

      // Main large cross
      drawCross3D(ctx, cx, cy, 180, 40, baseRotX, baseRotY, baseRotZ, 'rgb(14, 165, 233)', 0.8, 800, time * 2);

      // Nested smaller crosses at each arm tip
      const nestedSize = 60;
      const tipDist = 220;
      const nestAngles = [
        { ox: tipDist, oy: 0 },
        { ox: -tipDist, oy: 0 },
        { ox: 0, oy: -tipDist },
        { ox: 0, oy: tipDist },
      ];

      for (let i = 0; i < nestAngles.length; i++) {
        const n = nestAngles[i];
        // Project the offset through the same rotation
        const p = project(
          { x: n.ox, y: n.oy, z: 0 },
          cx, cy, 800, baseRotX, baseRotY
        );
        drawCross3D(ctx, p.x, p.y, nestedSize * p.scale, 15, baseRotX + time * 0.5, baseRotY + i * 0.3, baseRotZ, 'rgb(168, 85, 247)', 0.5, 800, time * 3 + i);
      }

      // Electric pulses
      drawElectricPulse(ctx, cx, cy, 180, time);

      animFrame = requestAnimationFrame(animate);
    }

    animate();

    return () => {
      cancelAnimationFrame(animFrame);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouse);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  );
}
