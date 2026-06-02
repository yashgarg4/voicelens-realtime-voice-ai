// A small pure-SVG radar (triangle) chart for the three scoring axes.
// No charting library — just trig on three axes at -90°, 30°, 150°.

interface Props {
  content: number; // 0-10
  depth: number; // 0-10
  structure: number; // 0-10
  size?: number;
}

const AXES = [
  { key: "Content", angle: -90 },
  { key: "Depth", angle: 30 },
  { key: "Structure", angle: 150 },
] as const;

function point(cx: number, cy: number, r: number, angleDeg: number) {
  const a = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

export function RadarChart({ content, depth, structure, size = 200 }: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const R = size * 0.36; // max radius (value = 10)
  const values = [content, depth, structure];

  // Grid rings at 1/3, 2/3, full.
  const rings = [1 / 3, 2 / 3, 1].map((frac) =>
    AXES.map((ax) => point(cx, cy, R * frac, ax.angle))
      .map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`)
      .join(" ")
  );

  // Data polygon (value/10 of R along each axis).
  const dataPts = AXES.map((ax, i) =>
    point(cx, cy, R * (Math.max(0, Math.min(10, values[i])) / 10), ax.angle)
  );
  const dataPoly = dataPts
    .map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(" ");

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label="Score radar chart"
    >
      {/* grid rings */}
      {rings.map((pts, i) => (
        <polygon
          key={i}
          points={pts}
          fill="none"
          stroke="rgb(51,65,85)"
          strokeWidth={1}
        />
      ))}
      {/* axis spokes + labels */}
      {AXES.map((ax, i) => {
        const end = point(cx, cy, R, ax.angle);
        const label = point(cx, cy, R + 18, ax.angle);
        return (
          <g key={ax.key}>
            <line
              x1={cx}
              y1={cy}
              x2={end.x}
              y2={end.y}
              stroke="rgb(51,65,85)"
              strokeWidth={1}
            />
            <text
              x={label.x}
              y={label.y}
              fill="rgb(148,163,184)"
              fontSize={11}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {ax.key} {values[i]}
            </text>
          </g>
        );
      })}
      {/* data polygon */}
      <polygon
        points={dataPoly}
        fill="rgba(16,185,129,0.25)"
        stroke="rgb(16,185,129)"
        strokeWidth={2}
      />
      {dataPts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3} fill="rgb(16,185,129)" />
      ))}
    </svg>
  );
}
