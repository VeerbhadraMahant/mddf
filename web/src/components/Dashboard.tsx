import { useEffect, useMemo, useState } from "react";
import { api, type BenchmarkRow, type ModelName } from "../api";

const MODEL_COLOR: Record<ModelName, string> = {
  padim: "#f59e0b",
  patchcore: "#22d3ee",
  efficient_ad: "#a78bfa",
};

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

export function Dashboard() {
  const [rows, setRows] = useState<BenchmarkRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .benchmark()
      .then((r) => setRows(r.rows))
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const byModel = useMemo(() => {
    const g: Partial<Record<ModelName, BenchmarkRow[]>> = {};
    for (const r of rows ?? []) (g[r.model] ??= []).push(r);
    return g;
  }, [rows]);

  if (error) return <p className="text-rose-400">{error}</p>;
  if (!rows) return <p className="text-gray-500">Loading…</p>;
  if (rows.length === 0)
    return (
      <p className="text-gray-500">
        No results yet — run <code>mddf train</code>, <code>mddf export</code>,{" "}
        <code>mddf benchmark --latency</code>.
      </p>
    );

  const models = Object.keys(byModel) as ModelName[];
  const latencies = rows.map((r) => r.latency_ms_p50).filter((x): x is number => x != null);
  const maxLat = Math.max(60, ...latencies) * 1.1;

  return (
    <div className="space-y-8">
      {/* stat tiles */}
      <div className="grid gap-3 sm:grid-cols-3">
        {models.map((m) => {
          const rs = byModel[m] ?? [];
          return (
            <div key={m} className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
              <div className="flex items-center gap-2">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: MODEL_COLOR[m] }}
                />
                <span className="text-sm font-medium text-gray-200">{m}</span>
              </div>
              <dl className="mt-2 space-y-1 text-xs text-gray-400">
                <div className="flex justify-between">
                  <dt>mean image AUROC</dt>
                  <dd className="tabular-nums text-gray-200">
                    {mean(rs.map((r) => r.image_auroc)).toFixed(4)}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt>mean AUPRO</dt>
                  <dd className="tabular-nums text-gray-200">
                    {mean(rs.map((r) => r.aupro ?? 0)).toFixed(4)}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt>mean CPU p50</dt>
                  <dd className="tabular-nums text-gray-200">
                    {mean(rs.map((r) => r.latency_ms_p50 ?? 0)).toFixed(1)} ms
                  </dd>
                </div>
              </dl>
            </div>
          );
        })}
      </div>

      {/* accuracy vs latency scatter */}
      <figure className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
        <figcaption className="mb-3 text-sm font-medium text-gray-200">
          Accuracy vs. CPU latency (each point = one category)
        </figcaption>
        <svg viewBox="0 0 520 300" className="w-full">
          {/* axes */}
          <line x1="46" y1="10" x2="46" y2="260" stroke="#374151" />
          <line x1="46" y1="260" x2="510" y2="260" stroke="#374151" />
          {[0.8, 0.85, 0.9, 0.95, 1.0].map((v) => {
            const y = 260 - ((v - 0.8) / 0.2) * 250;
            return (
              <g key={v}>
                <line x1="46" y1={y} x2="510" y2={y} stroke="#1f2937" />
                <text x="40" y={y + 4} textAnchor="end" fontSize="9" fill="#6b7280">
                  {v.toFixed(2)}
                </text>
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text key={f} x={46 + f * 464} y="275" textAnchor="middle" fontSize="9" fill="#6b7280">
              {Math.round(f * maxLat)}
            </text>
          ))}
          <text x="278" y="293" textAnchor="middle" fontSize="10" fill="#9ca3af">
            CPU p50 latency (ms)
          </text>
          {rows.map((r, i) => {
            if (r.latency_ms_p50 == null) return null;
            const x = 46 + (r.latency_ms_p50 / maxLat) * 464;
            const y = 260 - ((Math.max(0.8, r.image_auroc) - 0.8) / 0.2) * 250;
            return (
              <circle
                key={i}
                cx={x}
                cy={y}
                r="4"
                fill={MODEL_COLOR[r.model]}
                fillOpacity="0.8"
              >
                <title>
                  {r.category} · {r.model} · AUROC {r.image_auroc.toFixed(3)} ·{" "}
                  {r.latency_ms_p50.toFixed(1)} ms
                </title>
              </circle>
            );
          })}
        </svg>
        <Legend models={models} />
      </figure>

      {/* per-category image AUROC bars */}
      <figure className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
        <figcaption className="mb-3 text-sm font-medium text-gray-200">
          Image AUROC by category (dashed = published PatchCore baseline)
        </figcaption>
        <CategoryBars rows={rows} />
        <Legend models={models} />
      </figure>
    </div>
  );
}

function Legend({ models }: { models: ModelName[] }) {
  return (
    <div className="mt-2 flex flex-wrap gap-4 text-xs text-gray-400">
      {models.map((m) => (
        <span key={m} className="flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ background: MODEL_COLOR[m] }}
          />
          {m}
        </span>
      ))}
    </div>
  );
}

function CategoryBars({ rows }: { rows: BenchmarkRow[] }) {
  const cats = Array.from(new Set(rows.map((r) => r.category)));
  const models = Array.from(new Set(rows.map((r) => r.model))) as ModelName[];
  const W = 520;
  const rowH = 26;
  const H = cats.length * rowH + 20;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {cats.map((cat, ci) => {
        const y = ci * rowH + 6;
        const catRows = rows.filter((r) => r.category === cat);
        const published = catRows[0]?.published_image_auroc ?? 1;
        const scale = (v: number) => 90 + ((Math.max(0.8, v) - 0.8) / 0.2) * (W - 110);
        return (
          <g key={cat}>
            <text x="84" y={y + 13} textAnchor="end" fontSize="9" fill="#9ca3af">
              {cat}
            </text>
            {models.map((m, mi) => {
              const r = catRows.find((x) => x.model === m);
              if (!r) return null;
              const bw = mi === 0 ? 6 : 6;
              const by = y + mi * 6;
              return (
                <rect
                  key={m}
                  x="90"
                  y={by}
                  width={scale(r.image_auroc) - 90}
                  height={bw - 1}
                  fill={MODEL_COLOR[m]}
                  fillOpacity="0.85"
                >
                  <title>
                    {cat} · {m} · {r.image_auroc.toFixed(3)}
                  </title>
                </rect>
              );
            })}
            <line
              x1={scale(published)}
              y1={y}
              x2={scale(published)}
              y2={y + models.length * 6}
              stroke="#e5e7eb"
              strokeDasharray="2 2"
            />
          </g>
        );
      })}
    </svg>
  );
}
