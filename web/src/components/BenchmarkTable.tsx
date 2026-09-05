import { useEffect, useState } from "react";
import { api, type BenchmarkRow } from "../api";

export function BenchmarkTable() {
  const [rows, setRows] = useState<BenchmarkRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .benchmark()
      .then((r) => setRows(r.rows))
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <p className="text-rose-400">{error}</p>;
  if (!rows) return <p className="text-gray-500">Loading…</p>;
  if (rows.length === 0)
    return (
      <p className="text-gray-500">
        No results yet — train and export models, then run <code>mddf benchmark</code>.
      </p>
    );

  const fmt = (v: number | null, d = 3) => (v == null ? "—" : v.toFixed(d));

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase tracking-wide text-gray-500">
          <tr>
            <th className="py-2 pr-4">Category</th>
            <th className="py-2 pr-4">Model</th>
            <th className="py-2 pr-4 text-right">Image AUROC</th>
            <th className="py-2 pr-4 text-right">Published</th>
            <th className="py-2 pr-4 text-right">Δ</th>
            <th className="py-2 pr-4 text-right">Pixel AUROC</th>
            <th className="py-2 pr-4 text-right">AUPRO</th>
            <th className="py-2 pr-4 text-right">CPU p50 (ms)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {rows.map((r, i) => {
            const delta = r.image_auroc - r.published_image_auroc;
            return (
              <tr key={i} className="text-gray-300">
                <td className="py-2 pr-4">{r.category}</td>
                <td className="py-2 pr-4">{r.model}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{fmt(r.image_auroc)}</td>
                <td className="py-2 pr-4 text-right tabular-nums text-gray-500">
                  {fmt(r.published_image_auroc)}
                </td>
                <td
                  className={`py-2 pr-4 text-right tabular-nums ${
                    delta >= 0 ? "text-emerald-400" : "text-amber-400"
                  }`}
                >
                  {delta >= 0 ? "+" : ""}
                  {delta.toFixed(3)}
                </td>
                <td className="py-2 pr-4 text-right tabular-nums">{fmt(r.pixel_auroc)}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{fmt(r.aupro)}</td>
                <td className="py-2 pr-4 text-right tabular-nums">{fmt(r.latency_ms_p50, 1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
