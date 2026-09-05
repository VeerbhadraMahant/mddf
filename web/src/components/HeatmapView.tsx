import { useState } from "react";
import type { BBox } from "../api";

interface Props {
  original: string; // object URL
  heatmapPng: string; // base64 (no data: prefix)
  bboxes: BBox[];
  naturalWidth: number;
  naturalHeight: number;
}

export function HeatmapView({ original, heatmapPng, bboxes, naturalWidth, naturalHeight }: Props) {
  const [alpha, setAlpha] = useState(0.55);
  const heat = `data:image/png;base64,${heatmapPng}`;

  return (
    <div className="space-y-3">
      <div className="grid gap-4 sm:grid-cols-2">
        <figure className="space-y-1">
          <figcaption className="text-xs uppercase tracking-wide text-gray-500">Input</figcaption>
          <img src={original} className="w-full rounded-lg border border-gray-800" alt="input" />
        </figure>
        <figure className="space-y-1">
          <figcaption className="text-xs uppercase tracking-wide text-gray-500">
            Anomaly localization
          </figcaption>
          <div
            className="relative w-full overflow-hidden rounded-lg border border-gray-800"
            style={{ aspectRatio: `${naturalWidth} / ${naturalHeight}` }}
          >
            <img src={original} className="absolute inset-0 h-full w-full" alt="" />
            <img
              src={heat}
              className="absolute inset-0 h-full w-full mix-blend-normal"
              style={{ opacity: alpha }}
              alt="heatmap"
            />
            {bboxes.map((b, i) => (
              <div
                key={i}
                className="absolute border-2 border-cyan-300/90"
                style={{
                  left: `${(b.x / naturalWidth) * 100}%`,
                  top: `${(b.y / naturalHeight) * 100}%`,
                  width: `${(b.width / naturalWidth) * 100}%`,
                  height: `${(b.height / naturalHeight) * 100}%`,
                }}
                title={`score ${b.score.toFixed(3)}`}
              />
            ))}
          </div>
        </figure>
      </div>
      <label className="flex items-center gap-3 text-xs text-gray-400">
        Overlay opacity
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          className="flex-1 accent-cyan-400"
        />
        <span className="w-8 tabular-nums">{alpha.toFixed(2)}</span>
      </label>
    </div>
  );
}
