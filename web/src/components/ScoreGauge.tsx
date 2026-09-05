interface Props {
  normalized: number; // 0..1, 0.5 == decision threshold
  rawScore: number;
  threshold: number;
  verdict: "normal" | "defect";
}

export function ScoreGauge({ normalized, rawScore, threshold, verdict }: Props) {
  const pct = Math.max(0, Math.min(1, normalized)) * 100;
  const defect = verdict === "defect";
  return (
    <div className="w-full">
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-medium text-gray-300">Anomaly score</span>
        <span className={defect ? "text-rose-400 font-semibold" : "text-emerald-400 font-semibold"}>
          {verdict.toUpperCase()}
        </span>
      </div>
      <div className="relative mt-2 h-3 rounded-full bg-gray-800">
        <div
          className={`h-3 rounded-full ${defect ? "bg-rose-500" : "bg-emerald-500"}`}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute -top-1 h-5 w-0.5 bg-gray-200"
          style={{ left: "50%" }}
          title="decision threshold"
        />
      </div>
      <div className="mt-1 flex justify-between text-xs text-gray-500">
        <span>raw {rawScore.toFixed(3)}</span>
        <span>threshold {threshold.toFixed(3)}</span>
      </div>
    </div>
  );
}
