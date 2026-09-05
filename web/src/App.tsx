import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type CategoryInfo,
  type ModelName,
  type PredictResponse,
} from "./api";
import { ScoreGauge } from "./components/ScoreGauge";
import { HeatmapView } from "./components/HeatmapView";
import { BenchmarkTable } from "./components/BenchmarkTable";
import { Dashboard } from "./components/Dashboard";

type Tab = "inspect" | "dashboard" | "benchmark";

export function App() {
  const [tab, setTab] = useState<Tab>("inspect");
  const [categories, setCategories] = useState<CategoryInfo[]>([]);
  const [category, setCategory] = useState("");
  const [model, setModel] = useState<ModelName>("patchcore");
  const [file, setFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number }>({ w: 1, h: 1 });
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .categories()
      .then((r) => {
        setCategories(r.categories);
        const first = r.categories.find((c) => c.available_models.length) ?? r.categories[0];
        if (first) {
          setCategory(first.name);
          setModel(first.available_models[0] ?? "patchcore");
        }
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const activeCategory = useMemo(
    () => categories.find((c) => c.name === category),
    [categories, category],
  );
  const availableModels: ModelName[] = activeCategory?.available_models.length
    ? activeCategory.available_models
    : ["patchcore", "efficient_ad"];

  const pickFile = useCallback((f: File | null) => {
    setResult(null);
    setError(null);
    setFile(f);
    setObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return f ? URL.createObjectURL(f) : null;
    });
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) pickFile(f);
  };

  const run = async () => {
    if (!file || !category) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.predict(file, category, model));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto min-h-screen max-w-5xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-100">Manufacturing Defect Detection</h1>
        <p className="mt-1 text-sm text-gray-400">
          Unsupervised visual inspection on MVTec AD — PatchCore &amp; EfficientAD, trained on
          defect-free images only, served CPU-only.
        </p>
        <nav className="mt-4 flex gap-1 rounded-lg bg-gray-900 p-1 text-sm">
          {(["inspect", "dashboard", "benchmark"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-md px-3 py-1.5 capitalize ${
                tab === t ? "bg-gray-700 text-white" : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {error && (
        <div className="mb-4 rounded-lg border border-rose-800 bg-rose-950/50 px-4 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}

      {tab === "dashboard" ? (
        <Dashboard />
      ) : tab === "benchmark" ? (
        <section className="rounded-xl border border-gray-800 bg-gray-900/40 p-4">
          <BenchmarkTable />
        </section>
      ) : (
        <section className="grid gap-6 md:grid-cols-[280px_1fr]">
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-xs uppercase tracking-wide text-gray-500">
                Category
              </label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm"
              >
                {categories.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name} ({c.kind})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs uppercase tracking-wide text-gray-500">
                Model
              </label>
              <div className="flex gap-1 rounded-lg bg-gray-900 p-1">
                {availableModels.map((m) => (
                  <button
                    key={m}
                    onClick={() => setModel(m)}
                    className={`flex-1 rounded-md px-2 py-1.5 text-xs ${
                      model === m ? "bg-gray-700 text-white" : "text-gray-400"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-gray-700 bg-gray-900/40 px-4 py-8 text-center text-sm text-gray-400 hover:border-gray-500"
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
              />
              {file ? (
                <span className="text-gray-200">{file.name}</span>
              ) : (
                <>
                  <span>Drop an image here</span>
                  <span className="mt-1 text-xs text-gray-600">or click to browse</span>
                </>
              )}
            </div>

            <button
              onClick={run}
              disabled={!file || busy}
              className="w-full rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
            >
              {busy ? "Analyzing…" : "Detect defects"}
            </button>
          </div>

          <div className="min-h-[320px] rounded-xl border border-gray-800 bg-gray-900/40 p-4">
            {!objectUrl && <p className="text-sm text-gray-500">Upload an image to begin.</p>}

            {objectUrl && !result && (
              <img
                src={objectUrl}
                onLoad={(e) =>
                  setNatural({
                    w: e.currentTarget.naturalWidth,
                    h: e.currentTarget.naturalHeight,
                  })
                }
                className="max-h-80 rounded-lg border border-gray-800"
                alt="preview"
              />
            )}

            {result && objectUrl && (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400">
                  <span className="rounded bg-gray-800 px-2 py-1">{result.model}</span>
                  <span className="rounded bg-gray-800 px-2 py-1">{result.category}</span>
                  <span className="rounded bg-gray-800 px-2 py-1">
                    {result.latency_ms.toFixed(0)} ms
                  </span>
                  <span className="rounded bg-gray-800 px-2 py-1">v{result.model_version}</span>
                </div>
                <ScoreGauge
                  normalized={result.normalized_score}
                  rawScore={result.anomaly_score}
                  threshold={result.threshold}
                  verdict={result.verdict}
                />
                <HeatmapView
                  original={objectUrl}
                  heatmapPng={result.heatmap_png}
                  bboxes={result.bboxes}
                  naturalWidth={natural.w}
                  naturalHeight={natural.h}
                />
              </div>
            )}
          </div>
        </section>
      )}

      <footer className="mt-10 text-center text-xs text-gray-600">
        <a href="/api/docs" className="hover:text-gray-400">
          API docs
        </a>
        {" · "}
        <a
          href="https://github.com/VeerbhadraMahant/mddf"
          className="hover:text-gray-400"
        >
          source
        </a>
      </footer>
    </div>
  );
}
