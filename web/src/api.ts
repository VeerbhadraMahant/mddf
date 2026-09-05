export type ModelName = "padim" | "patchcore" | "efficient_ad";

export interface CategoryMetrics {
  image_auroc: number | null;
  pixel_auroc: number | null;
  aupro: number | null;
  f1_max: number | null;
  published_image_auroc: number;
}

export interface CategoryInfo {
  name: string;
  kind: "object" | "texture";
  available_models: ModelName[];
  metrics: Partial<Record<ModelName, CategoryMetrics>>;
}

export interface CategoriesResponse {
  dataset: string;
  categories: CategoryInfo[];
}

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
}

export interface PredictResponse {
  category: string;
  model: ModelName;
  model_version: string;
  verdict: "normal" | "defect";
  anomaly_score: number;
  normalized_score: number;
  threshold: number;
  latency_ms: number;
  heatmap_png: string;
  mask_png: string;
  bboxes: BBox[];
}

export interface BenchmarkRow {
  category: string;
  model: ModelName;
  image_auroc: number;
  pixel_auroc: number | null;
  aupro: number | null;
  f1_max: number | null;
  published_image_auroc: number;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
}

export interface Problem {
  type: string;
  title: string;
  status: number;
  detail: string;
  request_id?: string;
}

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as Problem;
      detail = body.detail || body.title || detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

// Static (GitHub Pages) build: no FastAPI backend — inference runs in the browser
// and metadata comes from the Hugging Face model repo.
export const CLIENT_MODE = import.meta.env.VITE_INFERENCE === "client";
const HF_BASE = "https://huggingface.co/bhadra244131/mddf-artifacts/resolve/main";

async function clientCategories(): Promise<CategoriesResponse> {
  const { CATALOG } = await import("./catalog");
  let measured: Record<string, Record<string, CategoryMetrics>> = {};
  try {
    const doc = (await fetch(`${HF_BASE}/benchmark/metrics.json`).then((r) => r.json())) as {
      results: Record<string, Record<string, Record<string, number>>>;
    };
    for (const [cat, perModel] of Object.entries(doc.results ?? {})) {
      measured[cat] = {};
      for (const [m, v] of Object.entries(perModel)) {
        measured[cat][m] = {
          image_auroc: v.image_auroc ?? null,
          pixel_auroc: v.pixel_auroc ?? null,
          aupro: v.aupro ?? null,
          f1_max: v.f1_max ?? null,
          published_image_auroc:
            CATALOG.find((c) => c.name === cat)?.published_image_auroc ?? 1,
        };
      }
    }
  } catch {
    measured = {};
  }
  return {
    dataset: "mvtec_ad",
    categories: CATALOG.map((c) => ({
      name: c.name,
      kind: c.kind,
      available_models: ["padim"],
      metrics: (measured[c.name] ?? {}) as Partial<Record<ModelName, CategoryMetrics>>,
    })),
  };
}

async function clientBenchmark(): Promise<{ generated_at: string; rows: BenchmarkRow[] }> {
  const { CATALOG } = await import("./catalog");
  const doc = (await fetch(`${HF_BASE}/benchmark/metrics.json`).then((r) => r.json())) as {
    generated_at: string;
    results: Record<string, Record<string, Record<string, number>>>;
  };
  const rows: BenchmarkRow[] = [];
  for (const [cat, perModel] of Object.entries(doc.results ?? {})) {
    for (const [m, v] of Object.entries(perModel)) {
      rows.push({
        category: cat,
        model: m as ModelName,
        image_auroc: v.image_auroc ?? 0,
        pixel_auroc: v.pixel_auroc ?? null,
        aupro: v.aupro ?? null,
        f1_max: v.f1_max ?? null,
        published_image_auroc:
          CATALOG.find((c) => c.name === cat)?.published_image_auroc ?? 1,
        latency_ms_p50: v.latency_ms_p50 ?? null,
        latency_ms_p95: v.latency_ms_p95 ?? null,
      });
    }
  }
  return { generated_at: doc.generated_at, rows };
}

export const api = {
  categories: () =>
    CLIENT_MODE
      ? clientCategories()
      : fetch("/api/v1/categories").then((r) => unwrap<CategoriesResponse>(r)),
  benchmark: () =>
    CLIENT_MODE
      ? clientBenchmark()
      : fetch("/api/v1/benchmark").then((r) =>
          unwrap<{ generated_at: string; rows: BenchmarkRow[] }>(r),
        ),
  predict: async (file: File, category: string, model: ModelName) => {
    if (CLIENT_MODE) {
      const { predictClient } = await import("./clientInference");
      return predictClient(model, category, file);
    }
    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", category);
    fd.append("model", model);
    return fetch("/api/v1/predict", { method: "POST", body: fd }).then((r) =>
      unwrap<PredictResponse>(r),
    );
  },
};
