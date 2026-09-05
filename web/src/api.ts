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

export const api = {
  categories: () => fetch("/api/v1/categories").then((r) => unwrap<CategoriesResponse>(r)),
  benchmark: () =>
    fetch("/api/v1/benchmark").then((r) => unwrap<{ generated_at: string; rows: BenchmarkRow[] }>(r)),
  predict: (file: File, category: string, model: ModelName) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", category);
    fd.append("model", model);
    return fetch("/api/v1/predict", { method: "POST", body: fd }).then((r) =>
      unwrap<PredictResponse>(r),
    );
  },
};
