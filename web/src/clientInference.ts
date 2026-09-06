// Client-side inference for the static (GitHub Pages) demo: the ONNX model + its
// preprocess/metrics JSON are fetched from the Hugging Face model repo and run in
// the browser via onnxruntime-web (WASM). No backend.

import * as ort from "onnxruntime-web";
import type { BBox, ModelName, PredictResponse } from "./api";

const HF_REPO = "bhadra244131/mddf-artifacts";
const HF_BASE = `https://huggingface.co/${HF_REPO}/resolve/main`;

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
// Run inference in a Web Worker so a slow model never freezes the UI thread.
ort.env.wasm.proxy = true;
// Serve the WASM binaries (and the proxy worker) from the pinned CDN build.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

interface PreprocessSpec {
  image_size: [number, number];
  center_crop: [number, number] | null;
  normalize: boolean;
  mean: [number, number, number];
  std: [number, number, number];
  baked_into_onnx: boolean;
  score_is_normalized?: boolean;
  decision_threshold?: number;
}

interface Loaded {
  session: ort.InferenceSession;
  spec: PreprocessSpec;
  threshold: number;
  metrics: Record<string, number>;
  version: string;
}

const cache = new Map<string, Promise<Loaded>>();

async function fetchJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} fetching ${url}`);
  return (await r.json()) as T;
}

export function loadClientModel(model: ModelName, category: string): Promise<Loaded> {
  const key = `${model}/${category}`;
  let entry = cache.get(key);
  if (!entry) {
    entry = (async () => {
      const dir = `${HF_BASE}/${model}/${category}`;
      const [specDoc, metricsDoc, onnxBuf] = await Promise.all([
        fetchJSON<PreprocessSpec>(`${dir}/preprocess.json`),
        fetchJSON<Record<string, unknown>>(`${dir}/metrics.json`),
        fetch(`${dir}/model.int8.onnx`).then((r) => {
          if (!r.ok) throw new Error(`${r.status} fetching model.int8.onnx`);
          return r.arrayBuffer();
        }),
      ]);
      const session = await ort.InferenceSession.create(onnxBuf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      const thresholds = (metricsDoc.thresholds ?? {}) as Record<string, number>;
      const metrics = (metricsDoc.metrics ?? {}) as Record<string, number>;
      // Anomalib's exported PostProcessor normalises the score → boundary is 0.5.
      const threshold =
        specDoc.score_is_normalized === false && typeof thresholds.image === "number"
          ? thresholds.image
          : (specDoc.decision_threshold ?? 0.5);
      return { session, spec: specDoc, threshold, metrics, version: String(metricsDoc.git_sha ?? "demo") };
    })();
    cache.set(key, entry);
  }
  return entry;
}

function toTensor(img: HTMLImageElement, spec: PreprocessSpec): {
  data: Float32Array;
  w: number;
  h: number;
} {
  const [H, W] = spec.image_size;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, W, H);
  const { data } = ctx.getImageData(0, 0, W, H);

  const out = new Float32Array(3 * H * W);
  const norm = spec.normalize && !spec.baked_into_onnx;
  for (let i = 0; i < H * W; i++) {
    for (let c = 0; c < 3; c++) {
      let v = data[i * 4 + c] / 255;
      if (norm) v = (v - spec.mean[c]) / spec.std[c];
      out[c * H * W + i] = v;
    }
  }
  return { data: out, w: W, h: H };
}

function pickOutput(results: ort.InferenceSession.OnnxValueMapType, names: string[]): ort.Tensor | null {
  for (const n of names) {
    for (const key of Object.keys(results)) {
      if (key.toLowerCase() === n) return results[key] as ort.Tensor;
    }
  }
  return null;
}

function normalizeMap(map: Float32Array): Float32Array {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of map) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo || 1;
  const out = new Float32Array(map.length);
  for (let i = 0; i < map.length; i++) out[i] = (map[i] - lo) / span;
  return out;
}

// simple JET colormap
function jet(t: number): [number, number, number] {
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
  return [r * 255, g * 255, b * 255];
}

function renderOverlay(
  img: HTMLImageElement,
  map: Float32Array,
  mapH: number,
  mapW: number,
  alpha = 0.55,
): string {
  const W = img.naturalWidth;
  const H = img.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, W, H);
  const base = ctx.getImageData(0, 0, W, H);
  const norm = normalizeMap(map);
  for (let y = 0; y < H; y++) {
    const my = Math.min(mapH - 1, Math.floor((y / H) * mapH));
    for (let x = 0; x < W; x++) {
      const mx = Math.min(mapW - 1, Math.floor((x / W) * mapW));
      const [r, g, b] = jet(norm[my * mapW + mx]);
      const p = (y * W + x) * 4;
      base.data[p] = base.data[p] * (1 - alpha) + r * alpha;
      base.data[p + 1] = base.data[p + 1] * (1 - alpha) + g * alpha;
      base.data[p + 2] = base.data[p + 2] * (1 - alpha) + b * alpha;
    }
  }
  ctx.putImageData(base, 0, 0);
  return canvas.toDataURL("image/png").split(",")[1];
}

export async function predictClient(
  model: ModelName,
  category: string,
  file: File,
): Promise<PredictResponse> {
  const loaded = await loadClientModel(model, category);
  const img = await fileToImage(file);
  const started = performance.now();

  const { data, w, h } = toTensor(img, loaded.spec);
  const input = new ort.Tensor("float32", data, [1, 3, h, w]);
  const inputName = loaded.session.inputNames[0];
  const results = await loaded.session.run({ [inputName]: input });

  const mapTensor = pickOutput(results, ["anomaly_map", "anomaly_maps", "map"]);
  const scoreTensor = pickOutput(results, ["pred_score", "pred_scores", "score"]);
  if (!mapTensor) throw new Error("model has no anomaly_map output");

  const dims = mapTensor.dims;
  const mapH = dims[dims.length - 2];
  const mapW = dims[dims.length - 1];
  const map = mapTensor.data as Float32Array;

  let score = 0;
  if (scoreTensor) score = Number((scoreTensor.data as Float32Array)[0]);
  else for (const v of map) score = Math.max(score, v);

  const latency = performance.now() - started;
  const heatmap = renderOverlay(img, map, mapH, mapW);
  const verdict: "normal" | "defect" = score >= loaded.threshold ? "defect" : "normal";
  const z = (score - loaded.threshold) / (Math.abs(loaded.threshold) * 0.15 + 1e-6);
  const normalized = 1 / (1 + Math.exp(-z));

  const bboxes: BBox[] = [];
  return {
    category,
    model,
    model_version: loaded.version,
    verdict,
    anomaly_score: score,
    normalized_score: Math.max(0, Math.min(1, normalized)),
    threshold: loaded.threshold,
    latency_ms: Math.round(latency * 10) / 10,
    heatmap_png: heatmap,
    mask_png: heatmap,
    bboxes,
  };
}

function fileToImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = reject;
    img.src = url;
  });
}
