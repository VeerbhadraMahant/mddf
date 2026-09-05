# mddf-web

React + Vite + TypeScript + Tailwind SPA for the Manufacturing Defect Detection service.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api -> 127.0.0.1:7860
npm run build    # -> web/dist, served by the FastAPI app at "/"
```

- **Inspect** tab: pick a category + model, drop an image, get verdict + score
  gauge + heatmap overlay (opacity slider) + defect bounding boxes + latency.
- **Benchmark** tab: measured image AUROC vs. the published PatchCore baseline,
  pixel AUROC, AUPRO and CPU p50 latency, from `GET /api/v1/benchmark`.
