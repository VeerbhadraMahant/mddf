// Static mirror of mddf/resources/categories.yaml for the client-only demo build.
export interface StaticCategory {
  name: string;
  kind: "object" | "texture";
  published_image_auroc: number;
}

export const CATALOG: StaticCategory[] = [
  { name: "bottle", kind: "object", published_image_auroc: 1.0 },
  { name: "cable", kind: "object", published_image_auroc: 0.995 },
  { name: "capsule", kind: "object", published_image_auroc: 0.98 },
  { name: "carpet", kind: "texture", published_image_auroc: 0.987 },
  { name: "grid", kind: "texture", published_image_auroc: 0.981 },
  { name: "hazelnut", kind: "object", published_image_auroc: 1.0 },
  { name: "leather", kind: "texture", published_image_auroc: 1.0 },
  { name: "metal_nut", kind: "object", published_image_auroc: 0.996 },
  { name: "pill", kind: "object", published_image_auroc: 0.979 },
  { name: "screw", kind: "object", published_image_auroc: 0.987 },
  { name: "tile", kind: "texture", published_image_auroc: 0.994 },
  { name: "toothbrush", kind: "object", published_image_auroc: 1.0 },
  { name: "transistor", kind: "object", published_image_auroc: 1.0 },
  { name: "wood", kind: "texture", published_image_auroc: 0.992 },
  { name: "zipper", kind: "object", published_image_auroc: 0.985 },
];
