export const COND_NAMES = [
  "official_sr",
  "analyzer_ln_percent",
  "analyzer_hb_row_ratio",
  "analyzer_stream",
  "analyzer_chordstream",
  "analyzer_jacks",
  "analyzer_coordination",
  "analyzer_density",
  "analyzer_wildcard",
  "msd_overall",
  "msd_stream",
  "msd_jumpstream",
  "msd_handstream",
  "msd_stamina",
  "msd_jack_speed",
  "msd_chordjack",
  "msd_technical",
] as const;

export type CondName = (typeof COND_NAMES)[number];

export type Range = { min: number; max: number };

export function sliderBounds(name: string): { min: number; max: number; step: number } {
  if (name === "official_sr") return { min: 0, max: 15, step: 0.1 };
  if (name.startsWith("msd_")) return { min: 0, max: 60, step: 0.1 };
  return { min: 0, max: 1, step: 0.005 };
}

export function formatCondValue(name: string, value: number): string {
  return sliderBounds(name).max > 1 ? value.toFixed(1) : value.toFixed(2);
}

export function formatCondLabel(name: string): string {
  if (name === "official_sr") return "osu star rating";
  return name.replace(/_/g, " ");
}

export function defaultRange(name: string): Range {
  if (name === "official_sr") return { min: 0, max: 10 };
  if (name.startsWith("msd_")) return { min: 0, max: 40 };
  return { min: 0, max: 1 };
}

export function rangesFromNames(names: readonly string[]): Record<string, Range> {
  return Object.fromEntries(names.map((name) => [name, defaultRange(name)]));
}

export function canonicalFromBpm(bpm: number): { canonicalBpm: number; canonicalBpmNorm: number } {
  let c = bpm;
  while (c < 120) c *= 2;
  while (c >= 240) c /= 2;
  return { canonicalBpm: c, canonicalBpmNorm: (c - 120) / 120 };
}
