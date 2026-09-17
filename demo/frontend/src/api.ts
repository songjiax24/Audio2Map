import type { Range } from "./constants";

const BASE = "/api";

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map(String).join("; ");
  } catch {
    /* ignore */
  }
  return res.statusText || "Request failed";
}

export async function checkHealth(): Promise<{
  status: string;
  checkpoint_exists: boolean;
  manifest_exists: boolean;
}> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function uploadAudio(file: File): Promise<{ file_id: string; filename: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function selectCondition(
  ranges: Record<string, Range>,
  canonicalBpmNorm: number,
): Promise<{
  selected_chart_id: string;
  selected_osu_path: string;
  matched_candidate_count: number;
  selected_source_values: Record<string, number>;
  final_cond_vec: number[];
}> {
  const payload = {
    ranges: Object.fromEntries(
      Object.entries(ranges).map(([k, v]) => [k, { min: v.min, max: v.max }]),
    ),
    canonical_bpm_norm: canonicalBpmNorm,
  };
  const res = await fetch(`${BASE}/select_condition`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchCondNames(): Promise<string[]> {
  const res = await fetch(`${BASE}/cond-names`);
  if (!res.ok) throw new Error(await parseError(res));
  const data = await res.json();
  const names = data.user_cond_names;
  if (!Array.isArray(names) || names.length === 0) {
    throw new Error("Backend returned no condition names");
  }
  return names.map(String);
}

export async function generateBeatmap(body: {
  file_id: string;
  bpm: number;
  offset_ms: number;
  metadata: {
    title: string;
    artist: string;
    creator: string;
    difficulty_name: string;
  };
  final_cond_vec: number[];
  selected_condition?: Record<string, unknown>;
}): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchJobStatus(jobId: string): Promise<{
  job_id: string;
  status: string;
  step: string | null;
  error: string | null;
  windows_done: number;
  windows_total: number;
  note_count?: number | null;
  osu_filename?: string | null;
  osz_filename?: string | null;
}> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export function downloadUrl(jobId: string, kind: "osu" | "osz"): string {
  return `${BASE}/download/${jobId}/${kind}`;
}
