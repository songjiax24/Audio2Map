import { useCallback, useEffect, useState } from "react";
import {
  checkHealth,
  downloadUrl,
  estimateTiming,
  fetchCondNames,
  fetchJobStatus,
  generateBeatmap,
  selectCondition,
  uploadAudio,
} from "./api";
import { CondRangeSlider } from "./CondRangeSlider";
import {
  COND_NAMES,
  canonicalFromBpm,
  rangesFromNames,
  type Range,
} from "./constants";

type Step =
  | "idle"
  | "uploading"
  | "timing"
  | "condition"
  | "inference"
  | "export"
  | "done";

export default function App() {
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [healthMsg, setHealthMsg] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");

  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("");
  const [creator, setCreator] = useState("Audio2Map");
  const [difficultyName, setDifficultyName] = useState("Generated");

  const [condNames, setCondNames] = useState<string[]>([...COND_NAMES]);
  const [ranges, setRanges] = useState<Record<string, Range>>(() => rangesFromNames(COND_NAMES));
  const [conditionResult, setConditionResult] = useState<{
    matched: number;
    values: Record<string, number>;
    finalCondVec: number[];
    payload: Record<string, unknown>;
  } | null>(null);
  const [conditionError, setConditionError] = useState("");

  const [useAutoTiming, setUseAutoTiming] = useState(true);
  const [bpm, setBpm] = useState<number | "">("");
  const [offsetMs, setOffsetMs] = useState<number | "">("");
  const [canonicalBpmNorm, setCanonicalBpmNorm] = useState<number | "">("");
  const [timingError, setTimingError] = useState("");

  const [step, setStep] = useState<Step>("idle");
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [noteCount, setNoteCount] = useState<number | null>(null);
  const [osuFilename, setOsuFilename] = useState("generated.osu");
  const [oszFilename, setOszFilename] = useState("generated.osz");
  const [busy, setBusy] = useState(false);
  const [genProgress, setGenProgress] = useState<{
    done: number;
    total: number;
    step: string | null;
  } | null>(null);

  useEffect(() => {
    checkHealth()
      .then((h) => {
        const ok = h.checkpoint_exists && h.manifest_exists;
        setHealthOk(ok);
        if (!h.checkpoint_exists) {
          setHealthMsg("Checkpoint not found.");
        } else if (!h.manifest_exists) {
          setHealthMsg("Chart library not found.");
        } else {
          setHealthMsg("Backend ready.");
        }
      })
      .catch(() => {
        setHealthOk(false);
        setHealthMsg("Cannot reach backend. Start uvicorn on port 8000.");
      });
    fetchCondNames()
      .then((names) => {
        setCondNames(names);
        setRanges(rangesFromNames(names));
      })
      .catch(() => {
        /* keep bundled COND_NAMES */
      });
  }, []);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const ext = f.name.toLowerCase();
    if (!ext.endsWith(".mp3") && !ext.endsWith(".wav")) {
      setError("Unsupported audio format. Please upload .mp3 or .wav.");
      return;
    }
    setFile(f);
    setError("");
    setUploadStatus("Uploading…");
    setStep("uploading");
    setConditionResult(null);
    setJobId(null);
    try {
      const res = await uploadAudio(f);
      setFileId(res.file_id);
      setUploadStatus(`Uploaded: ${res.filename}`);
      setStep("idle");
    } catch (err) {
      setUploadStatus("");
      setError(err instanceof Error ? err.message : "Upload failed");
      setStep("idle");
    }
  }, []);

  const handleEstimateTiming = useCallback(async () => {
    if (!fileId) {
      setTimingError("Please upload an .mp3 or .wav file first.");
      return;
    }
    setTimingError("");
    setBusy(true);
    setStep("timing");
    try {
      const res = await estimateTiming(fileId);
      setBpm(Math.round(res.bpm * 100) / 100);
      setOffsetMs(Math.round(res.offset_ms * 10) / 10);
      setCanonicalBpmNorm(Math.round(res.canonical_bpm_norm * 10000) / 10000);
      setUseAutoTiming(true);
      setStep("idle");
    } catch (err) {
      setTimingError(
        err instanceof Error
          ? err.message
          : "Could not estimate beats from the audio. Please enter BPM and offset manually.",
      );
      setStep("idle");
    } finally {
      setBusy(false);
    }
  }, [fileId]);

  useEffect(() => {
    if (!useAutoTiming && typeof bpm === "number" && bpm > 0) {
      const { canonicalBpmNorm: n } = canonicalFromBpm(bpm);
      setCanonicalBpmNorm(Math.round(n * 10000) / 10000);
      setConditionResult(null);
    }
  }, [bpm, useAutoTiming]);

  const handleSearchCondition = useCallback(async () => {
    if (canonicalBpmNorm === "") {
      setConditionError("Estimate BPM first (or enter timing manually).");
      return;
    }
    setConditionError("");
    setBusy(true);
    setStep("condition");
    try {
      const res = await selectCondition(ranges, Number(canonicalBpmNorm));
      setConditionResult({
        matched: res.matched_candidate_count,
        values: res.selected_source_values ?? {},
        finalCondVec: res.final_cond_vec,
        payload: res as unknown as Record<string, unknown>,
      });
      setStep("idle");
    } catch (err) {
      setConditionResult(null);
      setConditionError(
        err instanceof Error
          ? err.message
          : "No charts match these ranges. Widen them and search again.",
      );
      setStep("idle");
    } finally {
      setBusy(false);
    }
  }, [ranges, canonicalBpmNorm]);

  const handleGenerate = useCallback(async () => {
    if (!fileId) {
      setError("Please upload an .mp3 or .wav file.");
      return;
    }
    if (bpm === "" || offsetMs === "") {
      setError("Please estimate or enter BPM and offset.");
      return;
    }
    if (!conditionResult?.finalCondVec) {
      setError("Search for a matching style first.");
      return;
    }

    setError("");
    setBusy(true);
    setJobId(null);
    setNoteCount(null);
    setOsuFilename("generated.osu");
    setOszFilename("generated.osz");
    setGenProgress({ done: 0, total: 0, step: "setup" });
    setStep("inference");

    try {
      const started = await generateBeatmap({
        file_id: fileId,
        bpm: Number(bpm),
        offset_ms: Number(offsetMs),
        metadata: {
          title: title || "Untitled",
          artist: artist || "Unknown Artist",
          creator: creator || "Audio2Map",
          difficulty_name: difficultyName || "Generated",
        },
        final_cond_vec: conditionResult.finalCondVec,
        selected_condition: conditionResult.payload,
      });
      setJobId(started.job_id);

      for (;;) {
        const job = await fetchJobStatus(started.job_id);
        setGenProgress({
          done: job.windows_done ?? 0,
          total: job.windows_total ?? 0,
          step: job.step,
        });
        if (job.status === "completed") {
          setNoteCount(job.note_count ?? null);
          setOsuFilename(job.osu_filename || "generated.osu");
          setOszFilename(job.osz_filename || "generated.osz");
          setGenProgress(null);
          setStep("done");
          break;
        }
        if (job.status === "failed") {
          throw new Error("Generation failed. See backend logs for details.");
        }
        await new Promise((resolve) => setTimeout(resolve, 400));
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Generation failed. See backend logs for details.",
      );
      setStep("idle");
      setGenProgress(null);
    } finally {
      setBusy(false);
    }
  }, [
    fileId,
    bpm,
    offsetMs,
    title,
    artist,
    creator,
    difficultyName,
    conditionResult,
  ]);

  const updateRange = (name: string, next: Range) => {
    setRanges((prev) => ({ ...prev, [name]: next }));
    setConditionResult(null);
  };

  const steps: { key: Step; label: string }[] = [
    { key: "uploading", label: "Uploading audio" },
    { key: "timing", label: "Estimating BPM and offset" },
    { key: "condition", label: "Finding a matching chart style" },
    { key: "inference", label: "Generating chart" },
    { key: "export", label: "Exporting .osu / .osz" },
    { key: "done", label: "Done" },
  ];

  const stepOrder = steps.map((s) => s.key);
  const currentIdx = stepOrder.indexOf(step);
  const genIndeterminate =
    genProgress != null &&
    genProgress.total === 0 &&
    genProgress.step !== "export" &&
    genProgress.step !== "done";
  const genPct =
    genProgress == null
      ? 0
      : genProgress.step === "export" || genProgress.step === "done"
        ? 100
        : genProgress.total > 0
          ? Math.round((100 * genProgress.done) / genProgress.total)
          : 0;
  const genLabel = genIndeterminate
    ? "Preparing audio…"
    : genProgress?.step === "export"
      ? "Saving beatmap…"
      : `Generating chart… ${genPct}%`;

  return (
    <>
      <h1>Audio2Map Demo</h1>
      <p className="subtitle">Generate osu!mania 4K charts from audio</p>

      {healthOk !== null && (
        <div className={`health-banner ${healthOk ? "ok" : "warn"}`}>{healthMsg}</div>
      )}

      <section className="card">
        <h2>Upload Audio</h2>
        <input type="file" accept=".mp3,.wav,audio/mpeg,audio/wav" onChange={handleFileChange} />
        {file && <p className="status info">Selected: {file.name}</p>}
        {uploadStatus && <p className="status ok">{uploadStatus}</p>}
      </section>

      <section className="card">
        <h2>Beatmap Metadata</h2>
        <label>
          Title
          <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Song title" />
        </label>
        <label>
          Artist / Author
          <input type="text" value={artist} onChange={(e) => setArtist(e.target.value)} placeholder="Artist name" />
        </label>
        <div className="row-2">
          <label>
            Creator
            <input type="text" value={creator} onChange={(e) => setCreator(e.target.value)} />
          </label>
          <label>
            Difficulty Name
            <input
              type="text"
              value={difficultyName}
              onChange={(e) => setDifficultyName(e.target.value)}
            />
          </label>
        </div>
      </section>

      <section className="card">
        <h2>Timing</h2>
        <button type="button" onClick={handleEstimateTiming} disabled={busy || !fileId}>
          Estimate BPM / Offset
        </button>
        {timingError && <div className="status error">{timingError}</div>}

        <div className="checkbox-row">
          <input
            type="checkbox"
            id="auto-timing"
            checked={useAutoTiming}
            onChange={(e) => setUseAutoTiming(e.target.checked)}
          />
          <label htmlFor="auto-timing" style={{ margin: 0 }}>
            Use auto-estimated timing
          </label>
        </div>

        <div className="row-2">
          <label>
            BPM
            <input
              type="number"
              value={bpm}
              disabled={useAutoTiming}
              onChange={(e) => setBpm(e.target.value === "" ? "" : Number(e.target.value))}
            />
          </label>
          <label>
            Offset (ms)
            <input
              type="number"
              value={offsetMs}
              disabled={useAutoTiming}
              onChange={(e) => setOffsetMs(e.target.value === "" ? "" : Number(e.target.value))}
            />
          </label>
        </div>

        {!useAutoTiming && (
          <p style={{ fontSize: "0.8rem", color: "var(--muted)" }}>
            Edit BPM and offset above.
          </p>
        )}
      </section>

      <section className="card">
        <h2>Style</h2>
        {condNames.map((name) => (
          <CondRangeSlider
            key={name}
            name={name}
            value={ranges[name] ?? { min: 0, max: 1 }}
            selected={conditionResult?.values[name]}
            onChange={(next) => updateRange(name, next)}
          />
        ))}
        <button type="button" className="secondary" onClick={handleSearchCondition} disabled={busy}>
          Find matching charts
        </button>
        {conditionError && <div className="status error">{conditionError}</div>}
        {conditionResult && (
          <div className="status ok">Found {conditionResult.matched} matching charts.</div>
        )}
      </section>

      <section className="card">
        <h2>Generate</h2>
        <button type="button" className="large" onClick={handleGenerate} disabled={busy || healthOk === false}>
          Generate Beatmap
        </button>

        {genProgress && (
          <div className="gen-progress">
            <div
              className={genIndeterminate ? "progress-bar indeterminate" : "progress-bar"}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={genIndeterminate ? undefined : genPct}
            >
              <div
                className="progress-bar-fill"
                style={genIndeterminate ? undefined : { width: `${genPct}%` }}
              />
            </div>
            <div className="progress-label">{genLabel}</div>
          </div>
        )}

        {step !== "idle" && !genProgress && step !== "done" && (
          <ul className="progress-steps">
            {steps.map((s, i) => {
              let cls = "";
              if (i < currentIdx) cls = "done";
              else if (i === currentIdx) cls = "active";
              return (
                <li key={s.key} className={cls}>
                  {i + 1}. {s.label}
                </li>
              );
            })}
          </ul>
        )}

        {error && <div className="status error">{error}</div>}

        {jobId && step === "done" && (
          <div className="status ok">
            <div>Generation complete</div>
            {noteCount !== null && <div>Notes: {noteCount}</div>}
            <div className="download-links">
              <a href={downloadUrl(jobId, "osu")} download={osuFilename}>
                Download {osuFilename}
              </a>
              <a href={downloadUrl(jobId, "osz")} download={oszFilename}>
                Download {oszFilename}
              </a>
            </div>
          </div>
        )}
      </section>
    </>
  );
}
