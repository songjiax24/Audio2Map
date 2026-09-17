"""Run Audio2Map checkpoint inference for demo jobs."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audio2map.features.cond import COND_VEC_DIM
from audio2map.generate.overlap import GenerationRangeConfig, OverlapConfig
from audio2map.generate.service import (
    DecodeConfig,
    canonical_bpm_norm_from_bpm,
    estimate_bpm_offset,
    generate_chart,
    load_cond_candidates,
    select_condition,
    timing_from_bpm_offset,
)
from audio2map.model.model import load_checkpoint
from audio2map.osu.export import osu_export_names, sanitize_audio_filename
from audio2map.utils.paths import audio_grid_dir
from demo.backend.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    OUTPUT_DIR,
    UPLOAD_DIR,
    get_checkpoint_path,
    resolve_device_name,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class JobState:
    job_id: str
    status: str = "pending"
    step: str | None = None
    error: str | None = None
    windows_done: int = 0
    windows_total: int = 0
    note_count: int | None = None
    osu_filename: str | None = None
    osz_filename: str | None = None


class InferenceService:
    def __init__(self) -> None:
        self._model = None
        self._device: torch.device | None = None
        self._candidates = None
        self._candidates_key: tuple[str, float] | None = None
        self._jobs: dict[str, JobState] = {}
        self._generate_lock = threading.Lock()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        audio_grid_dir().mkdir(parents=True, exist_ok=True)

    def _torch_device(self) -> torch.device:
        name = resolve_device_name()
        if name == "cpu":
            return torch.device("cpu")
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _ensure_model(self) -> tuple[Any, torch.device]:
        if self._model is None:
            ckpt = get_checkpoint_path()
            if not ckpt.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            device = self._torch_device()
            log.info("loading checkpoint %s on %s", ckpt, device)
            self._model = load_checkpoint(ckpt, device)
            self._device = device
        return self._model, self._device  # type: ignore[return-value]

    def get_candidates(self, manifest_path: Path):
        path = manifest_path.resolve()
        mtime = path.stat().st_mtime if path.is_file() else -1.0
        key = (str(path), mtime)
        if self._candidates is None or self._candidates_key != key:
            self._candidates = load_cond_candidates(path)
            self._candidates_key = key
            log.info("loaded %d cond candidates from %s", len(self._candidates), path)
        return self._candidates

    def save_upload(self, filename: str, data: bytes) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")
        file_id = uuid.uuid4().hex
        folder = UPLOAD_DIR / file_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / sanitize_audio_filename(filename, default_suffix=ext)).write_bytes(data)
        return file_id

    def _upload_path(self, file_id: str) -> Path:
        folder = UPLOAD_DIR / file_id
        if not folder.is_dir():
            raise FileNotFoundError(f"Unknown file_id: {file_id}")
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS]
        if len(files) != 1:
            raise FileNotFoundError(f"No uploaded audio for file_id: {file_id}")
        return files[0]

    def estimate_timing(self, file_id: str) -> dict[str, Any]:
        audio_path = self._upload_path(file_id)
        bpm, offset_ms, debug = estimate_bpm_offset(audio_path)
        canonical_bpm, canonical_bpm_norm, scale_exp = canonical_bpm_norm_from_bpm(bpm)
        debug.setdefault("canonical_bpm", canonical_bpm)
        debug.setdefault("canonical_bpm_norm", canonical_bpm_norm)
        debug.setdefault("bpm_scale_exp", scale_exp)
        return {
            "bpm": bpm,
            "offset_ms": offset_ms,
            "canonical_bpm": canonical_bpm,
            "canonical_bpm_norm": canonical_bpm_norm,
            "debug": debug,
        }

    def select_condition_from_ranges(
        self,
        manifest_path: Path,
        ranges: dict[str, tuple[float, float]],
        canonical_bpm_norm: float,
    ) -> dict[str, Any]:
        candidates = self.get_candidates(manifest_path)
        return select_condition(candidates, ranges, canonical_bpm_norm)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def generate(
        self,
        *,
        file_id: str,
        bpm: float,
        offset_ms: float,
        metadata: dict[str, str],
        final_cond_vec: list[float],
        manifest_path: Path,
        condition_payload: dict[str, Any] | None = None,
        request_payload: dict[str, Any] | None = None,
    ) -> str:
        uploaded = self._upload_path(file_id)
        names = osu_export_names(
            artist=metadata.get("artist") or "Unknown Artist",
            title=metadata.get("title") or "Untitled",
            creator=metadata.get("creator") or "Audio2Map",
            version=metadata.get("difficulty_name") or "Generated",
            audio_filename=uploaded.name,
        )
        job_id = uuid.uuid4().hex
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        audio_path = job_dir / names.audio
        shutil.copy2(uploaded, audio_path)
        self._write_json(
            job_dir / "export_files.json",
            {"osu": names.osu, "osz": names.osz, "audio": names.audio},
        )
        if request_payload:
            self._write_json(job_dir / "request.json", request_payload)
        if condition_payload:
            self._write_json(job_dir / "selected_condition.json", condition_payload)
        canon_bpm, canon_norm, _ = canonical_bpm_norm_from_bpm(bpm)
        self._write_json(
            job_dir / "estimated_timing.json",
            {
                "bpm": bpm,
                "offset_ms": offset_ms,
                "canonical_bpm": canon_bpm,
                "canonical_bpm_norm": canon_norm,
            },
        )
        self._write_json(job_dir / "final_cond_vec.json", {"cond_vec": final_cond_vec})
        self._jobs[job_id] = JobState(
            job_id=job_id,
            status="running",
            step="setup",
            osu_filename=names.osu,
            osz_filename=names.osz,
        )
        thread = threading.Thread(
            target=self._run_generate,
            kwargs={
                "job_id": job_id,
                "job_dir": job_dir,
                "audio_path": audio_path,
                "names": names,
                "bpm": bpm,
                "offset_ms": offset_ms,
                "metadata": metadata,
                "final_cond_vec": final_cond_vec,
            },
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_generate(
        self,
        *,
        job_id: str,
        job_dir: Path,
        audio_path: Path,
        names: Any,
        bpm: float,
        offset_ms: float,
        metadata: dict[str, str],
        final_cond_vec: list[float],
    ) -> None:
        state = self._jobs[job_id]
        try:
            with self._generate_lock:
                timing = timing_from_bpm_offset(bpm=float(bpm), offset_ms=offset_ms)
                state.step = "inference"
                model, device = self._ensure_model()
                cond = np.array(final_cond_vec, dtype=np.float32)
                if cond.shape != (COND_VEC_DIM,):
                    raise ValueError(f"expected cond_vec length {COND_VEC_DIM}, got {cond.shape}")

                def on_progress(done: int, total: int) -> None:
                    state.windows_done = done
                    state.windows_total = total

                out_osu = job_dir / names.osu
                result = generate_chart(
                    model,
                    audio=audio_path,
                    timing=timing,
                    cond_vec=cond,
                    grid_dir=audio_grid_dir(),
                    overlap=OverlapConfig(),
                    range_cfg=GenerationRangeConfig(mode="audio_full"),
                    device=device,
                    decode=DecodeConfig(),
                    out_osu=out_osu,
                    export_version_suffix="",
                    export_title=metadata.get("title") or "Untitled",
                    export_artist=metadata.get("artist") or "Unknown Artist",
                    export_creator=metadata.get("creator") or "Audio2Map",
                    export_version=metadata.get("difficulty_name") or "Generated",
                    export_audio_filename=names.audio,
                    osz_audio=audio_path,
                    out_osz=job_dir / names.osz,
                    on_progress=on_progress,
                )
                notes, report = result.notes, result.report
                state.step = "export"
                self._write_json(
                    job_dir / "generation_report.json",
                    {
                        "note_count": len(notes),
                        "generation": report.to_dict(),
                        "checkpoint": str(get_checkpoint_path()),
                    },
                )
                state.note_count = len(notes)
                state.status = "completed"
                state.step = "done"
        except Exception as exc:
            log.exception("generation failed for job %s", job_id)
            state.status = "failed"
            state.error = str(exc)
            (job_dir / "error.txt").write_text(str(exc), encoding="utf-8")

    def job_status(self, job_id: str) -> dict[str, Any]:
        job_dir = OUTPUT_DIR / job_id
        if not job_dir.is_dir():
            raise FileNotFoundError(f"Unknown job_id: {job_id}")

        state = self._jobs.get(job_id, JobState(job_id=job_id, status="unknown"))
        out: dict[str, Any] = {
            "job_id": job_id,
            "status": state.status,
            "step": state.step,
            "error": state.error,
            "windows_done": state.windows_done,
            "windows_total": state.windows_total,
        }

        sel_path = job_dir / "selected_condition.json"
        if sel_path.is_file():
            out["selected_condition"] = json.loads(sel_path.read_text(encoding="utf-8"))

        rep_path = job_dir / "generation_report.json"
        if rep_path.is_file():
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            out["generation_report"] = rep
            out["note_count"] = rep.get("note_count")
        elif state.note_count is not None:
            out["note_count"] = state.note_count

        files = self._export_files(job_dir)
        if files.get("osu"):
            out["osu_filename"] = files["osu"]
        if files.get("osz"):
            out["osz_filename"] = files["osz"]

        osu_path = job_dir / files["osu"] if files.get("osu") else job_dir / "generated.osu"
        if state.status == "unknown" and osu_path.is_file():
            out["status"] = "completed"

        err_path = job_dir / "error.txt"
        if err_path.is_file() and out["status"] != "completed":
            out["status"] = "failed"
            out["error"] = err_path.read_text(encoding="utf-8").strip()

        return out

    def _export_files(self, job_dir: Path) -> dict[str, str]:
        path = job_dir / "export_files.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {key: str(data[key]) for key in ("osu", "osz", "audio") if key in data}
        if (job_dir / "generated.osu").is_file():
            return {"osu": "generated.osu", "osz": "generated.osz"}
        return {}

    def download_path(self, job_id: str, kind: str) -> Path:
        if kind not in {"osu", "osz"}:
            raise ValueError(f"unknown download kind: {kind}")
        job_dir = OUTPUT_DIR / job_id
        files = self._export_files(job_dir)
        name = files.get(kind)
        if not name:
            raise FileNotFoundError(f"{kind} not ready for job {job_id}")
        path = job_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{kind} not ready for job {job_id}")
        return path


service = InferenceService()
