"""FastAPI backend for Audio2Map web demo."""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from audio2map.features.cond import COND_VEC_DIM, COND_VEC_NAMES
from audio2map.generate.service import CondSelectionError, USER_COND_SOURCE_FIELDS
from demo.backend.config import get_checkpoint_path, get_manifest_path
from demo.backend.inference_service import service
from demo.backend.schemas import (
    ConditionRange,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatusResponse,
    SelectConditionRequest,
    SelectConditionResponse,
    UploadResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Audio2Map Demo", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ckpt = get_checkpoint_path()
    manifest = get_manifest_path()
    return HealthResponse(
        status="ok",
        checkpoint_exists=ckpt.is_file(),
        manifest_exists=manifest.is_file(),
        cond_vec_dim=COND_VEC_DIM,
    )


@app.get("/api/cond-names")
def cond_names() -> dict:
    return {
        "user_cond_names": list(USER_COND_SOURCE_FIELDS),
        "all_cond_names": list(COND_VEC_NAMES),
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(400, "Please upload an .mp3 or .wav file.")
    try:
        data = await file.read()
        file_id = service.save_upload(file.filename, data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return UploadResponse(file_id=file_id, filename=file.filename)


def _parse_condition_ranges(raw: dict[str, ConditionRange]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for name, span in raw.items():
        if name not in USER_COND_SOURCE_FIELDS:
            raise HTTPException(400, f"Invalid condition dimension: {name}")
        if span.min > span.max:
            raise HTTPException(400, f"Invalid range for {name}: min > max")
        ranges[name] = (span.min, span.max)
    missing = [n for n in USER_COND_SOURCE_FIELDS if n not in ranges]
    if missing:
        raise HTTPException(400, f"Missing condition ranges: {', '.join(missing)}")
    return ranges


@app.post("/api/select_condition", response_model=SelectConditionResponse)
def select_condition_api(body: SelectConditionRequest) -> SelectConditionResponse:
    manifest = get_manifest_path()
    if not manifest.is_file():
        raise HTTPException(500, "Chart library not found.")

    ranges = _parse_condition_ranges(body.ranges)

    try:
        payload = service.select_condition_from_ranges(
            manifest,
            ranges,
            body.canonical_bpm_norm,
        )
    except CondSelectionError as exc:
        raise HTTPException(
            404,
            "No charts match these ranges. Widen them and search again.",
        ) from exc
    return SelectConditionResponse(**payload)


@app.post("/api/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest) -> GenerateResponse:
    ckpt = get_checkpoint_path()
    if not ckpt.is_file():
        raise HTTPException(500, "Checkpoint not found. Please configure the checkpoint path.")

    manifest = get_manifest_path()
    final_cond_vec = body.final_cond_vec
    condition_payload = None

    if final_cond_vec is None:
        if body.condition_ranges is None or body.canonical_bpm_norm is None:
            raise HTTPException(400, "Provide final_cond_vec or condition_ranges + canonical_bpm_norm")
        ranges = _parse_condition_ranges(body.condition_ranges)
        try:
            condition_payload = service.select_condition_from_ranges(
                manifest,
                ranges,
                body.canonical_bpm_norm,
            )
        except CondSelectionError as exc:
            raise HTTPException(
                404,
                "No charts match these ranges. Widen them and search again.",
            ) from exc
        final_cond_vec = condition_payload["final_cond_vec"]

    try:
        job_id = service.generate(
            file_id=body.file_id,
            bpm=body.bpm,
            offset_ms=body.offset_ms,
            metadata=body.metadata.model_dump(),
            final_cond_vec=final_cond_vec,
            manifest_path=manifest,
            condition_payload=condition_payload or body.selected_condition,
            request_payload=body.model_dump(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Generation failed. See backend logs for details. ({exc})") from exc

    return GenerateResponse(job_id=job_id, status="running")


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    try:
        payload = service.job_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return JobStatusResponse(**payload)


@app.get("/api/download/{job_id}/osu")
def download_osu(job_id: str) -> FileResponse:
    try:
        path = service.download_path(job_id, "osu")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/download/{job_id}/osz")
def download_osz(job_id: str) -> FileResponse:
    try:
        path = service.download_path(job_id, "osz")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, filename=path.name, media_type="application/zip")
