"""Pydantic schemas for demo API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from audio2map.features.cond import COND_VEC_DIM


class HealthResponse(BaseModel):
    status: str
    checkpoint_exists: bool
    manifest_exists: bool
    cond_vec_dim: int = COND_VEC_DIM


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str = "uploaded"


class EstimateTimingRequest(BaseModel):
    file_id: str


class EstimateTimingResponse(BaseModel):
    bpm: float
    offset_ms: float
    canonical_bpm: float
    canonical_bpm_norm: float
    debug: dict[str, Any] = Field(default_factory=dict)


class ConditionRange(BaseModel):
    min: float
    max: float


class SelectConditionRequest(BaseModel):
    ranges: dict[str, ConditionRange]
    canonical_bpm_norm: float


class SelectConditionResponse(BaseModel):
    selected_chart_id: str
    selected_osu_path: str
    matched_candidate_count: int
    cond_vec_names: list[str]
    selected_original_cond_vec: list[float]
    selected_source_values: dict[str, float]
    final_cond_vec: list[float]
    canonical_bpm_norm_source: str = "uploaded_audio_estimated_bpm"
    selection_rule: str = "uniform_among_matches"


class BeatmapMetadata(BaseModel):
    title: str = "Untitled"
    artist: str = "Unknown Artist"
    creator: str = "Audio2Map"
    difficulty_name: str = "Generated"


class GenerateRequest(BaseModel):
    file_id: str
    bpm: float
    offset_ms: float
    metadata: BeatmapMetadata = Field(default_factory=BeatmapMetadata)
    final_cond_vec: list[float] | None = None
    condition_ranges: dict[str, ConditionRange] | None = None
    canonical_bpm_norm: float | None = None
    selected_condition: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    note_count: int | None = None
    osu_filename: str | None = None
    osz_filename: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    step: str | None = None
    error: str | None = None
    selected_condition: dict[str, Any] | None = None
    generation_report: dict[str, Any] | None = None
    note_count: int | None = None
    osu_filename: str | None = None
    osz_filename: str | None = None
    windows_done: int = 0
    windows_total: int = 0
