import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from audio2map.cli.eval_v1 import build_parser
from audio2map.dataset.split import ChartSnapshot, SplitManifest
from audio2map.dataset.v1.example import ChartAudio
from audio2map.eval.v1 import run
from audio2map.eval.v1.archive import load_archive
from audio2map.eval.v1.inputs import freeze_json, select_charts
from audio2map.eval.v1.motifs import fit_vocabulary
from audio2map.eval.v1.run import EvalConfig, evaluate
from audio2map.eval.v1.rescore import RescoreConfig, rescore
from audio2map.eval.v1 import rescore as rescore_module
from audio2map.eval.v1 import probes as probes_module
from audio2map.eval.v1.probes import ProbeConfig, run_probes
from audio2map.eval.v1.reporting.html import write_html
from audio2map.eval.v1.comparison import compare_runs
from audio2map.eval.v1 import protocol as protocol_module
from audio2map.eval.v1.protocol import CalibrationConfig, calibrate_patterns
from audio2map.eval.v1.inputs import file_digest
from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta
from audio2map.generate.v1.chart import V1Chart
from audio2map.generate.v1.notes import notes_from_events
from audio2map.generate.v1.sample import SampleResult, VALID
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import decode_tokens, encode_chart
from audio2map.tokens.v1.vocab import build_vocab


class GoldModel(torch.nn.Module):
    def forward(self, batch):
        logits = torch.full((*batch.token_ids[:, 1:].shape, len(build_vocab())), -4.0)
        return logits.scatter_(2, batch.token_ids[:, 1:, None], 4.0)


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    charts = []
    for index, split in enumerate(("val", "val", "train")):
        folder = root / str(index)
        folder.mkdir(parents=True)
        (folder / "audio.wav").write_bytes(b"fixture")
        path = folder / "a.osu"
        path.write_text(
            "osu file format v14\n[General]\nAudioFilename:audio.wav\nMode:3\n"
            "[Difficulty]\nCircleSize:4\n[TimingPoints]\n0,500,4,2,0,100,1,0\n"
            f"[Metadata]\nTitle:{index}\n[HitObjects]\n64,192,0,1,0,0:0:0:0:\n"
        )
        charts.append(ChartSnapshot(f"{index}/a.osu", str(index), split))
    split_path = tmp_path / "split.json"
    manifest = SplitManifest(
        "beatmapset_v1",
        "raw_parent",
        0,
        {"train": 0.8, "val": 0.1, "test": 0.1},
        {chart.group_id: chart.split for chart in charts},
        charts,
    )
    split_path.write_text(json.dumps(manifest.to_dict()))
    audio = ChartAudio(
        notes=(ManiaNote(0, 0, NoteType.TAP),),
        redlines=(Redline(0.0, 500.0, 4, 0),),
        pcm=np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32),
        t_audio_ms=1000.0,
    )
    encoded = encode_chart(list(audio.notes), audio.redlines, 1000.0)
    events = decode_tokens(encoded.tokens, encoded.grid)
    notes, warnings = notes_from_events(events)
    vocabulary = tmp_path / "motifs.json"
    vocabulary.write_text(
        json.dumps(
            fit_vocabulary(
                [encoded.tokens],
                k=8,
                corpus={
                    "split": "train", "split_sha256": file_digest(split_path),
                    "raw_root": str(root), "charts": ["2/a.osu"],
                    "chart_sha256": {"2/a.osu": file_digest(root / "2/a.osu")},
                    "subset": False,
                },
            )
        )
    )
    monkeypatch.setattr(
        "audio2map.eval.v1.inputs.filter_v1_charts", lambda charts, *args, **kwargs: (charts, None)
    )
    monkeypatch.setattr(run, "load_chart_audio", lambda _path, *, use_cache: audio)
    monkeypatch.setattr(
        run,
        "prepare_metadata",
        lambda charts, *args, **kwargs: {
            chart.relpath: ChartMeta(
                chart.relpath, official_sr=2 + int(chart.relpath.split("/")[0])
            )
            for chart in charts
        },
    )
    monkeypatch.setattr(
        run, "compute_chart_meta", lambda _path: ChartMeta("generated", official_sr=2)
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(run, "load_v1_model", lambda *args: GoldModel())
    calls = []

    def generate(_model, pcm, redlines, meta, **kwargs):
        calls.append((meta, kwargs["seed"]))
        return V1Chart(notes, warnings, tuple(encoded.tokens), 87, 1)

    monkeypatch.setattr(run, "generate_v1_chart", generate)
    monkeypatch.setattr(
        run,
        "generate_v1_unconstrained",
        lambda *args, **kwargs: SampleResult(tuple(encoded.tokens), VALID),
    )
    cfg = EvalConfig(
        checkpoint,
        split_path,
        "val",
        root,
        tmp_path / "out",
        device="cpu",
        workers=1,
        motif_vocab=vocabulary,
    )
    return cfg, calls


def test_all_modes_write_frozen_inputs_and_separate_metrics(local_run):
    cfg, calls = local_run
    report = evaluate(cfg)
    assert len(calls) == 18  # 5 matched_full seeds + 4 other scenarios, per chart.
    assert set(report["teacher"]) == set(cfg.scenarios)
    assert report["teacher"]["matched_full"]["acc_token"] == 1
    assert report["constrained"]["matched_full"]["match"]["micro"]["f1"] == 1
    assert report["constrained"]["none"]["condition_error"]["official_sr"]["mae"] is None
    assert set(report["unconstrained"]) == {"matched_full", "matched_partial", "none"}
    assert all(value["samples"] == 2 for value in report["unconstrained"].values())
    assert all(len(value["pairs"]) == 10 for value in report["diversity"].values())
    assert report["motif"]["matched_full"]["row"]["mae"] == 0
    assert (cfg.out / "summary.json").is_file()
    records = [json.loads(line) for line in (cfg.out / "records.jsonl").read_text().splitlines()]
    assert len(records) == 34
    assert all("match" not in row for row in records if row["mode"] == "unconstrained")
    resumed = evaluate(replace(cfg, resume=True))
    assert len(calls) == 18
    assert resumed["reused_records"] == 34
    assert resumed["constrained"] == report["constrained"]
    assert len((cfg.out / "records.jsonl").read_text().splitlines()) == 34
    generated = next(row for row in records if row["mode"] == "constrained")
    token_path = Path(generated["output"]).with_name("tokens.json")
    token_path.write_text('["<BOS>"]')
    repaired = evaluate(replace(cfg, resume=True))
    assert len(calls) == 19
    assert repaired["reused_records"] == 33
    assert repaired["constrained"] == report["constrained"]
    with pytest.raises(ValueError, match="frozen inputs differ"):
        evaluate(replace(cfg, temperature=0.5))


def test_requested_charts_must_belong_to_the_explicit_split(local_run):
    cfg, _calls = local_run
    with pytest.raises(ValueError, match="not in split"):
        select_charts(
            cfg.split_manifest, cfg.root, "val", limit=None, seed=0, paths=(Path("2/a.osu"),)
        )
    with pytest.raises(ValueError, match="fit-motifs requires"):
        evaluate(replace(cfg, mode="fit-motifs"))


def test_cuda_generation_rejects_an_unsupported_precision(local_run, monkeypatch):
    cfg, _calls = local_run
    monkeypatch.setattr(run, "_device", lambda _name: torch.device("cuda"))
    with pytest.raises(ValueError, match="fp32 is teacher-only"):
        evaluate(replace(cfg, precision="fp32"))


def test_failed_resume_does_not_leave_a_stale_complete_report(local_run, monkeypatch):
    cfg, _calls = local_run
    evaluate(cfg)

    def interrupted(*args):
        raise RuntimeError("fixture interrupted")

    monkeypatch.setattr(run, "load_v1_model", interrupted)
    with pytest.raises(RuntimeError, match="fixture interrupted"):
        evaluate(replace(cfg, resume=True))
    assert not (cfg.out / "summary.json").exists()
    assert not (cfg.out / "summary.md").exists()
    assert (cfg.out / "records.jsonl").exists()


def test_freeze_and_cli_distinguish_v0_and_v1(tmp_path):
    path = tmp_path / "frozen.json"
    freeze_json(path, {"seed": 0})
    freeze_json(path, {"seed": 0})
    with pytest.raises(ValueError):
        freeze_json(path, {"seed": 1})
    args = build_parser().parse_args(["--split", "val", "--mode", "teacher"])
    assert args.split == "val" and args.mode == "teacher"


def test_eval_rejects_vocabulary_fitted_on_validation(local_run):
    cfg, calls = local_run
    fitted = json.loads(cfg.motif_vocab.read_text())
    fitted["corpus"]["charts"] = ["0/a.osu"]
    cfg.motif_vocab.write_text(json.dumps(fitted))
    with pytest.raises(ValueError, match="outside the train split"):
        evaluate(cfg)
    assert not calls


def test_rescore_refreshes_statistics_without_sampling_or_modifying_source(local_run, monkeypatch):
    cfg, calls = local_run
    source = evaluate(cfg)
    original = (cfg.out / "records.jsonl").read_bytes()

    def decode_with_new_duration(path, *, use_cache):
        audio = run.load_chart_audio(path, use_cache=use_cache)
        return replace(audio, pcm=np.pad(audio.pcm, (0, SAMPLE_RATE // 2)), t_audio_ms=1500.0)

    monkeypatch.setattr(rescore_module, "load_chart_audio", decode_with_new_duration)
    monkeypatch.setattr(
        rescore_module, "direct_audio_settings",
        lambda: {**run.direct_audio_settings(), "ffmpeg": "new decoder"},
    )
    monkeypatch.setattr(rescore_module, "compute_chart_meta", run.compute_chart_meta)
    monkeypatch.setattr(rescore_module, "load_v1_model", run.load_v1_model)
    out = cfg.out.with_name("rescored")
    report = rescore(RescoreConfig(cfg.out, out, cfg.motif_vocab, True, "cpu"))
    assert report["run"]["teacher_precision"] == "fp32"
    assert report["run"]["ffmpeg"] == source["run"]["ffmpeg"]
    assert report["run"]["teacher_audio"]["ffmpeg"] == "new decoder"
    assert report["references"] == source["references"]
    with pytest.raises(ValueError, match="audio provenance differs: teacher/ffmpeg"):
        compare_runs(cfg.out, out, cfg.out.with_name("comparison"))
    rescored_again = rescore(RescoreConfig(out, cfg.out.with_name("rescored_again")))
    assert rescored_again["run"]["teacher_audio"] == report["run"]["teacher_audio"]
    assert len(calls) == 18
    assert (cfg.out / "records.jsonl").read_bytes() == original
    assert report["reused_records"] == 34
    assert report["teacher"]["matched_full"]["row_details"]["metrics"]["row_lane"]["rate"] == 1
    view = report["constrained"]["matched_full"]["match"]["views"]
    assert view["note"]["lane_agnostic"]["all"]["micro"]["f1"] == 1
    assert report["diversity"]["0/a.osu"]["pairs"][0]["motif"]["row"]["mae"] == 0
    assert report["reference_distribution"]["row_pattern"] == {"1000": 2}


def test_controlled_probes_use_frozen_inputs_and_resume_without_extra_generation(
    local_run, monkeypatch,
):
    cfg, calls = local_run
    evaluate(cfg)
    monkeypatch.setattr(probes_module, "load_chart_audio", run.load_chart_audio)
    monkeypatch.setattr(probes_module, "compute_chart_meta", run.compute_chart_meta)
    monkeypatch.setattr(probes_module, "load_v1_model", run.load_v1_model)
    out = cfg.out.with_name("probes")
    probe_cfg = ProbeConfig(cfg.out, out, charts=1, teacher_charts=1, device="cpu")
    result = run_probes(probe_cfg)
    assert len(calls) == 18 + 15
    assert result["audio_teacher"]["none"]["5.0"]["nll_shift_minus_correct"]["mean"] == 0
    assert result["audio_generation"]["0.0"]["f1"]["mean"] == 1
    resumed = run_probes(replace(probe_cfg, resume=True))
    assert len(calls) == 18 + 15
    assert resumed["reused_records"] == 21
    rows = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    shifted = next(row for row in rows if row["mode"] == "constrained" and row.get("shift_samples"))
    chart = Path(shifted["output"])
    audio = chart.with_name(shifted["playback_audio"])
    assert f"AudioFilename: {audio.name}" in chart.read_text()
    pcm, rate = sf.read(audio, dtype="float32")
    original = run.load_chart_audio(None, use_cache=False)
    assert rate == SAMPLE_RATE
    np.testing.assert_array_equal(pcm, np.roll(original.pcm, shifted["shift_samples"]))
    audio.unlink()
    repaired = run_probes(replace(probe_cfg, resume=True))
    assert repaired["reused_records"] == 20
    assert len(calls) == 18 + 16
    assert audio.is_file()


def test_html_embeds_figures_and_escapes_input_protocol_text(local_run):
    pytest.importorskip("matplotlib")
    cfg, _calls = local_run
    report = evaluate(cfg)
    rows = [json.loads(line) for line in (cfg.out / "records.jsonl").read_text().splitlines()]
    report["run"]["checkpoint"] = "<script>alert('x')</script>"
    write_html(report, rows, cfg.out)
    html = (cfg.out / "index.html").read_text()
    assert "Reference matching" in html and html.count("<svg") >= 10
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "不计算自动谱面质量总分" in html


@pytest.mark.parametrize("change", ["missing", "unexpected", "condition"])
def test_archive_rejects_complete_lines_that_do_not_match_frozen_cases(local_run, change):
    cfg, _calls = local_run
    evaluate(cfg)
    path = cfg.out / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(load_archive(cfg.out).rows) == len(rows)
    row = rows.pop()
    if change == "unexpected":
        rows.append({**row, "seed": 12345})
    if change == "condition":
        rows.append({**row, "condition": {}})
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))
    message = {
        "missing": "1 missing, 0 unexpected",
        "unexpected": "1 missing, 1 unexpected",
        "condition": "source journal condition differs",
    }[change]
    with pytest.raises(ValueError, match=message):
        load_archive(cfg.out)


def test_checkpoint_comparison_pairs_charts_and_rejects_sampler_changes(local_run):
    cfg, _calls = local_run
    evaluate(cfg)
    right = cfg.out.with_name("right")
    evaluate(replace(cfg, out=right))
    result = compare_runs(cfg.out, right, cfg.out.with_name("comparison"))
    f1 = next(row for row in result["metrics"] if row["metric"] == "event/lane_aware F1")
    assert f1["charts"] == 2
    assert f1["mean"] == f1["low"] == f1["high"] == 0
    run_path = right / "run.json"
    summary_path = right / "summary.json"
    run_data = json.loads(run_path.read_text())
    summary = json.loads(summary_path.read_text())

    def save_run():
        run_path.write_text(json.dumps(run_data))
        summary["run"] = run_data
        summary_path.write_text(json.dumps(summary))

    run_data["temperature"] = 0.8
    save_run()
    with pytest.raises(ValueError, match="temperature"):
        compare_runs(cfg.out, right, cfg.out.with_name("bad_comparison"))
    run_data["temperature"] = cfg.temperature
    run_data["teacher_precision"] = "bf16"
    save_run()
    with pytest.raises(ValueError, match="effective precision.*teacher"):
        compare_runs(cfg.out, right, cfg.out.with_name("bad_precision"))

    run_data["teacher_precision"] = "fp32"
    run_data.pop("audio_source")
    save_run()
    with pytest.raises(ValueError, match="audio provenance differs: teacher/audio_source"):
        compare_runs(cfg.out, right, cfg.out.with_name("bad_audio_source"))

    run_data["audio_source"] = run.direct_audio_settings()["audio_source"]
    run_data["rescore"] = {"refresh_teacher": True, "precision": "fp32", "device": "cpu"}
    save_run()
    with pytest.raises(ValueError, match="audio provenance is missing: teacher"):
        compare_runs(cfg.out, right, cfg.out.with_name("unknown_teacher_audio"))


def test_html_failure_does_not_publish_a_complete_summary(local_run, monkeypatch):
    cfg, _calls = local_run
    monkeypatch.setattr(run, "require_plotting", lambda: None)

    def failed(*args):
        raise RuntimeError("figure failed")

    monkeypatch.setattr(run, "write_html", failed)
    with pytest.raises(RuntimeError, match="figure failed"):
        evaluate(replace(cfg, html=True))
    assert not (cfg.out / "summary.json").exists()
    assert not (cfg.out / "summary.md").exists()


def test_new_probe_protocol_reuses_only_compatible_completed_cases(local_run, monkeypatch):
    cfg, calls = local_run
    evaluate(cfg)
    monkeypatch.setattr(probes_module, "load_chart_audio", run.load_chart_audio)
    monkeypatch.setattr(probes_module, "compute_chart_meta", run.compute_chart_meta)
    monkeypatch.setattr(probes_module, "load_v1_model", run.load_v1_model)
    original = cfg.out.with_name("original_probes")
    run_probes(ProbeConfig(cfg.out, original, charts=1, teacher_charts=1, device="cpu"))
    original_bytes = (original / "records.jsonl").read_bytes()
    protocol = cfg.out.with_name("protocol.json")
    protocol.write_text(json.dumps({
        "protocol_version": 1, "audio_offsets_s": [-5, -0.05, 0, 0.05, 5],
        "controls": {"pattern_jacks": [0, 0.1]},
    }))
    out = cfg.out.with_name("new_probes")
    probe_cfg = ProbeConfig(cfg.out, out, charts=1, teacher_charts=1, device="cpu",
                            protocol=protocol, reuse_runs=(original,))
    result = run_probes(probe_cfg)
    assert result["external_reused_records"] == 9
    assert len(calls) == 18 + 15 + 6
    assert len((out / "records.jsonl").read_text().splitlines()) == 19
    assert (original / "records.jsonl").read_bytes() == original_bytes
    resumed = run_probes(replace(probe_cfg, resume=True))
    assert resumed["reused_records"] == 19
    assert len(calls) == 18 + 15 + 6


def test_pattern_calibration_uses_verified_train_membership(local_run, monkeypatch):
    cfg, _calls = local_run
    evaluate(cfg)
    fitted = json.loads(cfg.motif_vocab.read_text())
    corpus = {"split": "train", "raw_root": str(cfg.root),
              "split_sha256": file_digest(cfg.split_manifest), "charts": ["2/a.osu"],
              "chart_sha256": {"2/a.osu": file_digest(cfg.root / "2/a.osu")}}
    fitted["corpus"] = corpus
    training_vocab = cfg.out.with_name("training_vocab.json")
    training_vocab.write_text(json.dumps(fitted))

    def prepare(paths, root, *, workers, path):
        rows = []
        for key in paths:
            row = ChartMeta(key, pattern_jacks=0.1).to_dict()
            rows.append(json.dumps({**row, "relpath": key}))
        path.write_text("\n".join(rows) + "\n")

    monkeypatch.setattr(protocol_module, "ensure_v1_chart_meta", prepare)
    result = calibrate_patterns(CalibrationConfig(
        cfg.out, training_vocab, cfg.out.with_name("calibration"), charts=1,
    ))
    assert result["calibration"]["selected"] == ["2/a.osu"]
    assert result["controls"]["pattern_jacks"] == [0.0, 0.1]
    fitted["corpus"]["charts"] = ["0/a.osu"]
    training_vocab.write_text(json.dumps(fitted))
    with pytest.raises(ValueError, match="outside the train"):
        calibrate_patterns(CalibrationConfig(
            cfg.out, training_vocab, cfg.out.with_name("bad_calibration"), charts=1,
        ))
