"""regression/parity_smoke.py's checks, without a server: the response parser, the lookup, the preflight."""

import csv
import importlib
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "regression"))
ps = importlib.import_module("parity_smoke")
from sidewalk_ai_api import model_map  # noqa: E402

MAPS = {t: json.load(open(os.path.join(REPO_ROOT, "accuracy_mappings", f"{t}.json")))
        for t in ("crosswalk", "curbramp", "nocurbramp", "obstacle", "surfaceproblem")}


def good_response(label_type="curbramp", score=0.99995, result="correct"):
    resp = {"label_type": label_type, "api_version": "1.2.0", "image_source": "cache", "tilt_applied": False,
            "coord_frame": "level", "validation_result": result, "validation_score": score,
            "validation_estimated_accuracy": ps.lookup_accuracy(MAPS[label_type][result], score),
            "validator_model_id": f"sidewalk-validator-ai-{label_type}", "validator_training_date": "10-21-2026",
            "validator_revision": "0" * 40}
    if label_type in ps.TAGGER_LABEL_TYPES:
        resp.update(tags=[], tags_not_present=[], tag_scores={"a": 0.1}, tagger_model_id="x",
                    tagger_training_date="12-03-2024", tagger_revision="1" * 40)
    return resp


def test_good_responses_pass():
    assert ps.check_response(good_response(), "curbramp", MAPS) == []
    assert ps.check_response(good_response("nocurbramp", 0.7, "incorrect"), "nocurbramp", MAPS) == []


@pytest.mark.parametrize("field, value, what", [
    ("validator_training_date", "2026-10-21", "not MM-dd-yyyy"),       # ISO, which the website cannot parse
    ("validator_training_date", "13-01-2026", "not MM-dd-yyyy"),       # month 13
    ("tagger_training_date", "10/21/2026", "not MM-dd-yyyy"),
    ("image_source", "disk", "not in the website's enum"),
    ("validator_revision", "main", "not a 40-hex"),
    ("coord_frame", "gravity", "coord_frame ="),
    ("validation_result", "unsure", "validation_result ="),
    ("validation_score", 1.5, "validation_score ="),
    ("validation_score", True, "is bool"),
    ("label_type", "crosswalk", "expected curbramp"),
])
def test_bad_fields_are_reported(field, value, what):
    resp = good_response()
    resp[field] = value
    problems = ps.check_response(resp, "curbramp", MAPS)
    assert any(what in p for p in problems), problems


def test_missing_fields_are_reported():
    resp = good_response()
    del resp["tags"]
    del resp["validation_estimated_accuracy"]
    problems = ps.check_response(resp, "curbramp", MAPS)
    assert "missing tags" in problems and "missing validation_estimated_accuracy" in problems


def test_lookup_is_the_floor_not_the_ceiling():
    """The API keeps the highest threshold <= confidence; a ceiling lookup would serve the next grid point's value."""
    table = {"0.5": 0.0, "0.6": 0.5, "0.7": 0.9, "1.0": 0.95}
    assert ps.lookup_accuracy(table, 0.65) == 0.5
    assert ps.lookup_accuracy(table, 0.7) == 0.9
    assert ps.lookup_accuracy(table, 0.49) == 0
    assert ps.lookup_accuracy(table, 1.0) == 0.95
    resp = good_response(score=0.65)
    resp["validation_estimated_accuracy"] = 0.9                      # the ceiling's value
    maps = dict(MAPS)
    maps["curbramp"] = {"correct": table, "incorrect": table}
    problems = ps.check_response(resp, "curbramp", maps)
    assert any("this checkout's map gives 0.5" in p for p in problems), problems


def rows_for(revision, label_types=("curbramp", "crosswalk")):
    return [{"label_type": t, "validator_revision": revision} for t in label_types]


def test_staleness_check(monkeypatch, capsys):
    pinned = {t: "a" * 40 for t in MAPS}
    for t in pinned:
        monkeypatch.setitem(model_map.MODEL_MAP["validator_models"][t], "revision", pinned[t])
    assert ps.preflight(rows_for("a" * 40), None, False) == []
    fatal = ps.preflight(rows_for("b" * 40), None, False)
    assert len(fatal) == 1 and "rows " + "b" * 40 in fatal[0] and "pinned " + "a" * 40 in fatal[0]   # not truncated
    fatal = ps.preflight(rows_for(""), None, False)                  # blank revision is stale, not skipped
    assert len(fatal) == 1 and "(blank)" in fatal[0]
    fatal = ps.preflight([{"label_type": "curbramp"}], None, False)  # no column at all
    assert any("no validator_revision column" in f for f in fatal)
    fatal = ps.preflight(rows_for("a" * 40, ("curbramp", "signal")), None, False)
    assert len(fatal) == 1 and "label_type(s) ['signal']" in fatal[0]
    monkeypatch.setitem(model_map.MODEL_MAP["validator_models"]["curbramp"], "revision", model_map.PLACEHOLDER_REVISION)
    fatal = ps.preflight(rows_for("a" * 40, ("curbramp",)), None, False)
    assert len(fatal) == 1 and model_map.PLACEHOLDER_REVISION in fatal[0]
    assert "no parity_expected.provenance.json" in capsys.readouterr().out


def test_isa_preflight(monkeypatch, capsys):
    for t in MAPS:
        monkeypatch.setitem(model_map.MODEL_MAP["validator_models"][t], "revision", "a" * 40)
    rows = rows_for("a" * 40)
    monkeypatch.setattr(ps.platform, "machine", lambda: "x86_64")
    prov = {"generated_on": {"machine": "arm64", "date": "d"}}
    fatal = ps.preflight(rows, prov, False)
    assert len(fatal) == 1 and "generated on arm64; this machine is x86_64" in fatal[0]
    assert ps.preflight(rows, prov, True) == [] and "WARNING: expected values were generated on arm64" in capsys.readouterr().out
    monkeypatch.setattr(ps.platform, "machine", lambda: "aarch64")   # Linux arm64 spelling matches macOS arm64
    assert ps.preflight(rows, prov, False) == []
    monkeypatch.setattr(ps.platform, "machine", lambda: "AMD64")     # Windows spelling of x86_64
    assert ps.preflight(rows, {"generated_on": {"machine": "x86_64"}}, False) == []
    assert ps.preflight(rows, {"generated_on": {"machine": ""}}, False) == []
    assert "generated_on.machine is empty" in capsys.readouterr().out
    assert ps.preflight(rows, {"file": "x"}, False) == []
    assert "no generated_on block" in capsys.readouterr().out


def test_committed_rows_and_sidecar_are_consistent():
    """The committed CSV, its sidecar and the current model map: the sidecar names the CSV's revisions and arch."""
    path = os.path.join(REPO_ROOT, "regression", "parity_expected.csv")
    rows = list(csv.DictReader(open(path, newline="")))
    prov = ps.provenance_for(path)
    assert prov["generated_on"]["machine"] == "arm64" and prov["validator_revisions"]
    for r in rows:
        assert r["validator_revision"] == prov["validator_revisions"][r["label_type"]]
        assert r["expected_validation_result"] in ("correct", "incorrect") and 0 <= float(r["expected_accuracy"]) <= 1
    assert ps.normalized_arch("arm64") == ps.normalized_arch("aarch64") != ps.normalized_arch("AMD64")
