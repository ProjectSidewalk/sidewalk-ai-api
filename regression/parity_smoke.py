"""
Deploy smoke test: /process on the parity panorama must reproduce the probabilities training's pipeline produced.

sidewalk-validator-ai's train/serve parity check (its docs/serving-parity-2026-10.md) ran both pipelines on one
panorama (sidewalk-panorama-tools samples/sample_pano.jpg) at ten points and four poses, and wrote the validators'
p(incorrect) for every case to parity_expected.csv here (a copy of its docs/serving-parity-2026-10/expected_process.csv).
This script sends those rows to a running instance and requires |p(incorrect) - expected_p_incorrect| <= --tolerance
for every row with in_smoke_set true (the rows whose centre depth is at least 5 mm from a crop-size boundary; the
others can legitimately differ by a pixel of crop, which moves p by up to 0.19, and are reported but never fail).
Every response is also parsed whole, the way the website's AiService parses it: a missing or mistyped field, a
training date not in MM-dd-yyyy, an image_source outside the website's enum or a revision that is not a commit fails
the row (the website would log and cast no vote). The served validation_estimated_accuracy must equal this checkout's
accuracy_mappings/<type>.json at the served confidence (so an image built from other maps fails) and be within
--accuracy-tolerance of the CSV's expected_accuracy (so a stale map or stale expected rows fail).

One-time setup on the server: put the sample panorama in the pano cache the instance mounts, under a city named
`parity`, so the instance reads it like any cached pano:

    mkdir -p <SIDEWALK_SCRAPES_DIR>/parity/pa
    cp sidewalk-panorama-tools/samples/sample_pano.jpg <SIDEWALK_SCRAPES_DIR>/parity/pa/parityPano01.jpg

Then, with the dev instance up (`./docker_build.sh dev && ./docker_start.sh dev`):

    python3 regression/parity_smoke.py                       # dev on :60655
    python3 regression/parity_smoke.py --url http://127.0.0.1:60654/process   # prod

Uses only the standard library. The API key comes from SIDEWALK_AI_API_KEY or ~/.sidewalk-ai-api.env, as compare.py.

The expected values are for the validator revisions the CSV was generated with (its provenance is in the
sidewalk-validator-ai doc); after a model-map change, regenerate it there (`analysis/serving_parity.py expected`
against this repo's model_map.py) and copy it here, or every row fails for the right reason. p(incorrect) is
`validation_score` when `validation_result` is `incorrect`, else 1 - `validation_score`.

Exit status: 0 when every smoke-set row is within tolerance and no request errored; 1 otherwise.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load_api_key  # noqa: E402

DEFAULT_EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_expected.csv")


# What the website reads from a /process response (AiService.scala): a field missing, of the wrong type, a
# training_date not in MM-dd-yyyy or an image_source outside its enum makes its parse throw, and it then logs and
# casts no vote. So every response is checked whole, not just its probabilities.
WEBSITE_IMAGE_SOURCES = {"cache", "download"}
TAGGER_LABEL_TYPES = {"crosswalk", "curbramp", "obstacle", "surfaceproblem"}
REVISION = re.compile(r"[0-9a-f]{40}")
DEFAULT_MAPS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "accuracy_mappings")


def lookup_accuracy(class_map, confidence):
    """The API's lookup: walk ascending thresholds, keep the highest one <= confidence (0 below the first)."""
    accuracy = 0
    for threshold, acc in class_map.items():
        if float(threshold) <= confidence:
            accuracy = acc
    return accuracy


def check_response(resp, label_type, maps=None):
    """Problems with a /process response that would make the website's parse fail or that show the wrong model or
    map is being served. Returns a list of strings, empty when the response is sound."""
    problems = []

    def need(name, types, ok=lambda v: True, why=""):
        if name not in resp:
            problems.append(f"missing {name}")
            return None
        v = resp[name]
        if not isinstance(v, types) or isinstance(v, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
            problems.append(f"{name} is {type(v).__name__}, not {types}")
            return None
        if not ok(v):
            problems.append(f"{name} = {v!r}{': ' + why if why else ''}")
        return v

    def is_date(v):
        try:
            return bool(re.fullmatch(r"\d{2}-\d{2}-\d{4}", v)) and datetime.strptime(v, "%m-%d-%Y") is not None
        except ValueError:
            return False

    need("label_type", str, lambda v: v == label_type, f"expected {label_type}")
    need("api_version", str, lambda v: bool(v.strip()))
    need("image_source", str, lambda v: v in WEBSITE_IMAGE_SOURCES, f"not in the website's enum {sorted(WEBSITE_IMAGE_SOURCES)}")
    need("tilt_applied", bool)
    need("coord_frame", str, lambda v: v in ("level", "rig"))
    result = need("validation_result", str, lambda v: v in ("correct", "incorrect"))
    score = need("validation_score", (int, float), lambda v: 0.0 <= v <= 1.0)
    accuracy = need("validation_estimated_accuracy", (int, float), lambda v: 0.0 <= v <= 1.0)
    need("validator_model_id", str, lambda v: v == f"sidewalk-validator-ai-{label_type}")
    need("validator_training_date", str, is_date, "not MM-dd-yyyy")
    need("validator_revision", str, lambda v: bool(REVISION.fullmatch(v)), "not a 40-hex HuggingFace commit")
    if label_type in TAGGER_LABEL_TYPES:
        need("tags", list)
        need("tags_not_present", list)
        need("tag_scores", dict, lambda v: all(isinstance(x, (int, float)) for x in v.values()), "non-numeric score")
        need("tagger_model_id", str)
        need("tagger_training_date", str, is_date, "not MM-dd-yyyy")
        need("tagger_revision", str, lambda v: bool(REVISION.fullmatch(v)), "not a 40-hex HuggingFace commit")
    if maps and result in ("correct", "incorrect") and score is not None and accuracy is not None:
        want = lookup_accuracy(maps[label_type][result], float(score))
        if abs(float(accuracy) - want) > 1e-9:
            problems.append(f"validation_estimated_accuracy {accuracy} but this checkout's map gives {want} at "
                            f"confidence {score}: the instance serves a different accuracy_mappings/{label_type}.json")
    return problems


def p_incorrect(resp):
    """The served probability of the `incorrect` verdict, whichever verdict the model chose."""
    score = float(resp["validation_score"])
    return score if resp["validation_result"] == "incorrect" else 1.0 - score


def call(url, key, row, city, pano_id, timeout):
    """POSTs one expected row's request; returns (status, parsed JSON or None)."""
    fields = {"label_type": row["label_type"], "panorama_id": pano_id, "city": city, "x": row["x"], "y": row["y"],
              "tilt_beta": row["tilt_beta"]}
    for name in ("camera_pitch", "camera_roll"):
        if row.get(name, "") != "":
            fields[name] = row[name]
    req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(),
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, body = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    except Exception as e:
        return f"error: {e}", None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", default="http://127.0.0.1:60655/process", help="the instance's /process URL")
    ap.add_argument("--expected", default=DEFAULT_EXPECTED, help="expected rows CSV (default: parity_expected.csv)")
    ap.add_argument("--tolerance", type=float, default=0.01, help="max |p(incorrect) - expected| (default 0.01)")
    ap.add_argument("--city", default="parity", help="cache folder the sample pano was put under (default parity)")
    ap.add_argument("--panorama-id", default="parityPano01")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--accuracy-tolerance", type=float, default=0.02,
                    help="max |validation_estimated_accuracy - expected_accuracy| on smoke-set rows (default 0.02: "
                         "a 0.01 shift in p can cross one grid step of the map, a stale map moves it far more)")
    ap.add_argument("--maps-dir", default=DEFAULT_MAPS_DIR,
                    help="this checkout's accuracy_mappings/, which the instance must reproduce exactly at the served "
                         "confidence (default: next to this script's repo; '' to skip)")
    ap.add_argument("--out", help="also write every row's served value and difference to this CSV")
    args = ap.parse_args()
    key = load_api_key()
    maps = None
    if args.maps_dir and os.path.isdir(args.maps_dir):
        maps = {t: json.load(open(os.path.join(args.maps_dir, f"{t}.json")))
                for t in ("crosswalk", "curbramp", "nocurbramp", "obstacle", "surfaceproblem")}
    elif args.maps_dir:
        print(f"WARNING: {args.maps_dir} not found; the served map is not checked against this checkout")

    with open(args.expected, newline="") as f:
        rows = list(csv.DictReader(f))
    results, failures, errors, first_source = [], 0, 0, None
    for row in rows:
        in_smoke = row["in_smoke_set"].strip().lower() == "true"
        status, resp = call(args.url, key, row, args.city, args.panorama_id, args.timeout)
        rec = {"point_id": row["point_id"], "pose": row["pose"], "label_type": row["label_type"], "in_smoke_set": in_smoke,
               "expected_p_incorrect": row["expected_p_incorrect"], "expected_accuracy": row.get("expected_accuracy", ""),
               "status": status, "served_p_incorrect": "", "abs_diff": "", "served_accuracy": "", "accuracy_diff": "",
               "verdict": "", "problems": ""}
        tag = f"{row['point_id']} {row['pose']:5s} {row['label_type']:14s}"
        problems = [] if not isinstance(resp, dict) else check_response(resp, row["label_type"], maps)
        if status != 200 or not isinstance(resp, dict) or "validation_score" not in resp:
            errors += 1
            rec["verdict"] = "ERROR"
            detail = resp.get("error") if isinstance(resp, dict) else resp
            print(f"ERROR {tag} status {status}: {detail}")
        else:
            if first_source is None:
                first_source = resp.get("image_source")
            served = p_incorrect(resp)
            diff = abs(served - float(row["expected_p_incorrect"]))
            rec.update(served_p_incorrect=f"{served:.6f}", abs_diff=f"{diff:.6f}")
            acc_text = ""
            if row.get("expected_accuracy", "") != "" and isinstance(resp.get("validation_estimated_accuracy"), (int, float)):
                acc_diff = abs(float(resp["validation_estimated_accuracy"]) - float(row["expected_accuracy"]))
                rec.update(served_accuracy=f"{resp['validation_estimated_accuracy']:.4f}", accuracy_diff=f"{acc_diff:.4f}")
                acc_text = f" accuracy {float(row['expected_accuracy']):.4f} served {resp['validation_estimated_accuracy']:.4f}"
                if in_smoke and not acc_diff <= args.accuracy_tolerance:
                    problems.append(f"validation_estimated_accuracy off by {acc_diff:.4f} from expected_accuracy "
                                    f"(stale map or stale expected rows)")
            if (in_smoke and not diff <= args.tolerance) or problems:   # a NaN fails
                failures += 1
                rec["verdict"] = "FAIL"
            else:
                rec["verdict"] = "ok" if in_smoke else "info"
            flag = "FAIL" if rec["verdict"] == "FAIL" else ("    " if in_smoke else "info")
            print(f"{flag} {tag} expected {float(row['expected_p_incorrect']):.4f} served {served:.4f} diff {diff:.4f}{acc_text}")
        for problem in problems:
            print(f"     {tag} {problem}")
        rec["problems"] = "; ".join(problems)
        results.append(rec)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)

    smoke = [r for r in results if r["in_smoke_set"]]
    diffs = [float(r["abs_diff"]) for r in smoke if r["abs_diff"]]
    print(f"\n{len(smoke)} smoke-set rows of {len(results)}: {failures} failed (p over {args.tolerance}, accuracy over "
          f"{args.accuracy_tolerance}, or a response the website could not use), {errors} errors"
          + (f", max |diff| {max(diffs):.4f}" if diffs else ""))
    if first_source and first_source != "cache":
        print(f"WARNING: image_source was {first_source!r}, not 'cache': the sample pano is not in the cache under "
              f"city {args.city!r} and pano id {args.panorama_id!r}, so the served pixels are not the parity pano's.")
        return 1
    return 1 if failures or errors or not smoke else 0


if __name__ == "__main__":
    sys.exit(main())
