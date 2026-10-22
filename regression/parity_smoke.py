"""
Deploy smoke test: /process on the parity panorama must reproduce the probabilities training's pipeline produced.

sidewalk-validator-ai's train/serve parity check (its docs/serving-parity-2026-10.md) ran both pipelines on one
panorama (sidewalk-panorama-tools samples/sample_pano.jpg) at ten points and four poses, and wrote the validators'
p(incorrect) for every case to parity_expected.csv here (a copy of its docs/serving-parity-2026-10/expected_process.csv).
This script sends those rows to a running instance and requires |p(incorrect) - expected_p_incorrect| <= --tolerance
for every row with in_smoke_set true (the rows whose centre depth is at least 5 mm from a crop-size boundary; the
others can legitimately differ by a pixel of crop, which moves p by up to 0.19, and are reported but never fail).

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
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load_api_key  # noqa: E402

DEFAULT_EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_expected.csv")


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
    ap.add_argument("--out", help="also write every row's served value and difference to this CSV")
    args = ap.parse_args()
    key = load_api_key()

    with open(args.expected, newline="") as f:
        rows = list(csv.DictReader(f))
    results, failures, errors, first_source = [], 0, 0, None
    for row in rows:
        in_smoke = row["in_smoke_set"].strip().lower() == "true"
        status, resp = call(args.url, key, row, args.city, args.panorama_id, args.timeout)
        rec = {"point_id": row["point_id"], "pose": row["pose"], "label_type": row["label_type"], "in_smoke_set": in_smoke,
               "expected_p_incorrect": row["expected_p_incorrect"], "status": status, "served_p_incorrect": "",
               "abs_diff": "", "verdict": ""}
        if status != 200 or not isinstance(resp, dict) or "validation_score" not in resp:
            errors += 1
            rec["verdict"] = "ERROR"
            detail = resp.get("error") if isinstance(resp, dict) else resp
            print(f"ERROR {row['point_id']} {row['pose']:5s} {row['label_type']:14s} status {status}: {detail}")
        else:
            if first_source is None:
                first_source = resp.get("image_source")
            served = p_incorrect(resp)
            diff = abs(served - float(row["expected_p_incorrect"]))
            rec.update(served_p_incorrect=f"{served:.6f}", abs_diff=f"{diff:.6f}")
            if in_smoke and not diff <= args.tolerance:   # a NaN fails
                failures += 1
                rec["verdict"] = "FAIL"
            else:
                rec["verdict"] = "ok" if in_smoke else "info"
            flag = "FAIL" if rec["verdict"] == "FAIL" else ("    " if in_smoke else "info")
            print(f"{flag} {row['point_id']} {row['pose']:5s} {row['label_type']:14s} expected {float(row['expected_p_incorrect']):.4f} "
                  f"served {served:.4f} diff {diff:.4f}")
        results.append(rec)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)

    smoke = [r for r in results if r["in_smoke_set"]]
    diffs = [float(r["abs_diff"]) for r in smoke if r["abs_diff"]]
    print(f"\n{len(smoke)} smoke-set rows of {len(results)}: {failures} over tolerance {args.tolerance}, {errors} errors"
          + (f", max |diff| {max(diffs):.4f}" if diffs else ""))
    if first_source and first_source != "cache":
        print(f"WARNING: image_source was {first_source!r}, not 'cache': the sample pano is not in the cache under "
              f"city {args.city!r} and pano id {args.panorama_id!r}, so the served pixels are not the parity pano's.")
        return 1
    return 1 if failures or errors or not smoke else 0


if __name__ == "__main__":
    sys.exit(main())
