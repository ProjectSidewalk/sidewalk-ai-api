"""
Regression check: send the same labels to two running instances of the API and report any output differences.

Meant for checking that a change (dependency bump, refactor, new image) doesn't change what the models say. Run it on
the server with prod and a dev instance (`./docker_build.sh dev && ./docker_start.sh dev`) both up:

    python3 regression/compare.py                     # prod on :60654 (A) vs dev on :60655 (B)
    python3 regression/compare.py --out results.csv   # also save every response

Uses only the standard library, so the server's system python3 is enough. The API key comes from SIDEWALK_AI_API_KEY,
or from ~/.sidewalk-ai-api.env like docker_start.sh.

Each label is classified as:
    FAIL   an outcome changed: HTTP status, validation result (including the website's "unsure" bucket, i.e. estimated
           accuracy below UNSURE_BELOW), or the recommended tags to add/remove.
    DRIFT  outcomes match, but the validator score or a tag score moved more than --tolerance.
    same   everything within tolerance.
    SKIP   both instances errored on the label, so there's nothing to compare.
Exits 1 if any label FAILs, 0 otherwise.
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

UNSURE_BELOW = 0.92  # The website treats a validation result as "unsure" below this estimated accuracy.
DEFAULT_LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.csv")


def load_api_key():
    key = os.environ.get("SIDEWALK_AI_API_KEY", "").strip()
    env_path = os.path.expanduser("~/.sidewalk-ai-api.env")
    if not key and os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line.startswith("SIDEWALK_AI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("SIDEWALK_AI_API_KEY is not set and wasn't found in ~/.sidewalk-ai-api.env")
    return key


def call(url, key, label, timeout):
    """Returns (status, parsed JSON or None, seconds)."""
    data = urllib.parse.urlencode({
        "label_type": label["label_type"],
        "panorama_id": label["pano_id"],
        "city": label["city_id"],
        "x": label["x"],
        "y": label["y"],
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, body = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    except Exception as e:
        return f"error: {e}", None, time.perf_counter() - t0
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    return status, parsed, time.perf_counter() - t0


def website_validation(resp):
    """The validation outcome as the website shows it: the result, or "unsure" when estimated accuracy is low."""
    if "validation_result" not in resp:
        return None
    if resp.get("validation_estimated_accuracy", 0) < UNSURE_BELOW:
        return "unsure"
    return resp["validation_result"]


def compare(a, b, tolerance):
    """Returns (verdict, list of difference descriptions, dict of score deltas)."""
    status_a, ra, _ = a
    status_b, rb, _ = b
    if status_a != 200 or status_b != 200 or ra is None or rb is None:
        if status_a == status_b:
            return "SKIP", [f"both returned {status_a}"], {}
        return "FAIL", [f"status {status_a} -> {status_b}"], {}

    fails, drifts, deltas = [], [], {}
    if ra.get("validation_result") != rb.get("validation_result"):
        fails.append(f"validation {ra.get('validation_result')} -> {rb.get('validation_result')}")
    elif website_validation(ra) != website_validation(rb):
        fails.append(f"website validation {website_validation(ra)} -> {website_validation(rb)}")
    for field in ("tags", "tags_not_present"):
        sa, sb = set(ra.get(field, [])), set(rb.get(field, []))
        if sa != sb:
            fails.append(f"{field}: -{sorted(sa - sb)} +{sorted(sb - sa)}")

    if "validation_score" in ra and "validation_score" in rb:
        deltas["validation_score"] = abs(ra["validation_score"] - rb["validation_score"])
    scores_a, scores_b = ra.get("tag_scores", {}), rb.get("tag_scores", {})
    if set(scores_a) != set(scores_b):
        fails.append(f"tag_scores keys: -{sorted(set(scores_a) - set(scores_b))} +{sorted(set(scores_b) - set(scores_a))}")
    for tag in set(scores_a) & set(scores_b):
        deltas[f"tag:{tag}"] = abs(scores_a[tag] - scores_b[tag])
    for name, d in sorted(deltas.items()):
        if d > tolerance:
            drifts.append(f"{name} moved {d:.2e}")

    if fails:
        return "FAIL", fails + drifts, deltas
    if drifts:
        return "DRIFT", drifts, deltas
    return "same", [], deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="CSV of label_id,label_type,pano_id,city_id,x,y")
    parser.add_argument("--a", default="http://127.0.0.1:60654/process", help="Baseline endpoint (default: prod)")
    parser.add_argument("--b", default="http://127.0.0.1:60655/process", help="Candidate endpoint (default: dev)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Largest score change (absolute) still counted as the same (default 1e-3)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds (default 300)")
    parser.add_argument("--out", help="Optional CSV path to save both full responses for every label")
    args = parser.parse_args()

    key = load_api_key()
    with open(args.labels) as f:
        labels = list(csv.DictReader(f))
    print(f"Comparing A={args.a} vs B={args.b} on {len(labels)} labels (tolerance {args.tolerance:g})\n")

    verdicts = Counter()
    max_deltas = defaultdict(float)  # label_type -> largest score change seen
    sources = Counter()
    version_diffs = set()  # (field, A value, B value) for api_version / model ids that differ
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, label in enumerate(labels, 1):
            fut_a = pool.submit(call, args.a, key, label, args.timeout)
            fut_b = pool.submit(call, args.b, key, label, args.timeout)
            a, b = fut_a.result(), fut_b.result()
            if 401 in (a[0], b[0]):
                raise SystemExit("Got 401 Unauthorized: the API key doesn't match the running instance(s).")
            verdict, notes, deltas = compare(a, b, args.tolerance)
            verdicts[verdict] += 1
            if deltas:
                max_deltas[label["label_type"]] = max(max_deltas[label["label_type"]], max(deltas.values()))
            for side, (status, resp, _) in (("A", a), ("B", b)):
                if status == 200 and resp:
                    sources[f"{side}:{resp.get('image_source')}"] += 1
            if a[0] == b[0] == 200 and a[1] and b[1]:
                for field in ("api_version", "validator_model_id", "validator_training_date",
                              "tagger_model_id", "tagger_training_date"):
                    if a[1].get(field) != b[1].get(field):
                        version_diffs.add((field, a[1].get(field), b[1].get(field)))

            line = (f"[{i}/{len(labels)}] {verdict:5} {label['label_type']:14} label {label['label_id']} "
                    f"(A {a[2]:.1f}s, B {b[2]:.1f}s)")
            print(line + ("".join(f"\n      {n}" for n in notes) if verdict != "same" else ""))
            rows.append({**label, "verdict": verdict, "notes": "; ".join(notes),
                         "status_a": a[0], "response_a": json.dumps(a[1]),
                         "status_b": b[0], "response_b": json.dumps(b[1])})

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote full responses to {args.out}")

    print("\n=== summary ===")
    print("  " + ", ".join(f"{v}: {verdicts[v]}" for v in ("same", "DRIFT", "FAIL", "SKIP")))
    print("  largest score change by label type: "
          + ", ".join(f"{t} {d:.2e}" for t, d in sorted(max_deltas.items())))
    print("  image sources: " + ", ".join(f"{s} x{n}" for s, n in sorted(sources.items())))
    if sources.keys() - {"A:cache", "B:cache"}:
        print("  note: some panoramas came from Google rather than the cache, so differences there may be the image.")
    for field, va, vb in sorted(version_diffs, key=str):
        print(f"  note: {field} differs: A={va} B={vb}")

    raise SystemExit(1 if verdicts["FAIL"] else 0)


if __name__ == "__main__":
    main()
