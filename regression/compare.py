"""
Regression check: send the same labels to two running instances of the API and report any output differences.

Meant for checking that a change (dependency bump, refactor, new image) doesn't change what the models say. Run it on
the server with prod and a dev instance (`./docker_build.sh dev && ./docker_start.sh dev`) both up:

    python3 regression/compare.py                     # prod on :60654 (A) vs dev on :60655 (B)
    python3 regression/compare.py --out results.csv   # also save every response

Every label should be served from the panorama cache so both instances see identical pixels; labels downloaded from
Google are marked, since Google's tiles can differ between downloads. The labels' city_id must match the cache's
folder names (e.g. chicago-il), the same city id the Sidewalk website sends.

Uses only the standard library, so the server's system python3 is enough. The API key comes from SIDEWALK_AI_API_KEY,
or from ~/.sidewalk-ai-api.env like docker_start.sh.

Each label is classified as:
    FAIL   an outcome changed: HTTP status, validation result (including the website's "unsure" bucket, i.e. estimated
           accuracy below UNSURE_BELOW), or the recommended tags to add/remove.
    DRIFT  outcomes match, but the validator score or a tag score moved more than --tolerance.
    same   everything within tolerance.
    SKIP   both instances errored on the label, so there's nothing to compare.
Then a download check sends one label per type to both instances with no city, forcing a download from Google, and
checks that B succeeds wherever A does (scores aren't compared, since separate downloads can differ).

It also compares speed: if B's median response time on the cached labels is more than --max-slowdown times A's, the
run fails. That catches a model silently falling back to the CPU (every model picks CUDA only when torch can see the
GPU) or a similar performance regression. The download check's times include Google and aren't compared.

Exits 0 only if every label is "same" or DRIFT, B isn't too slow, and the download check passes. A SKIP also fails
the run, so a broken setup (cache mount, instance down) can't pass by comparing nothing.
"""

import argparse
import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

UNSURE_BELOW = 0.92  # The website treats a validation result as "unsure" below this estimated accuracy.
DEFAULT_LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.csv")


def load_api_key():
    """Same precedence as docker_start.sh: the key file wins over an exported variable, which may be stale."""
    key = ""
    env_path = os.path.expanduser("~/.sidewalk-ai-api.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line.startswith("SIDEWALK_AI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
    key = key or os.environ.get("SIDEWALK_AI_API_KEY", "").strip()
    if not key:
        raise SystemExit("SIDEWALK_AI_API_KEY is not set and wasn't found in ~/.sidewalk-ai-api.env")
    return key


def call(url, key, label, timeout, use_cache=True):
    """Returns (status, parsed JSON or None, seconds). With use_cache=False, the city is left out so the API has to
    download the panorama from Google."""
    data = urllib.parse.urlencode({
        "label_type": label["label_type"],
        "panorama_id": label["pano_id"],
        "city": label["city_id"] if use_cache else "",
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
    if status_a != status_b:
        return "FAIL", [f"status {status_a} -> {status_b}"], {}
    if status_a != 200:
        return "SKIP", [f"both returned {status_a}"], {}
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return "FAIL", [f"unparseable 200 response from {'A' if not isinstance(ra, dict) else 'B'}"], {}

    fails, drifts, deltas = [], [], {}
    if ra.get("validation_result") != rb.get("validation_result"):
        fails.append(f"validation {ra.get('validation_result')} -> {rb.get('validation_result')}")
    elif website_validation(ra) != website_validation(rb):
        fails.append(f"website validation {website_validation(ra)} -> {website_validation(rb)}")
    for field in ("tags", "tags_not_present"):
        sa, sb = set(ra.get(field, [])), set(rb.get(field, []))
        if sa != sb:
            fails.append(f"{field}: -{sorted(sa - sb)} +{sorted(sb - sa)}")

    for field in ("validation_score", "validation_estimated_accuracy"):
        if field in ra and field in rb:
            deltas[field] = abs(ra[field] - rb[field])
    scores_a, scores_b = ra.get("tag_scores", {}), rb.get("tag_scores", {})
    if set(scores_a) != set(scores_b):
        fails.append(f"tag_scores keys: -{sorted(set(scores_a) - set(scores_b))} +{sorted(set(scores_b) - set(scores_a))}")
    for tag in set(scores_a) & set(scores_b):
        deltas[f"tag:{tag}"] = abs(scores_a[tag] - scores_b[tag])
    for name, d in sorted(deltas.items()):
        if not d <= tolerance:  # Written this way so a NaN counts as drift.
            drifts.append(f"{name} moved {d:.2e}")

    if fails:
        return "FAIL", fails + drifts, deltas
    if drifts:
        return "DRIFT", drifts, deltas
    return "same", [], deltas


def download_check(args, key, labels):
    """The comparison above reads every panorama from the cache, which skips the Google download path (requests,
    urllib3, Brotli, idna, Pillow's tile decoding). This sends one label per type with no city, so both instances
    have to download, and checks that B manages whenever A does. Scores aren't compared: each instance downloads its
    own copy, and those can differ. Returns the number of failures."""
    picks = list({label["label_type"]: label for label in reversed(labels)}.values())
    print(f"\n=== download check: {len(picks)} labels fetched from Google ===")
    failures = 0
    for label in picks:
        a = call(args.a, key, label, args.timeout, use_cache=False)
        b = call(args.b, key, label, args.timeout, use_cache=False)
        b_ok = b[0] == 200 and isinstance(b[1], dict) and b[1].get("image_source") == "download"
        if b_ok:
            result = "ok"
        elif a[0] != 200:
            result = f"SKIP (A also failed with {a[0]}; panorama may be gone from Google)"
        else:
            source = f", image_source={b[1].get('image_source')}" if b[0] == 200 and isinstance(b[1], dict) else ""
            result = f"FAIL (A {a[0]}, B {b[0]}{source})"
            failures += 1
        print(f"  {label['label_type']:14} label {label['label_id']}: {result} (A {a[2]:.1f}s, B {b[2]:.1f}s)")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="CSV of label_id,label_type,pano_id,city_id,x,y")
    parser.add_argument("--a", default="http://127.0.0.1:60654/process", help="Baseline endpoint (default: prod)")
    parser.add_argument("--b", default="http://127.0.0.1:60655/process", help="Candidate endpoint (default: dev)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Largest score change (absolute) still counted as the same (default 1e-3)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds (default 300)")
    parser.add_argument("--max-slowdown", type=float, default=1.5,
                        help="Fail if B's median time on cached labels is more than this many times A's (default 1.5)")
    parser.add_argument("--out", help="Optional CSV path to save both full responses for every label")
    args = parser.parse_args()

    key = load_api_key()
    with open(args.labels) as f:
        labels = list(csv.DictReader(f))
    if not labels:
        raise SystemExit(f"No labels in {args.labels}")
    print(f"Comparing A={args.a} vs B={args.b} on {len(labels)} labels (tolerance {args.tolerance:g})\n")

    verdicts = Counter()
    max_deltas = defaultdict(float)  # label_type -> largest score change seen
    sources = Counter()
    times = {"A": [], "B": []}  # Response times on labels both served successfully from the cache.
    version_diffs = set()  # (field, A value, B value) for api_version / model ids that differ
    out_file = open(args.out, "w", newline="") if args.out else None
    writer = None
    if out_file:
        writer = csv.DictWriter(out_file, fieldnames=list(labels[0].keys()) + [
            "verdict", "notes", "seconds_a", "seconds_b", "status_a", "response_a", "status_b", "response_b"])
        writer.writeheader()
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
                if status == 200 and isinstance(resp, dict):
                    sources[f"{side}:{resp.get('image_source')}"] += 1
            if a[0] == b[0] == 200 and isinstance(a[1], dict) and isinstance(b[1], dict):
                for field in ("api_version", "validator_model_id", "validator_training_date",
                              "tagger_model_id", "tagger_training_date"):
                    if a[1].get(field) != b[1].get(field):
                        version_diffs.add((field, a[1].get(field), b[1].get(field)))

            from_google = any(isinstance(r, dict) and r.get("image_source") != "cache"
                              for s, r, _ in (a, b) if s == 200)
            if a[0] == b[0] == 200 and not from_google:
                times["A"].append(a[2])
                times["B"].append(b[2])
            line = (f"[{i}/{len(labels)}] {verdict:5} {label['label_type']:14} label {label['label_id']} "
                    f"(A {a[2]:.1f}s, B {b[2]:.1f}s){'  [image from Google, not cache]' if from_google else ''}")
            print(line + ("".join(f"\n      {n}" for n in notes) if verdict != "same" else ""))
            if writer:
                writer.writerow({**label, "verdict": verdict, "notes": "; ".join(notes),
                                 "seconds_a": f"{a[2]:.2f}", "seconds_b": f"{b[2]:.2f}",
                                 "status_a": a[0], "response_a": json.dumps(a[1]),
                                 "status_b": b[0], "response_b": json.dumps(b[1])})
                out_file.flush()
    if out_file:
        out_file.close()
        print(f"\nWrote full responses to {args.out}")

    download_failures = download_check(args, key, labels)

    print("\n=== summary ===")
    print("  " + ", ".join(f"{v}: {verdicts[v]}" for v in ("same", "DRIFT", "FAIL", "SKIP"))
          + f"; download check failures: {download_failures}")
    print("  largest score change by label type: "
          + ", ".join(f"{t} {d:.2e}" for t, d in sorted(max_deltas.items())))
    print("  image sources: " + ", ".join(f"{s} x{n}" for s, n in sorted(sources.items())))
    if sources.keys() - {"A:cache", "B:cache"}:
        print("  WARNING: some panoramas came from Google rather than the cache. Each instance downloads its own copy, and\n"
              "  Google's tiles can differ between downloads, so differences on those labels may be the image, not the code.\n"
              "  Check that the labels' city_id matches the cache's folder names (e.g. chicago-il).")
    too_slow = False
    if times["A"]:
        med_a, med_b = statistics.median(times["A"]), statistics.median(times["B"])
        ratio = med_b / med_a if med_a else float("inf")
        too_slow = not ratio <= args.max_slowdown
        print(f"  median time on {len(times['A'])} cached labels: A {med_a:.2f}s, B {med_b:.2f}s (B is {ratio:.2f}x A)")
        if too_slow:
            print(f"  SLOW: B is more than {args.max_slowdown:g}x slower than A. Check that B is using the GPU, e.g.\n"
                  "  docker exec sidewalk-ai-api-dev python -c 'import torch; print(torch.cuda.is_available())'\n"
                  "  and that nvidia-smi lists a process for each instance.")
    for field, va, vb in sorted(version_diffs, key=str):
        print(f"  note: {field} differs: A={va} B={vb}")
    if verdicts["SKIP"]:
        print(f"  {verdicts['SKIP']} label(s) errored on both instances and weren't compared. Every label is cached, so\n"
              "  that points at the setup (cache mount, instance down) rather than the change being tested.")

    passed = not verdicts["FAIL"] and not verdicts["SKIP"] and not download_failures and not too_slow
    print(f"\n{'PASS' if passed else 'NOT PASSED'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
