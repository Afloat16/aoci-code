import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

BASE = "7ecfbb392285e03f73b4dd4cce62fc3fb8349c33"
CANDIDATE = "8f1fa7740f3ddf4c53e9a65eb6e75ffee8859ebb"

WORKER = r'''
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import kornia
from benchmarks import common

mode, expected_sha, out_path = sys.argv[1:]
actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert actual_sha == expected_sha, (actual_sha, expected_sha)
assert Path(kornia.__file__).resolve() == Path("kornia/__init__.py").resolve()
print("PROVENANCE", sys.executable, torch.__version__, kornia.__file__, actual_sha, flush=True)
torch.set_num_threads(1)
torch.manual_seed(0)

def dataset(images, objects, detections, classes, dtype=torch.float32, seed=0, variable=False):
    gen = torch.Generator().manual_seed(seed)
    predictions, labels, scores, targets, target_labels = [], [], [], [], []
    for i in range(images):
        n = (objects + i * 7) % (objects + 1) if variable else objects
        m = (detections + i * 11) % (detections + 1) if variable else detections
        idx = torch.arange(n)
        xy = torch.stack([(idx % 25) * 20, (idx // 25) * 20], 1).float()
        boxes = torch.cat([xy, xy + 10], 1)
        gt_labels = 1 + idx % classes
        if n:
            chosen = torch.randint(n, (m,), generator=gen)
            pred = boxes[chosen].clone()
            pred_labels = gt_labels[chosen].clone()
            pred[torch.rand(m, generator=gen) < 0.25] += 1024
        else:
            pred = torch.zeros(m, 4)
            pred[:, 2:] = 10
            pred_labels = 1 + torch.arange(m) % classes
        confidence = torch.rand(m, generator=gen)
        predictions.append(pred.to(dtype))
        labels.append(pred_labels)
        scores.append(confidence.to(dtype))
        targets.append(boxes.to(dtype))
        target_labels.append(gt_labels)
    return predictions, labels, scores, targets, target_labels, classes + 1

def result(args):
    ap, per_class = kornia.metrics.mean_average_precision(*args)
    return {"dtype": str(ap.dtype), "shape": list(ap.shape), "value": ap.item(), "per_class": per_class}

metadata = common.run_metadata(torch.device("cpu"))
metadata["git_commit_full"] = actual_sha
metadata["executable"] = sys.executable
metadata["kornia_module"] = str(Path(kornia.__file__).resolve())
rows = []
if mode == "equivalence":
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for seed in range(20):
            args = dataset(1 + seed % 7, 3 + seed % 6, 4 + seed % 8, 1 + seed % 3, dtype, seed, True)
            for reverse in (False, True):
                selected = [list(reversed(x)) for x in args[:-1]] + [args[-1]] if reverse else args
                before = [x.clone() for tensor_list in selected[:-1] for x in tensor_list]
                value = result(selected)
                after = [x for tensor_list in selected[:-1] for x in tensor_list]
                assert all(torch.equal(a, b) for a, b in zip(before, after))
                rows.append({"seed": seed, "reverse": reverse, **value})
elif mode == "timing":
    common.warm_up_cpu()
    for name, shape in (
        ("single-image", (1, 20, 40, 1)),
        ("16-images", (16, 20, 40, 1)),
        ("128-images", (128, 40, 80, 1)),
        ("dense-single-image", (1, 1000, 2000, 1)),
        ("64-images-four-classes", (64, 32, 64, 4)),
    ):
        args = dataset(*shape, seed=17)
        expected = result(args)
        median_us, iqr_us = common.time_us(lambda: kornia.metrics.mean_average_precision(*args), min_run_time=1.0)
        rows.append({"workload": name, "shape": shape, "median_us": median_us, "iqr_us": iqr_us, "output": expected})
elif mode == "allocation":
    args = dataset(16, 32, 64, 1, seed=17)
    result(args)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        expected = result(args)
    trace_path = str(Path(out_path).with_suffix(".trace.json"))
    prof.export_chrome_trace(trace_path)
    trace = json.loads(Path(trace_path).read_text())
    events = [e for e in trace["traceEvents"] if e.get("name") == "[memory]"]
    positive = [e["args"]["Bytes"] for e in events if e["args"].get("Bytes", 0) > 0]
    live = peak = 0
    for e in events:
        live += e["args"].get("Bytes", 0)
        peak = max(peak, live)
    rows.append({"workload": "16-images-allocation", "positive_allocation_bytes": sum(positive),
                 "positive_allocations": len(positive), "peak_net_tensor_bytes": peak, "output": expected})
else:
    raise ValueError(mode)
common.save_json(out_path, metadata, rows)
print("RESULT", out_path, len(rows), flush=True)
'''

root = Path.cwd()
results = root / "validation-results"
results.mkdir(exist_ok=True)
variants = {"base": (root / "baseline-source", BASE), "candidate": (root, CANDIDATE)}

def worker(variant, mode, suffix=""):
    cwd, sha = variants[variant]
    output = results / f"{mode}-{variant}{suffix}.json"
    completed = subprocess.run([sys.executable, "-u", "-", mode, sha, str(output)],
                               input=WORKER, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (results / f"{mode}-{variant}{suffix}.log").write_text(completed.stdout)
    print(completed.stdout, flush=True)
    assert completed.returncode == 0, (variant, mode, completed.returncode)
    return json.loads(output.read_text())

base = worker("base", "equivalence")
candidate = worker("candidate", "equivalence")
assert base["results"] == candidate["results"], "Public API outputs differ"
print("EXACT DIFFERENTIAL CASES", len(base["results"]), flush=True)
rounds = {"base": [], "candidate": []}
for i in range(5):
    for variant in (("base", "candidate") if i % 2 == 0 else ("candidate", "base")):
        rounds[variant].append(worker(variant, "timing", f"-{i}"))
summary = []
for index, first in enumerate(rounds["base"][0]["results"]):
    row = {"workload": first["workload"], "shape": first["shape"]}
    for variant in variants:
        cases = [r["results"][index] for r in rounds[variant]]
        assert all(c["output"] == first["output"] for c in cases)
        row[variant + "_us"] = [c["median_us"] for c in cases]
        row[variant + "_median_us"] = statistics.median(row[variant + "_us"])
    row["change_percent"] = 100 * (row["candidate_median_us"] / row["base_median_us"] - 1)
    summary.append(row)
allocations = {variant: worker(variant, "allocation") for variant in variants}
assert allocations["base"]["results"][0]["output"] == allocations["candidate"]["results"][0]["output"]
report = {"base": BASE, "candidate": CANDIDATE, "exact_differential_cases": len(base["results"]),
          "timing_rounds": 5, "timings": summary, "allocations": allocations}
(results / "comparison-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
