import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

BASE = "7ecfbb392285e03f73b4dd4cce62fc3fb8349c33"
CANDIDATE = "5e7a484a3fb1150e0facd3516c487325e6f901ea"

WORKER = r'''
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import kornia
from benchmarks import common
from kornia.geometry.depth import depth_to_3d, depth_to_3d_v2, depth_to_normals, unproject_meshgrid

mode, expected_sha, out_path = sys.argv[1:]
actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert actual_sha == expected_sha, (actual_sha, expected_sha)
assert Path(kornia.__file__).resolve() == Path("kornia/__init__.py").resolve()
print("PROVENANCE", sys.executable, torch.__version__, kornia.__file__, actual_sha, flush=True)
torch.set_num_threads(1)
torch.manual_seed(17)

def camera(batch, height, width, dtype):
    matrix = torch.tensor([[500.0, 0.0, width / 2], [0.0, 450.0, height / 2], [0.0, 0.0, 1.0]], dtype=dtype).repeat(batch, 1, 1)
    for b in range(batch):
        matrix[b, 0, 0] += b * 7
        matrix[b, 1, 1] += b * 11
        matrix[b, 0, 2] -= b * 3
        matrix[b, 1, 2] += b * 2
    return matrix

def digest(tensor):
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), sha256=hashlib.sha256(raw).hexdigest())

def workload(api, shape, normalize=False, cached=False):
    batch, height, width = shape
    matrix = camera(batch, height, width, torch.float32)
    depth = torch.linspace(0.5, 2.0, batch * height * width).reshape(batch, height, width)
    if api == "rays":
        return lambda: unproject_meshgrid(height, width, matrix, normalize, device=depth.device, dtype=depth.dtype)
    if api == "v1":
        return lambda: depth_to_3d(depth[:, None], matrix, normalize)
    if api == "normals":
        return lambda: depth_to_normals(depth[:, None], matrix, normalize)
    assert api == "v2"
    grid = unproject_meshgrid(height, width, matrix, normalize, device=depth.device, dtype=depth.dtype) if cached else None
    return lambda: depth_to_3d_v2(depth, matrix, normalize, grid)

metadata = common.run_metadata(torch.device("cpu"))
metadata.update(git_commit_full=actual_sha, executable=sys.executable, kornia_module=str(Path(kornia.__file__).resolve()))
metadata["cpu_model"] = next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
rows = []
if mode == "equivalence":
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for grid_dtype in (None, torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for batch, height, width in ((0, 3, 4), (1, 0, 4), (1, 3, 0), (2, 1, 5), (2, 5, 1), (3, 11, 17), (1, 1, 2051)):
                matrix = camera(batch, height, width, dtype)
                before = matrix.clone()
                for normalize in (False, True):
                    value = unproject_meshgrid(height, width, matrix, normalize, device=matrix.device, dtype=grid_dtype)
                    rows.append(dict(api="rays", camera_dtype=str(dtype), grid_dtype=str(grid_dtype), normalize=normalize, **digest(value)))
                assert torch.equal(matrix, before)
        for camera_dtype in dict.fromkeys((torch.float32, torch.float64, dtype)):
            for batch, height, width in ((0, 3, 4), (1, 0, 4), (1, 3, 0), (2, 1, 5), (2, 5, 1), (3, 11, 17), (1, 1, 2051)):
                matrix = camera(batch, height, width, camera_dtype)
                depth = torch.linspace(0.5, 2.0, batch * height * width).reshape(batch, 1, height, width).to(dtype)
                before_depth, before_matrix = depth.clone(), matrix.clone()
                for normalize in (False, True):
                    for api, value in (("v1", depth_to_3d(depth, matrix, normalize)), ("v2", depth_to_3d_v2(depth[:, 0], matrix, normalize))):
                        rows.append(dict(api=api, depth_dtype=str(dtype), camera_dtype=str(camera_dtype), normalize=normalize, **digest(value)))
                assert torch.equal(depth, before_depth) and torch.equal(matrix, before_matrix)
elif mode == "gradients":
    for dtype in (torch.float32, torch.float64):
        for normalize in (False, True):
            for api in ("v1", "v2"):
                matrix = camera(2, 3, 5, dtype).requires_grad_()
                depth = torch.linspace(0.5, 2.0, 30, dtype=dtype).reshape(2, 1, 3, 5).requires_grad_()
                value = depth_to_3d(depth, matrix, normalize).permute(0, 2, 3, 1) if api == "v1" else depth_to_3d_v2(depth[:, 0], matrix, normalize)
                weights = torch.linspace(0.25, 1.5, value.numel(), dtype=dtype).reshape_as(value)
                gradients = torch.autograd.grad((value.square() * weights).sum(), (depth, matrix))
                assert all(torch.isfinite(grad).all() for grad in gradients)
                rows.append(dict(api=api, dtype=str(dtype), normalize=normalize, depth_gradient=gradients[0].reshape(-1).tolist(), matrix_gradient=gradients[1].reshape(-1).tolist()))
elif mode == "timing":
    common.warm_up_cpu()
    cases = [
        ("tiny", "v2", (1, 3, 4), False, False),
        ("small-v1", "v1", (1, 64, 64), False, False),
        ("small-v2", "v2", (1, 64, 64), False, False),
        ("batched-vga-v1", "v1", (4, 480, 640), False, False),
        ("batched-vga-v2", "v2", (4, 480, 640), False, False),
        ("hd-v1", "v1", (1, 720, 1280), False, False),
        ("hd-v2", "v2", (1, 720, 1280), False, False),
        ("hd-v1-unit-rays", "v1", (1, 720, 1280), True, False),
        ("hd-v2-unit-rays", "v2", (1, 720, 1280), True, False),
        ("batched-vga-rays", "rays", (4, 480, 640), False, False),
        ("hd-rays", "rays", (1, 720, 1280), False, False),
        ("batched-vga-normals", "normals", (4, 480, 640), False, False),
        ("hd-normals", "normals", (1, 720, 1280), False, False),
        ("hd-cached-v2", "v2", (1, 720, 1280), False, True),
    ]
    for name, api, shape, normalize, cached in cases:
        fn = workload(api, shape, normalize, cached)
        output = digest(fn())
        median_us, iqr_us = common.time_us(fn, min_run_time=1.0)
        rows.append(dict(workload=name, api=api, shape=shape, normalize=normalize, cached=cached, median_us=median_us, iqr_us=iqr_us, output=output))
elif mode == "allocation":
    fn = workload("v2", (4, 480, 640))
    fn()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        output = fn()
    trace_path = str(Path(out_path).with_suffix(".trace.json"))
    prof.export_chrome_trace(trace_path)
    trace = json.loads(Path(trace_path).read_text())
    events = sorted((event for event in trace["traceEvents"] if event.get("name") == "[memory]"), key=lambda event: event["ts"])
    positive = [event["args"]["Bytes"] for event in events if event["args"].get("Bytes", 0) > 0]
    live = peak = 0
    for event in events:
        live += event["args"].get("Bytes", 0)
        peak = max(peak, live)
    rows.append(dict(workload="batched-vga-v2", positive_allocation_bytes=sum(positive), positive_allocations=len(positive), peak_net_tensor_bytes=peak, final_net_tensor_bytes=live, output=digest(output)))
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
    completed = subprocess.run([sys.executable, "-u", "-", mode, sha, str(output)], input=WORKER, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (results / f"{mode}-{variant}{suffix}.log").write_text(completed.stdout)
    print(completed.stdout, flush=True)
    assert completed.returncode == 0, (variant, mode, completed.returncode)
    return json.loads(output.read_text())

base = worker("base", "equivalence")
candidate = worker("candidate", "equivalence")
assert base["results"] == candidate["results"], "Public API output bits, dtype or shape differ"
print("EXACT DIFFERENTIAL CASES", len(base["results"]), flush=True)
gradients = {variant: worker(variant, "gradients") for variant in variants}
gradient_summary = []
for expected, actual in zip(gradients["base"]["results"], gradients["candidate"]["results"]):
    assert all(expected[key] == actual[key] for key in ("api", "dtype", "normalize"))
    tolerance = 2e-5 if expected["dtype"] == "torch.float32" else 2e-12
    row = {key: expected[key] for key in ("api", "dtype", "normalize")}
    for name in ("depth_gradient", "matrix_gradient"):
        assert all(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance) for a, b in zip(expected[name], actual[name]))
        row[name + "_max_abs_difference"] = max(abs(a - b) for a, b in zip(expected[name], actual[name]))
    gradient_summary.append(row)
print("GRADIENT DIFFERENTIAL CASES", len(gradient_summary), flush=True)
rounds = {"base": [], "candidate": []}
for i in range(5):
    for variant in (("base", "candidate") if i % 2 == 0 else ("candidate", "base")):
        rounds[variant].append(worker(variant, "timing", f"-{i}"))
summary = []
for index, first in enumerate(rounds["base"][0]["results"]):
    row = {key: first[key] for key in ("workload", "api", "shape", "normalize", "cached")}
    for variant in variants:
        cases = [run["results"][index] for run in rounds[variant]]
        assert all(case["output"] == first["output"] for case in cases), first["workload"]
        row[variant + "_us"] = [case["median_us"] for case in cases]
        assert all(math.isfinite(value) for value in row[variant + "_us"])
        row[variant + "_median_us"] = statistics.median(row[variant + "_us"])
    row["change_percent"] = 100 * (row["candidate_median_us"] / row["base_median_us"] - 1)
    summary.append(row)
allocations = {variant: worker(variant, "allocation") for variant in variants}
assert allocations["base"]["results"][0]["output"] == allocations["candidate"]["results"][0]["output"]
report = dict(base=BASE, candidate=CANDIDATE, exact_differential_cases=len(base["results"]), gradient_differential=gradient_summary, timing_rounds=5, timings=summary, allocations=allocations)
(results / "comparison-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
