import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
import torch
import kornia
from benchmarks.common import run_metadata, save_json, time_us, warm_up_cpu

assert Path(kornia.__file__).resolve() == ROOT / "kornia/__init__.py"
assert Path(sys.executable).parent.name == "bin" and ".venv-py312" in sys.executable
print("PROVENANCE", sys.executable, kornia.__file__, torch.__version__, flush=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
if hasattr(os, "sched_getaffinity"):
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
warm_up_cpu()

spec = importlib.util.spec_from_file_location("baseline_connected_components", ROOT / "validation-results/connected_components-base.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
functions = {"base": module.connected_components, "candidate": kornia.contrib.connected_components}
mid_spec = importlib.util.spec_from_file_location("multiply_connected_components", ROOT / "validation-results/connected_components-multiply.py")
mid_module = importlib.util.module_from_spec(mid_spec)
mid_spec.loader.exec_module(mid_module)
functions["inplace_multiply"] = mid_module.connected_components
metadata = run_metadata(torch.device("cpu"))
metadata.update(base_sha="7ecfbb392285e03f73b4dd4cce62fc3fb8349c33", candidate_sha="17d9f6273e612e89b60162a20e6d0a00bb5abe50", seed=0,
                comparison="Public connected_components functions, exact complete source modules, same process alternating order",
                rounds=5, min_run_time_seconds=0.3,
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
metadata["source_sha256"] = {
    "base": hashlib.sha256((ROOT / "validation-results/connected_components-base.py").read_bytes()).hexdigest(),
    "candidate": hashlib.sha256((ROOT / "kornia/contrib/connected_components.py").read_bytes()).hexdigest(),
}
results = []
generator = torch.Generator().manual_seed(0)
cases = []
for size, batch in ((256, 1), (1024, 1), (256, 4)):
    mask = (torch.rand((batch, 1, size, size), generator=generator) < 0.35).float()
    cases.append((f"random-{size}-b{batch}", mask, 100))
image_path = ROOT / "docs/source/_static/img/models/loftr.jpg"
if image_path.exists():
    import numpy as np
    from PIL import Image
    array = np.array(Image.open(image_path).convert("L").resize((512, 512)))
    mask = torch.from_numpy(array < 128).float().unsqueeze(0).unsqueeze(0)
    cases.append(("loftr-threshold-512", mask, 100))
    metadata["image"] = {"path": str(image_path.relative_to(ROOT)),
                         "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                         "transformation": "PIL convert L, resize 512x512 default, threshold <128"}
for name, value, iterations in cases:
    snapshot = value.clone()
    expected = functions["base"](value, iterations)
    actual = functions["candidate"](value, iterations)
    assert torch.equal(expected, actual) and actual.dtype == torch.int64
    assert torch.equal(value, snapshot)
    calls = {revision: (lambda fn=fn: fn(value, iterations)) for revision, fn in functions.items()}
    times = {revision: [] for revision in functions}
    for round_index in range(metadata["rounds"]):
        keys = list(functions)
        shift = round_index % len(keys)
        order = keys[shift:] + keys[:shift]
        for revision in order:
            median, iqr = time_us(calls[revision], min_run_time=metadata["min_run_time_seconds"])
            assert median > 0 and iqr >= 0
            times[revision].append({"round": round_index, "median_us": median, "iqr_us": iqr})
    allocation = {}
    for revision in functions:
        gc.collect()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            out = calls[revision]()
        assert torch.equal(out, expected)
        trace_path = Path(f"validation-results/{name}-{revision}-trace.json")
        prof.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text())["traceEvents"]
        memory = [e["args"] for e in events if e.get("name") == "[memory]" and e.get("args", {}).get("Device Type") == 0]
        assert memory and all("Bytes" in e and "Total Allocated" in e for e in memory)
        baseline_live = memory[0]["Total Allocated"] - memory[0]["Bytes"]
        allocation[revision] = {
            "positive_allocation_bytes": sum(e["Bytes"] for e in memory if e["Bytes"] > 0),
            "allocation_count": sum(e["Bytes"] > 0 for e in memory),
            "peak_extra_live_tensor_bytes": max(e["Total Allocated"] for e in memory) - baseline_live,
            "baseline_live_tensor_bytes": baseline_live,
            "trace_memory_events": len(memory),
            "trace_file": str(trace_path),
        }
        del out
        print("ALLOCATION", name, revision, allocation[revision], flush=True)
    row = {"case": name, "shape": list(value.shape), "iterations": iterations, "exact_equal": True,
           "input_unchanged": torch.equal(value, snapshot), "timings": times, "allocation": allocation}
    row["inplace_multiply_us"] = statistics.median(t["median_us"] for t in times["inplace_multiply"])
    row["base_us"] = statistics.median(t["median_us"] for t in times["base"])
    row["candidate_us"] = statistics.median(t["median_us"] for t in times["candidate"])
    row["relative_time_change_percent"] = 100 * (row["candidate_us"] / row["base_us"] - 1)
    row["positive_allocation_byte_reduction"] = allocation["base"]["positive_allocation_bytes"] - allocation["candidate"]["positive_allocation_bytes"]
    results.append(row)
    save_json("validation-results/exploratory.json", metadata, results)
    print(json.dumps(row), flush=True)
print("EXPLORATORY COMPLETE", flush=True)
