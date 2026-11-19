import importlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import kornia
from benchmarks import common

BASE = "7ecfbb392285e03f73b4dd4cce62fc3fb8349c33"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == BASE
assert Path(kornia.__file__).resolve() == Path("kornia/__init__.py").resolve()
torch.set_num_threads(1)
depth_module = importlib.import_module("kornia.geometry.depth")
original = depth_module.unproject_meshgrid

def separable(height, width, camera_matrix, normalize_points=False, device=None, dtype=None):
    ramp_dtype = dtype if dtype is not None else torch.empty(0, device=device).dtype
    xs = torch.arange(width, device=device, dtype=ramp_dtype)
    ys = torch.arange(height, device=device, dtype=ramp_dtype)
    xs = (xs - camera_matrix[:, 0, 2, None]) / camera_matrix[:, 0, 0, None]
    ys = (ys - camera_matrix[:, 1, 2, None]) / camera_matrix[:, 1, 1, None]
    xx = xs[:, None, :].expand(-1, height, -1)
    yy = ys[:, :, None].expand(-1, -1, width)
    points = torch.stack((xx, yy, xs.new_ones(()).expand_as(xx)), dim=-1)
    if normalize_points:
        points = torch.nn.functional.normalize(points, dim=-1, p=2.0)
    return points

def camera(batch, height, width, dtype):
    matrix = torch.tensor([[500.0, 0.0, width / 2], [0.0, 450.0, height / 2], [0.0, 0.0, 1.0]], dtype=dtype).repeat(batch, 1, 1)
    return matrix

metadata = common.run_metadata(torch.device("cpu"))
metadata.update(git_commit_full=BASE, executable=sys.executable, kornia_module=kornia.__file__, prototype="separable camera rays; exploratory, not a final candidate source")
equivalence = []
for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
    for grid_dtype in (None, torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for batch, height, width in ((0, 3, 4), (1, 0, 4), (1, 3, 0), (2, 1, 5), (2, 5, 1), (3, 11, 17), (1, 1, 2051)):
            matrix = camera(batch, height, width, dtype)
            for normalize in (False, True):
                args = (height, width, matrix, normalize, torch.device("cpu"), grid_dtype)
                expected = original(*args)
                actual = separable(*args)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                equivalence.append((str(dtype), str(grid_dtype), batch, height, width, normalize))
print("EXPLORATORY EXACT CASES", len(equivalence), flush=True)

common.warm_up_cpu()
rows = []
for name, shape in (("small", (1, 64, 64)), ("medium", (1, 256, 256)), ("batched-vga", (4, 480, 640)), ("hd", (1, 720, 1280))):
    batch, height, width = shape
    matrix = camera(batch, height, width, torch.float32)
    depth = torch.full((batch, height, width), 2.0)
    for normalize in (False, True):
        for variant, implementation in (("base", original), ("prototype", separable)):
            depth_module.unproject_meshgrid = implementation
            fn = lambda: depth_module.depth_to_3d_v2(depth, matrix, normalize)
            median_us, iqr_us = common.time_us(fn, min_run_time=0.3)
            row = dict(workload=name, shape=shape, normalize=normalize, variant=variant, median_us=median_us, iqr_us=iqr_us)
            rows.append(row)
            print(json.dumps(row), flush=True)
depth_module.unproject_meshgrid = original
common.save_json("validation-results/probe-depth.json", metadata, rows)
