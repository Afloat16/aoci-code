import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
import kornia

ROOT = Path.cwd().resolve()
assert Path(kornia.__file__).resolve() == ROOT / "kornia/__init__.py"
assert ".venv-py312" in sys.executable
print("PROVENANCE", sys.executable, kornia.__file__, torch.__version__, flush=True)
torch.set_num_threads(1)
torch.manual_seed(0)
spec = importlib.util.spec_from_file_location("base_cc", ROOT / "validation-results/connected_components-base.py")
baseline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_module)
base = baseline_module.connected_components
candidate = kornia.contrib.connected_components
count = 0
for dtype in (torch.bool, torch.uint8, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64):
    seed = torch.tensor([[[[1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]]]], dtype=dtype)
    expected = torch.tensor([[[[5, 0, 0, 4], [5, 0, 0, 0], [0, 0, 12, 12]]]], dtype=torch.int64)
    torch.testing.assert_close(candidate(seed), expected, rtol=0, atol=0)
    shapes = [
        ("contiguous", seed),
        ("spatial-transpose", seed.transpose(-1, -2)),
        ("expanded-batch", seed.expand(3, -1, -1, -1)),
        ("permuted-leading-batch", seed.expand(2, 3, 1, 3, 4).permute(1, 0, 2, 3, 4)),
        ("empty-height", seed[..., :0, :]),
    ]
    for name, value in shapes:
        for iterations in (1, 2, 100):
            before = value.clone()
            version = value._version
            expected = base(value, iterations)
            actual = candidate(value, iterations)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value, before, rtol=0, atol=0)
            assert value._version == version
            assert not actual.requires_grad and actual.dtype == torch.int64
            count += 1
    if dtype.is_floating_point:
        value = seed.clone().requires_grad_()
        before = value.clone()
        actual = candidate(value, 5)
        torch.testing.assert_close(actual, base(value, 5), rtol=0, atol=0)
        torch.testing.assert_close(value, before, rtol=0, atol=0)
        assert not actual.requires_grad
        count += 1
for iterations in (1, 5, 100):
    generator = torch.Generator().manual_seed(0)
    value = (torch.rand((3, 2, 1, 7, 9), generator=generator) < 0.35).float()
    actual = torch.vmap(lambda image: candidate(image, iterations))(value)
    expected = torch.vmap(lambda image: base(image, iterations))(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    functional = torch.func.functionalize(candidate)
    torch.testing.assert_close(functional(value, iterations), base(value, iterations), rtol=0, atol=0)
    count += 2
jit = torch.jit.script(candidate)
compiled = torch.compile(candidate, fullgraph=True)
for shape in ((2, 1, 7, 9), (3, 1, 7, 9)):
    value = (torch.rand(shape) < 0.35).float()
    before = value.clone()
    expected = base(value, 5)
    torch.testing.assert_close(jit(value, 5), expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled(value, 5), expected, rtol=0, atol=0)
    torch.testing.assert_close(value, before, rtol=0, atol=0)
    count += 2
payload = {"differential_cases": count, "independent_literal_cases": 7, "torch": torch.__version__,
           "base_sha": "7ecfbb392285e03f73b4dd4cce62fc3fb8349c33", "candidate_sha": "0d684f8ff301861768f16c8fe058942d34a8e253",
           "source_sha256": hashlib.sha256(Path("kornia/contrib/connected_components.py").read_bytes()).hexdigest()}
suffix = "minimum" if torch.__version__.startswith("2.5.1") else "current"
Path(f"validation-results/compatibility-{suffix}.json").write_text(json.dumps(payload, indent=2) + "\n")
print("COMPATIBILITY", payload, flush=True)
