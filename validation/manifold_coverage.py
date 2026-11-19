import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

variant, expected_sha = sys.argv[1:]
assert variant in ("baseline", "candidate")
root = Path.cwd()
results = root.parent / "coverage-evidence"
results.mkdir(exist_ok=True)
summary = {"variant": variant, "source": expected_sha, "commands": []}

def run(name, command, cwd=root, check=True):
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (results / (name + ".log")).write_text(completed.stdout)
    print("$ " + " ".join(command), flush=True)
    print(completed.stdout, flush=True)
    summary["commands"].append({"name": name, "argv": command,
                                "cwd": str(cwd), "returncode": completed.returncode})
    if check and completed.returncode:
        raise RuntimeError(f"{name} exited {completed.returncode}")
    return completed

try:
    actual = run("source", ["git", "rev-parse", "HEAD"]).stdout.strip()
    assert actual == expected_sha, (actual, expected_sha)
    run("diff-check", ["git", "diff", "--check"])
    run("compiler", ["g++-14", "--version"])
    run("gcov-version", ["gcov-14", "--version"])
    run("lcov-version", ["lcov", "--version"])
    run("cmake-version", ["cmake", "--version"])
    run("packages", ["dpkg-query", "-W", "gcc-14", "g++-14", "lcov",
                      "libgtest-dev", "libtbb-dev", "cmake"])
    run("configure", [
        "cmake", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DBUILD_SHARED_LIBS=ON",
        "-DMANIFOLD_STRICT=ON", "-DMANIFOLD_PYBIND=OFF", "-DMANIFOLD_DEBUG=ON",
        "-DMANIFOLD_ASSERT=ON", "-DMANIFOLD_PAR=OFF", "-DCODE_COVERAGE=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", ".", "-B", "build",
    ])
    run("build", ["cmake", "--build", "build"])
    build = root / "build"
    commands = json.loads((build / "compile_commands.json").read_text())
    instrumented = [row for row in commands if "-coverage" in row["command"]]
    assert instrumented, "No coverage-instrumented translation units"
    for row in instrumented:
        assert ("-fprofile-update=atomic" in row["command"]) == (variant == "candidate"), row
    summary["instrumented_translation_units"] = len(instrumented)
    shutil.copy2(build / "compile_commands.json", results / "compile_commands.json")
    shutil.copy2(build / "CMakeCache.txt", results / "CMakeCache.txt")
    run("capture-initial", [
        "lcov", "--capture", "--gcov-tool", "gcov-14", "--ignore-errors", "mismatch",
        "--initial", "--directory", ".", "--output-file", "./code_coverage_init.info",
    ], build)
    tests = run("tests", ["test/manifold_test"], build)
    counts = re.findall(r"\[  PASSED  \] (\d+) tests?\.", tests.stdout)
    assert counts == ["693"], counts
    assert "[       OK ] Manifold.SharedTopologyConcurrentCopies" in tests.stdout
    summary["tests_passed"] = int(counts[0])
    summary["concurrent_test_passed"] = True
    capture = run("capture-tests", [
        "lcov", "--capture", "--gcov-tool", "gcov-14", "--ignore-errors", "mismatch",
        "--directory", ".", "--output-file", "./code_coverage_test.info",
    ], build, check=False)
    summary["capture_returncode"] = capture.returncode
    if variant == "baseline":
        assert capture.returncode != 0, "Baseline did not reproduce coverage failure"
        errors = re.findall(r"geninfo: ERROR: Unexpected negative count[^\n]+", capture.stdout)
        assert errors, "Baseline failed without the expected negative coverage count"
        summary["baseline_negative_errors"] = errors
        print("Confirmed baseline negative coverage count:", errors, flush=True)
    else:
        assert capture.returncode == 0, "Candidate coverage capture failed"
        run("merge-traces", [
            "lcov", "--add-tracefile", "./code_coverage_init.info",
            "--add-tracefile", "./code_coverage_test.info",
            "--output-file", "./code_coverage_total.info",
        ], build)
        run("extract-source", [
            "lcov", "--extract", "./code_coverage_total.info", "/*/manifold/src/*",
            "--output-file", "./code_coverage.info",
        ], build)
        run("coverage-summary", ["lcov", "--summary", "./code_coverage.info"], build)
        coverage = (build / "code_coverage.info").read_text()
        assert "SF:" in coverage and "DA:" in coverage
        assert all(int(line.split(",")[1]) >= 0
                   for line in coverage.splitlines() if line.startswith("DA:"))
        totals = {key: sum(int(line.split(":")[1]) for line in coverage.splitlines()
                           if line.startswith(key + ":")) for key in ("LF", "LH", "FNF", "FNH")}
        summary["source_coverage"] = totals
        summary["coverage_source_files"] = sum(line.startswith("SF:") for line in coverage.splitlines())
        summary["complete_coverage_report"] = True
    run("unchanged-source", ["git", "diff", "--exit-code"])
    summary["validated"] = True
finally:
    build = root / "build"
    for source in build.glob("code_coverage*.info"):
        shutil.copy2(source, results / source.name)
    for suffix in (".gcno", ".gcda"):
        source = build / ("src/CMakeFiles/manifold.dir/subdivision.cpp" + suffix)
        if source.exists():
            shutil.copy2(source, results / source.name)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
