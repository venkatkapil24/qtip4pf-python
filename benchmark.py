#!/usr/bin/env -S uv run --locked --script
# /// script
# requires-python = "==3.12.2"
# dependencies = [
#   "numpy==2.4.4",
#   "pytest==7.4.4",
#   "jax-triton==0.3.1; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "absl-py==2.3.1; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "scipy==1.17.1",
#   "matplotlib==3.11.2",
#   "torch==2.10.0+cu130; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "jax[cuda13]==0.10.0; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "openmm[cuda13]==8.6.1; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "array-api-compat==1.16.0; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "ase==3.26.0; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "ipi @ https://github.com/i-pi/i-pi/archive/1d3682e90584b2e072feb95c9aaad510a3367475.tar.gz ; sys_platform == 'linux' and platform_machine == 'x86_64'",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Run the 64-water comparison: uv run --locked benchmark.py.

The adjacent lockfile pins all Python dependencies. i-PI source trees are
additionally pinned by commit, archive SHA256 and a checked source-tree digest.
The installed i-PI distribution supplies the PR dependencies; isolated worker
processes explicitly import the appropriate pinned main or PR source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import urlopen

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE
METHODS = ("pure_torch", "pure_jax", "openmm", "main_torch", "main_jax", "pr_torch", "pr_jax")
PRECISIONS = ("mixed", "double")
REPLICAS = (1, 2, 4, 8, 16, 32)


CONFIG = {'structure': 'data/water64.xyz',
 'force_field': 'data/qtip4pf.xml',
 'replicas': [1, 2, 4, 8, 16, 32],
 'temperature_K': 300.0,
 'dt_fs': 0.5,
 'tau_fs': 100.0,
 'cutoff_nm': 0.6,
 'ewald_tolerance': 0.0005,
 'dispersion_correction': True,
 'seed': 12345,
 'steps_per_sample': 100,
 'repeats': 5,
 'warmup_steps': 20,
 'safe_stride': 100,
 'openmm_blocking_sync': False,
 'openmm_device_index': '0',
 'precision_modes': ['mixed', 'double'],
 'methods': ['pure_torch',
             'pure_jax',
             'openmm',
             'main_torch',
             'main_jax',
             'pr_torch',
             'pr_jax']}
SOURCES = {'main': {'commit': '93162d280d15fd954da6dc3580a5ed3503e207ac',
          'url': 'https://codeload.github.com/i-pi/i-pi/tar.gz/93162d280d15fd954da6dc3580a5ed3503e207ac',
          'archive_sha256': 'b34312d0afde8bdb4c2ef1e866a89b54651d679c9f190c1159cb3dc94054136b'},
 'pr': {'commit': '1d3682e90584b2e072feb95c9aaad510a3367475',
        'pull_requests': [608, 609],
        'note': 'Replacement next PR plus latest bugfix PR; superseded '
                '602/605/606/607 excluded.',
        'url': 'https://codeload.github.com/i-pi/i-pi/tar.gz/1d3682e90584b2e072feb95c9aaad510a3367475',
        'archive_sha256': '7805d1a7fb32aa19798ccaf45f15a27ab4b7e777cda0a85e377462493b69fd72'}}
CUDA_BARRIER = '// Timing helper only: synchronize the actual CUDA context owned by OpenMM.\n// No integration or force code is changed, and no particle arrays are copied.\n#include <openmm/Context.h>\n#include <openmm/Platform.h>\n#include <openmm/internal/ContextImpl.h>\n#include <openmm/cuda/CudaContext.h>\n#include <openmm/cuda/CudaPlatform.h>\n\nstruct PlatformAccess : OpenMM::Platform {\n    static OpenMM::ContextImpl& implementation(OpenMM::Context& context) {\n        using Getter = OpenMM::ContextImpl& (OpenMM::Platform::*)(OpenMM::Context&) const;\n        Getter getter = &PlatformAccess::getContextImpl;\n        return (context.getPlatform().*getter)(context);\n    }\n};\n\nextern "C" int synchronize_openmm(OpenMM::Context* context) {\n    try {\n        auto& impl = PlatformAccess::implementation(*context);\n        auto* data = static_cast<OpenMM::CudaPlatform::PlatformData*>(impl.getPlatformData());\n        for (auto* gpu : data->contexts) {\n            CUresult result = cuCtxPushCurrent(gpu->getContext());\n            if (result != CUDA_SUCCESS) return int(result);\n            result = cuCtxSynchronize();\n            CUcontext popped;\n            CUresult popResult = cuCtxPopCurrent(&popped);\n            if (result != CUDA_SUCCESS) return int(result);\n            if (popResult != CUDA_SUCCESS) return int(popResult);\n        }\n        return 0;\n    } catch (...) {\n        return -1;\n    }\n}\n'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_digest(path):
    files = sorted(Path(path).rglob("*.py")) + [Path(path) / "bin/i-pi", Path(path) / "pyproject.toml"]
    digest = hashlib.sha256()
    for file in files:
        if "__pycache__" in file.parts:
            continue
        digest.update(str(file.relative_to(path)).encode() + b"\0")
        digest.update(file.read_bytes())
    return digest.hexdigest()


def fetch_source(destination, specification, downloads):
    """Cache immutable, verified source archives; reject locally edited trees."""
    marker = destination / ".benchmark-source.json"
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError(f"Incomplete source tree: {destination}. Choose a new --output directory.")
        stored = json.loads(marker.read_text())
        if any(stored.get(k) != specification[k] for k in ("commit", "url", "archive_sha256")) or stored["tree_sha256"] != source_digest(destination):
            raise RuntimeError(f"Pinned source tree was modified: {destination}")
        return stored
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / (specification["commit"] + ".tar.gz")
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        print(f"Downloading i-PI {specification['commit'][:12]}", flush=True)
        with urlopen(specification["url"], timeout=30) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(temporary) != specification["archive_sha256"]:
            temporary.unlink()
            raise RuntimeError("i-PI archive checksum mismatch")
        temporary.replace(archive)
    if sha256(archive) != specification["archive_sha256"]:
        raise RuntimeError(f"Cached archive checksum mismatch: {archive}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        with tarfile.open(archive) as source:
            source.extractall(temporary, filter="data")
        roots = list(Path(temporary).iterdir())
        if len(roots) != 1 or not (roots[0] / "bin/i-pi").is_file():
            raise RuntimeError("Unexpected i-PI archive layout")
        shutil.move(str(roots[0]), destination)
    record = dict(specification, tree_sha256=source_digest(destination))
    write_json(marker, record)
    return record


def prepare(output):
    work = output / ".work"
    work.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG)
    for key in ("structure", "force_field"):
        config[key] = str(PACKAGE / config[key])
    config["uv_lock_sha256"] = sha256(PACKAGE / "benchmark.py.lock")
    write_json(work / "config.json", config)
    sources = {name: fetch_source(work / "sources" / name, SOURCES[name], work / "downloads")
               for name in ("main", "pr")}
    return PACKAGE, sources


def environment(gpu, output, kernels="triton"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["QTIP4PF_KERNELS"] = kernels
    env["PYTHONPATH"] = str(PACKAGE) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLCONFIGDIR"] = str(output / ".matplotlib")
    # WSL2's driver library is outside the default C++ linker search path.
    if Path("/usr/lib/wsl/lib").is_dir():
        env["LIBRARY_PATH"] = "/usr/lib/wsl/lib" + os.pathsep + env.get("LIBRARY_PATH", "")
    return env


def preflight(worker, env, output):
    """Use a child so no parent GPU allocations survive into the timings."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The GPU benchmark requires Linux x86_64 or WSL2 with an NVIDIA GPU; native Windows/macOS are not supported.")
    compiler = shlex.split(env.get("CXX", "c++"))[0]
    if not shutil.which(compiler):
        raise RuntimeError("A C++17 compiler is required for the no-download OpenMM CUDA timer (e.g. install build-essential on Ubuntu).")
    code = """
import importlib.metadata as m,json
from benchmark import build_barrier
import torch,jax,openmm
assert torch.cuda.is_available(), 'Torch cannot access the NVIDIA GPU'
assert jax.devices('gpu'), 'JAX cannot access the NVIDIA GPU'
assert 'CUDA' in [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())], 'OpenMM CUDA plugin missing'
assert build_barrier().is_file()
packages=('torch','jax','jaxlib','numpy','scipy','matplotlib','array-api-compat','openmm','ipi','triton','jax-triton','absl-py')
print('ENVIRONMENT_JSON='+json.dumps(dict(versions={p:m.version(p) for p in packages},gpu=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda)))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=worker, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "preflight.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"GPU/timing setup failed; see {output / 'preflight.log'}\n{result.stdout[-3000:]}")
    line = next(line for line in result.stdout.splitlines() if line.startswith("ENVIRONMENT_JSON="))
    return json.loads(line.partition("=")[2])


def render(output):
    results = output / "results" / "summary.json"
    if not results.is_file():
        raise RuntimeError(f"No saved measurements: {results}")
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    plot(results, output / "plots")
    summary = json.loads(results.read_text())
    rows = summary["records"]
    if rows:
        r = rows[0]
        gpu = next((row.get("gpu") for row in rows if row.get("gpu")), "NVIDIA GPU")
        modes={row['precision'] for row in rows}
        precision=("Both panels use fp64 integration. Left: fp32 force arithmetic; right: fp64 force arithmetic. "
                   if len(modes)==2 else
                   "Force arithmetic is fp32; integration is fp64. " if modes=={'mixed'} else
                   "Force arithmetic and integration are fp64. ")
        kernels = sorted({row.get("potential_kernels","array") for row in rows if row["method"]!="openmm"})
        caption = (
            f"{r['nwater']} q-TIP4P/F waters; NVT/PILE-L at {r['temperature_K']:g} K, timestep {r['dt_fs']:g} fs, centroid thermostat time {r['tau_fs']:g} fs, and COM removal. Ordinary Ewald: cutoff {r['cutoff_nm']:g} nm, tolerance {r['ewald_tolerance']:g}, with dispersion correction.\n"
            + "Potential kernels: " + ", ".join(kernels) + ". " + precision + "OpenMM thermostat noise remains fp32 and force accumulation uses 64-bit fixed-point integers.\n"
            "All Torch/JAX potentials are compiled. Pure/PR Torch uses torch.compile and CUDA graphs (segmented in the PR); pure/PR JAX uses jax.jit. i-PI main uses NumPy integration. Native OpenMM requests forces; Torch/JAX calculate energy, forces and full virials.\n"
            f"Hardware: {gpu}. Median of {r['repeats']} synchronized {r['steps_per_sample']}-step blocks after compilation and {r['warmup_steps']} warm-up steps; error bars show the sample range.\n"
        )
        (output / "plots/caption.txt").write_text(caption.encode("ascii", "replace").decode())
    print(f"Figure: {output / 'plots/throughput.png'}", flush=True)
    print(f"Caption: {output / 'plots/caption.txt'}", flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "benchmark-results")
    parser.add_argument("--gpu", default="0", help="Physical GPU index (default: 0)")
    parser.add_argument("--kernels", choices=("array","triton"), default="triton", help="Potential implementation (default: shared Triton kernels)")
    parser.add_argument("--resume", action="store_true", help="Reuse completed cases if code, lockfile and settings match")
    parser.add_argument("--smoke", action="store_true", help="Short execution check at P=2; seven cases per precision")
    parser.add_argument("--replicas", nargs="+", type=int, choices=REPLICAS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--precision", nargs="+", choices=PRECISIONS, default=["mixed"])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--example", choices=("torch","jax"), help="Run one FFDirect example")
    modes.add_argument("--validate", action="store_true", help="Run independent GPU validation tests")
    modes.add_argument("--plan", action="store_true", help="Print requested cases and source pins without preparing or running MD")
    modes.add_argument("--prepare-only", action="store_true", help="Fetch pinned sources and check GPU/timer setup, without MD")
    modes.add_argument("--plot-only", action="store_true", help="Regenerate the figure and ASCII caption from saved results; no GPU required")
    args = parser.parse_args()
    args.replicas = args.replicas or ([2] if args.smoke else list(REPLICAS))
    return args


def main():
    args = arguments()
    output = args.output.resolve()
    expected = {(mode, method, p) for mode in args.precision for method in args.methods for p in args.replicas}
    if args.plan:
        print(json.dumps(dict(cases=len(expected),replicas=args.replicas,methods=args.methods,
                              precision=args.precision,kernels=args.kernels,sources=SOURCES),indent=2))
        return 0
    if args.plot_only:
        render(output)
        return 0
    if args.validate:
        return subprocess.run([sys.executable,"-m","pytest",str(PACKAGE/"tests")],
                              cwd=PACKAGE,env=environment(args.gpu,output,args.kernels)).returncode
    if args.example:
        command=[str(Path(sys.executable).with_name("i-pi")),
                 str(PACKAGE/"examples"/f"ffdirect-{args.example}.xml"),
                 "--backend",args.example,"--device","cuda","--jit"]
        return subprocess.run(command,cwd=PACKAGE,env=environment(args.gpu,output,args.kernels)).returncode
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Run the GPU comparison on Linux x86_64 or WSL2 with an NVIDIA GPU.")
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker, sources = prepare(output)
        env = environment(args.gpu, output,args.kernels)
        runtime = preflight(worker, env, output)
        write_json(output / "environment.json", dict(runtime, python=sys.version, source_pins=sources,
                   uv_lock_sha256=sha256(PACKAGE / "benchmark.py.lock")))
        if args.prepare_only:
            print(f"Pinned environment and sources ready: {output}")
            return 0
        run_cases(argparse.Namespace(config=output/".work/config.json", output=output/"results",
            source_main=output/".work/sources/main", source_pr=output/".work/sources/pr",
            gpu=str(args.gpu), replicas=args.replicas, methods=args.methods, precision=args.precision,
            kernels=args.kernels, smoke=args.smoke, resume=args.resume))
        summary = json.loads((output / "results/summary.json").read_text())
        completed = {(r['precision'],r['method'],r['nbeads']) for r in summary['records']}
        failures = [r for r in summary['failed'] if (r['precision'],r['method'],r['nbeads']) in expected]
        missing = sorted(expected - completed)
        write_json(output / "completion.json", dict(requested_cases=len(expected),completed_cases=len(expected & completed),failed=failures,missing=missing,smoke=args.smoke))
        render(output)
        if failures or missing:
            print(f"Incomplete: {len(failures)} failed, {len(missing)} missing. See completion.json and per-case run.log files.",file=sys.stderr)
            return 1
        print(f"Completed all {len(expected)} requested cases.")
        return 0


from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys



MASS_DA = {"O": 15.99943, "H": 1.007947}
# Same recorded i-PI unit definitions as the validated functional reference.
UNITS = dict(angstrom=1.8897261, femtosecond=41.341373000000004,
             kelvin=3.1668152e-6, amu=1822.8885)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def load_config(path, overrides=None):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if overrides:
        config.update({key:value for key,value in overrides.items() if value is not None})
    for key in ("structure", "force_field"):
        config[key] = str((path.parent / config[key]).resolve())
        if not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}: {config[key]}")
    for key in ("steps_per_sample", "repeats", "safe_stride"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["warmup_steps"]) is not int or config["warmup_steps"] < 1:
        raise ValueError("warmup_steps must be a positive integer")
    for key in ("temperature_K", "dt_fs", "tau_fs", "cutoff_nm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 < config["ewald_tolerance"] < .5:
        raise ValueError("ewald_tolerance must be between 0 and 0.5")
    if not config["replicas"] or any(type(p) is not int or p not in (1, 2, 4, 8, 16, 32)
                                     for p in config["replicas"]):
        raise ValueError("Supported replica counts: 1,2,4,8,16,32")
    if len(set(config["replicas"])) != len(config["replicas"]):
        raise ValueError("Replica counts must be distinct")
    for key in ("dispersion_correction", "openmm_blocking_sync"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    positions, box = read_structure(config["structure"])
    if 2 * config["cutoff_nm"] >= min(box):
        raise ValueError("The cutoff must be strictly below half the shortest box length")
    config.update(nwater=len(positions)//3, box_nm=box)
    return config


def read_structure(path):
    lines = Path(path).read_text().splitlines()
    count = int(lines[0])
    match = re.search(r"CELL\(abcABC\):\s+(.+?)\s+cell\{angstrom\}", lines[1])
    if not match or "positions{angstrom}" not in lines[1]:
        raise ValueError("Structure must declare angstrom positions and CELL(abcABC)")
    a, b, c, alpha, beta, gamma = map(float, match[1].split())
    if any(abs(x-90) > 1e-10 for x in (alpha, beta, gamma)):
        raise ValueError("This ordinary-Ewald example requires an orthorhombic box")
    if min(a,b,c) <= 0:
        raise ValueError("Box lengths must be positive")
    atoms = [line.split() for line in lines[2:2+count]]
    if len(atoms) != count or count < 3 or count % 3:
        raise ValueError("Structure must contain complete O,H,H water molecules")
    if [atom[0] for atom in atoms] != ["O", "H", "H"] * (count//3):
        raise ValueError("Expected atom order O,H,H repeated")
    positions = [[float(x)*.1 for x in atom[1:4]] for atom in atoms]
    if any(len(row) != 3 or not all(math.isfinite(x) for x in row) for row in positions):
        raise ValueError("Positions must have three finite coordinates")
    return positions, [a/10, b/10, c/10]


def total_steps(config):
    return config["warmup_steps"] + config["steps_per_sample"] * config["repeats"]


def projected_ns_day(seconds, steps, dt_fs):
    # Replicas evolve together: P does NOT multiply the simulated time.
    return 86400 * steps * dt_fs * 1e-6 / seconds


def functional_parameters(config, beads):
    import numpy as np
    return dict(nbeads=beads, temperature_K=config["temperature_K"],
        timestep_fs=config["dt_fs"], tau_fs=config["tau_fs"], seed=config["seed"],
        torch_seed=int(np.random.SeedSequence(config["seed"]).generate_state(1, dtype=np.uint64)[0]),
        fixcom=True, pilescale=1., potential="qtip4pf", transform="matrix",
        propagator="exact", splitting="obabo", masses_amu=MASS_DA,
        device="cuda", steps=total_steps(config), units=UNITS)


def input_xml(config, beads, method, precision):
    from xml.sax.saxutils import escape
    implementation = method.split('_')[1]
    main = method.startswith('main_')
    masses = ', '.join(str(MASS_DA[s]) for _ in range(config['nwater']) for s in ('O','H','H'))
    driver = (PACKAGE/'qtip4pf/ffdirect.py').resolve()
    parameters = (f"{{implementation: {implementation}, precision: {precision}, kernels: {config.get('kernels','array')}, device: cuda, "
        f"host_output: {str(main).lower()}, compiled: true, fixed_box: true, cutoff_nm: {config['cutoff_nm']}, "
        f"box_nm: {config['box_nm'][0]}, ewald_tolerance: {config['ewald_tolerance']}, "
        f"dispersion_correction: {str(config['dispersion_correction']).lower()}}}")
    return f"""<simulation verbosity='quiet' threading='false' safe_stride='{config['safe_stride']}'>
  <ffdirect name='water' threaded='false' pbc='false'>
    <pes>custom</pes><pes_path>{escape(str(driver))}</pes_path><batch_size>{beads}</batch_size>
    <parameters>{parameters}</parameters>
  </ffdirect>
  <total_steps>{total_steps(config)}</total_steps>
  <output prefix='simulation'/><prng><seed>{config['seed']}</seed></prng>
  <system><initialize nbeads='{beads}'>
    <file mode='xyz'>{escape(config['structure'])}</file>
    <masses mode='manual' units='dalton'>[{masses}]</masses>
    <velocities mode='thermal' units='kelvin'>{config['temperature_K']}</velocities>
  </initialize>
  <normal_modes transform='matrix' propagator='exact'/>
  <forces><force forcefield='water'/></forces>
  <ensemble><temperature units='kelvin'>{config['temperature_K']}</temperature></ensemble>
  <motion mode='dynamics'><fixcom>true</fixcom><dynamics mode='nvt'>
    <timestep units='femtosecond'>{config['dt_fs']}</timestep>
    <thermostat mode='pile_l'><tau units='femtosecond'>{config['tau_fs']}</tau></thermostat>
  </dynamics></motion></system>
</simulation>"""


def source_record(path):
    path = Path(path).resolve()
    commit = None
    if (path/".git").exists():
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        commit = result.stdout.strip()
    checksum = hashlib.sha256()
    for file in sorted((path/"ipi").rglob("*.py")):
        checksum.update(str(file.relative_to(path)).encode()+b"\0")
        checksum.update(file.read_bytes())
    checksum.update((path/"bin/i-pi").read_bytes())
    return dict(path=str(path), git_commit=commit, source_sha256=checksum.hexdigest())


def result_record(config, method, beads, samples, **metadata):
    if len(samples) != config["repeats"] or any(not math.isfinite(s) or s <= 0 for s in samples):
        raise ValueError("Missing or invalid timing samples")
    median = statistics.median(samples)
    return dict(schema_version=1, method=method, mode={"ipi_main":"main_normal", "ipi_pr":"pr_normal"}.get(method, method),
        nbeads=beads, nwater=config["nwater"], natoms=3*config["nwater"],
        steps_per_sample=config["steps_per_sample"], repeats=config["repeats"],
        warmup_steps=config["warmup_steps"], total_steps=total_steps(config),
        dt_fs=config["dt_fs"], temperature_K=config["temperature_K"], tau_fs=config["tau_fs"],
        cutoff_nm=config["cutoff_nm"], ewald_tolerance=config["ewald_tolerance"],
        dispersion_correction=config["dispersion_correction"], box_nm=config["box_nm"],
        config=config, seconds_per_100_steps_samples=samples if config["steps_per_sample"]==100 else None,
        samples_seconds=samples, median_seconds=median, min_seconds=min(samples), max_seconds=max(samples),
        projected_ns_day=projected_ns_day(median,config["steps_per_sample"],config["dt_fs"]),
        integration_dtype="float64", fixcom=True, initial_structure_sha256=digest(config["structure"]),
        force_field_sha256=digest(config["force_field"]),
        recorded_utc=datetime.now(timezone.utc).isoformat(), python=sys.version, **metadata)


import ctypes
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys



def build_barrier():
    import openmm
    site = Path(openmm.__file__).resolve().parent.parent
    roots = [site / "OpenMM.libs", site / "openmm.libs", Path(sys.prefix)]
    include = next((p / "include" for p in roots if (p / "include/openmm/Context.h").exists()), None)
    library = next((p / "lib" for p in roots if (p / "lib/libOpenMM.so").exists()), None)
    if os.environ.get("OPENMM_INCLUDE_DIR"):
        include = Path(os.environ["OPENMM_INCLUDE_DIR"])
    if os.environ.get("OPENMM_LIB_DIR"):
        library = Path(os.environ["OPENMM_LIB_DIR"])
    candidates = [site / "nvidia/cu13/include", site / "nvidia/cuda_runtime/include", site / "nvidia/cuda_nvcc/include",
                  Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"]
    cuda = next((p for p in candidates if (p / "cuda.h").exists()), None)
    if include is None or library is None or cuda is None:
        raise RuntimeError("Building the CUDA timer requires OpenMM development headers/libraries and cuda.h. "
                           "Set OPENMM_INCLUDE_DIR, OPENMM_LIB_DIR and CUDA_HOME as needed. "
                           "For an explicitly labelled public-API fallback use --synchronizer state.")
    source_text = CUDA_BARRIER
    key = hashlib.sha256(source_text.encode() + str(library.resolve()).encode() +
                         (library / "libOpenMM.so").read_bytes()).hexdigest()[:20]
    folder = ROOT / ".build" / key
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "cuda_barrier.cpp"
    source.write_text(source_text)
    target = folder / "cuda_barrier.so"
    if not target.exists():
        command = shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(target),
            "-I" + str(include), "-I" + str(cuda), "-L" + str(library),
            "-Wl,-rpath," + str(library), "-lOpenMM", "-l:libcuda.so.1"]
        subprocess.run(command, check=True)
        atomic_json(folder / "build.json", dict(command=command, openmm=openmm.__version__))
    return target


def openmm_synchronizer(context, integrator, method="barrier"):
    if method == "state":
        # Public API fallback: explicitly includes copy-to-context and one bead's
        # coordinate download at each block boundary. Never silently selected.
        return (lambda: integrator.getState(0, getPositions=True)), "public state download"
    library = ctypes.CDLL(str(build_barrier()))
    function = library.synchronize_openmm
    function.argtypes = [ctypes.c_void_p]
    function.restype = ctypes.c_int
    pointer = ctypes.c_void_p(int(context.this))

    def synchronize():
        result = function(pointer)
        if result:
            raise RuntimeError(f"OpenMM CUDA synchronization failed: {result}")

    return synchronize, "OpenMM CUDA-context barrier; no particle download"


STYLES={
 'pure_torch':('Pure / Torch','#9467bd','-','o'),
 'pure_jax':('Pure / JAX','#9467bd','--','s'),
 'openmm':('OpenMM','#2ca02c','-','D'),
 'main_torch':('i-PI main / Torch','#1f77b4','-','o'),
 'main_jax':('i-PI main / JAX','#1f77b4','--','s'),
 'pr_torch':('i-PI PR / Torch','#ff7f0e','-','o'),
 'pr_jax':('i-PI PR / JAX','#ff7f0e','--','s')}


def plot(path,output):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    path=Path(path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=json.loads((path/'summary.json' if path.is_dir() else path).read_text());rows=data['records']
    for key in ('dt_fs','temperature_K','tau_fs','cutoff_nm','ewald_tolerance','dispersion_correction','nwater'):
        if len({r[key] for r in rows})>1:raise ValueError(f'Mismatched physical setting: {key}')
    plt.style.use('default')
    modes=[mode for mode in ('mixed','double') if any(r['precision']==mode for r in rows)] or ['mixed']
    fig,axes=plt.subplots(1,len(modes),figsize=(5*len(modes),4.8),sharey=True,squeeze=False)
    axes=axes[0]
    for ax,mode in zip(axes,modes):
        for method,(label,color,line,marker) in STYLES.items():
            selected=sorted([r for r in rows if r['method']==method and r['precision']==mode],key=lambda r:r['nbeads'])
            if not selected:
                ax.plot([],[],label=label,color=color,ls=line,marker=marker);continue
            p=np.array([r['nbeads'] for r in selected]);y=np.array([r['projected_ns_day'] for r in selected])
            low=np.array([86400*r['steps_per_sample']*r['dt_fs']*1e-6/r['max_seconds'] for r in selected])
            high=np.array([86400*r['steps_per_sample']*r['dt_fs']*1e-6/r['min_seconds'] for r in selected])
            ax.errorbar(p,y,yerr=[y-low,high-y],color=color,ls=line,marker=marker,capsize=2,lw=1.6,label=label)
        ax.set_title('Forces: fp32; integration: fp64' if mode=='mixed' else 'Forces: fp64; integration: fp64')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([1,2,4,8,16,32],labels=[1,2,4,8,16,32]);ax.set_xlabel('Replicas P')
        ax.set_axisbelow(True)
        ax.grid(True,which='major',alpha=.25,linewidth=.6)
    if rows:
        lower=min(86400*r['steps_per_sample']*r['dt_fs']*1e-6/r['max_seconds'] for r in rows)
        upper=max(86400*r['steps_per_sample']*r['dt_fs']*1e-6/r['min_seconds'] for r in rows)
        axes[0].set_ylim(lower/1.15,upper*1.15)
    axes[0].set_ylabel('Projected throughput [ns/day]')
    handles,labels=axes[0].get_legend_handles_labels()
    order=[0,1,3,4,5,6,2]
    handles=[handles[i] for i in order];labels=[labels[i] for i in order]
    columns=4 if len(modes)==2 else 2
    bottom=.17 if len(modes)==2 else .25
    fig.legend(handles,labels,loc='lower center',ncol=columns,fontsize=9,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=[0,bottom,1,1])
    for ext in ('png','pdf','svg'):fig.savefig(output/f'throughput.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    keys=['precision','method','nbeads','median_seconds','min_seconds','max_seconds','projected_ns_day']
    with (output/'timings.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in rows)
    (output/'missing-and-failed.json').write_text(json.dumps({k:data.get(k,[]) for k in ('failed','pending')},indent=2)+'\n')


def fingerprint(config,sources):
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for folder in (PACKAGE,):
        for f in sorted(folder.rglob('*.py')):
            if any(part in ('historical','results','sources','__pycache__','.build') for part in f.parts):continue
            h.update(str(f.relative_to(folder)).encode());h.update(f.read_bytes())
    h.update(json.dumps(sources,sort_keys=True).encode())
    return h.hexdigest()


def summarize(output):
    records=[];failed=[];pending=[]
    for precision in ('mixed','double'):
        for method in METHODS:
            for p in (1,2,4,8,16,32):
                folder=output/precision/method/f'P{p}'
                if (folder/'result.json').exists():records.append(json.loads((folder/'result.json').read_text()))
                elif (folder/'failure.json').exists():failed.append(json.loads((folder/'failure.json').read_text()))
                else:pending.append(dict(precision=precision,method=method,nbeads=p))
    diffs=[]
    import numpy as np
    for precision in ('mixed','double'):
        for p in (1,2,4,8,16,32):
            ref=output/precision/'pr_torch'/f'P{p}'/'final-state.npz'
            if not ref.exists():continue
            with np.load(ref) as expected:
                for method in METHODS:
                    path=output/precision/method/f'P{p}'/'final-state.npz'
                    if not path.exists():continue
                    with np.load(path) as actual:
                        diffs.append(dict(precision=precision,nbeads=p,method=method,reference='pr_torch',
                            byte_identical=all(actual[k].tobytes()==expected[k].tobytes() for k in ('q','p')),
                            **{k+'_max_abs_au':float(np.max(np.abs(actual[k]-expected[k]))) for k in ('q','p')}))
    atomic_json(output/'summary.json',dict(records=records,failed=failed,pending=pending,final_state_differences=diffs,
        differences_note='Native RNG streams/initial momenta differ. This is diagnostic, not a force-field accuracy test.'))


def run_cases(a):
    import fcntl
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=load_config(a.config)
        c["kernels"]=a.kernels
        if a.smoke:c.update(warmup_steps=20,steps_per_sample=4,repeats=1)
        sources=SOURCES
        key=fingerprint(c,sources)
        if (output/'run.json').exists():
            old=json.loads((output/'run.json').read_text())
            if old['fingerprint']!=key:raise ValueError('Code/settings changed; use a new output directory')
        atomic_json(output/'run.json',dict(config=c,sources=sources,fingerprint=key,pid=os.getpid(),started=time.time()))
        atomic_json(output/'config.json',c)
        env=dict(os.environ,QTIP4PF_KERNELS=a.kernels,CUDA_VISIBLE_DEVICES=a.gpu,XLA_PYTHON_CLIENT_PREALLOCATE='false',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        env['PYTHONPATH']=str(PACKAGE)+os.pathsep+env.get('PYTHONPATH','')
        for p in a.replicas:
            for precision in a.precision:
                for method in a.methods:
                    folder=output/precision/method/f'P{p}';folder.mkdir(parents=True,exist_ok=True)
                    if a.resume and (folder/'result.json').exists():continue
                    worker='torch' if method=='pure_torch' else 'jax' if method=='pure_jax' else 'openmm' if method=='openmm' else 'ipi'
                    cmd=[sys.executable,str(PACKAGE/"benchmark.py"),"--worker",worker,'--config',str(output/'config.json'),'--nbeads',str(p),'--precision',precision,'--output',str(folder/'result.json')]
                    if method.startswith(('main_','pr_')):
                        cmd+=['--method',method,'--source',str(a.source_main if method.startswith('main_') else a.source_pr)]
                    (folder/'result.json').unlink(missing_ok=True)
                    atomic_json(folder/'command.json',dict(command=cmd,fingerprint=key))
                    print(f'START {precision} {method} P={p}',flush=True)
                    with (folder/'run.log').open('w') as log:
                        result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode or not (folder/'result.json').exists():
                        atomic_json(folder/'failure.json',dict(method=method,precision=precision,nbeads=p,returncode=result.returncode,log=str(folder/'run.log')))
                        print(f'FAILED {precision} {method} P={p}',flush=True)
                    else:
                        (folder/'failure.json').unlink(missing_ok=True)
                        print(f'DONE {precision} {method} P={p}',flush=True)
                    summarize(output)
        summarize(output)
        summary=json.loads((output/'summary.json').read_text())
        atomic_json(output/'finished.json',dict(successes=len(summary['records']),failures=len(summary['failed']),pending=len(summary['pending']),time=time.time()))



def openmm_worker():
    import argparse
    from pathlib import Path
    import time

    import numpy as np
    import openmm as mm
    from openmm import app, unit



    def make_context(config, beads):
        q, box = read_structure(config["structure"])
        q = np.asarray(q)
        water = q.reshape(-1, 3, 3)
        topology = app.Topology()
        chain = topology.addChain()
        for _ in water:
            residue = topology.addResidue("HOO", chain)
            atoms = [topology.addAtom(name, element, residue) for name, element in
                     [("O", app.element.oxygen), ("H1", app.element.hydrogen),
                      ("H2", app.element.hydrogen), ("M", None)]]
            topology.addBond(atoms[0], atoms[1])
            topology.addBond(atoms[0], atoms[2])
        topology.setPeriodicBoxVectors([mm.Vec3(*row)*unit.nanometer for row in np.diag(box)])
        system = app.ForceField(config["force_field"]).createSystem(
            topology, nonbondedMethod=app.Ewald, nonbondedCutoff=config["cutoff_nm"]*unit.nanometer,
            ewaldErrorTolerance=config["ewald_tolerance"], rigidWater=False,
            constraints=None, removeCMMotion=True)
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setUseDispersionCorrection(config["dispersion_correction"])
        if system.getNumConstraints():
            raise RuntimeError("Flexible q-TIP4P/F must have no constraints")
        integrator = mm.RPMDIntegrator(beads, config["temperature_K"]*unit.kelvin,
            (1000/config["tau_fs"])/unit.picosecond, (config["dt_fs"]/1000)*unit.picosecond)
        integrator.setRandomNumberSeed(config["seed"])
        platform = mm.Platform.getPlatformByName("CUDA")
        context = mm.Context(system, integrator, platform, {
            "Precision": config["precision"], "DeviceIndex": str(config["openmm_device_index"]),
            "UseBlockingSync": str(config["openmm_blocking_sync"]).lower(),
            "DeterministicForces": "false"})
        properties = {key: platform.getPropertyValue(context, key) for key in platform.getPropertyNames()}
        if properties["Precision"] != config["precision"] or dict(integrator.getContractions()):
            raise RuntimeError("Expected CUDA double and no ring-polymer contraction")
        assert integrator.getApplyThermostat()
        assert abs(integrator.getTemperature().value_in_unit(unit.kelvin)-config["temperature_K"]) < 1e-12
        assert abs(integrator.getFriction().value_in_unit(unit.picosecond**-1)-1000/config["tau_fs"]) < 1e-12
        extended = np.zeros((len(water), 4, 3))
        extended[:, :3] = water
        for molecule in range(len(water)):
            site = system.getVirtualSite(4*molecule+3)
            extended[molecule, 3] = sum(site.getWeight(j)*water[molecule, j] for j in range(3))
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                          for i in range(system.getNumParticles()) if not system.isVirtualSite(i)])
        np.testing.assert_array_equal(masses, np.tile([MASS_DA["O"], MASS_DA["H"], MASS_DA["H"]], len(water)))
        rng = np.random.default_rng(config["seed"])
        velocities = rng.normal(size=(beads, len(q), 3))*np.sqrt(
            beads*.00831446261815324*config["temperature_K"]/masses[None, :, None])
        velocities -= (velocities*masses[None, :, None]).sum(axis=(0, 1), keepdims=True)/(masses.sum()*beads)
        for bead in range(beads):
            integrator.setPositions(bead, extended.reshape(-1, 3)*unit.nanometer)
            v = np.zeros((len(water), 4, 3))
            v[:, :3] = velocities[bead].reshape(-1, 3, 3)
            integrator.setVelocities(bead, v.reshape(-1, 3)*unit.nanometer/unit.picosecond)
        return context, integrator, system, properties, masses


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--precision", choices=("mixed","double"), required=True)
        parser.add_argument("--nbeads", type=int, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--synchronizer", choices=("barrier", "state"), default="barrier")
        args = parser.parse_args()
        config = load_config(args.config); config["precision"] = args.precision
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        context, integrator, system, properties, masses = make_context(config, args.nbeads)
        synchronize, timing_method = openmm_synchronizer(context, integrator, args.synchronizer)
        integrator.step(config["warmup_steps"])
        synchronize()
        setup_seconds = time.perf_counter()-started
        samples = []
        for _ in range(config["repeats"]):
            synchronize()
            start = time.perf_counter_ns()
            integrator.step(config["steps_per_sample"])
            synchronize()
            samples.append((time.perf_counter_ns()-start)*1e-9)
        # Output downloads and serialization start only after the last timed block.
        positions, momenta = [], []
        for bead in range(args.nbeads):
            state = integrator.getState(bead, getPositions=True, getVelocities=True)
            q = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).reshape(-1, 4, 3)[:, :3].reshape(-1, 3)
            v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond).reshape(-1, 4, 3)[:, :3].reshape(-1, 3)
            positions.append(q*(10*UNITS["angstrom"]))
            momenta.append(v*(masses[:, None]*UNITS["amu"])*(10*UNITS["angstrom"]/(1000*UNITS["femtosecond"])))
        if not np.isfinite(positions).all() or not np.isfinite(momenta).all():
            raise RuntimeError("Nonfinite final OpenMM state")
        elapsed = state.getTime().value_in_unit(unit.picosecond)
        if abs(elapsed-total_steps(config)*config["dt_fs"]/1000) > 1e-9:
            raise RuntimeError("OpenMM step count/time mismatch")
        np.savez(output.parent/"final-state.npz", q=np.array(positions).reshape(args.nbeads, -1),
                 p=np.array(momenta).reshape(args.nbeads, -1))
        (output.parent/"system.xml").write_text(mm.XmlSerializer.serialize(system))
        result = result_record(config, "openmm", args.nbeads, samples,
            device="cuda", gpu=properties["DeviceName"], openmm=mm.__version__,
            openmm_git_revision=getattr(mm.version, "git_revision", None),
            platform_properties=properties, setup_seconds=setup_seconds, finite=True,
            virtual_sites=config["nwater"], synchronizer=timing_method,
            timing_scope="native RPMDIntegrator.step(block_size); barrier at block boundaries only",
            outputs_during_timing=False, checkpointing=False, arithmetic="CUDA "+args.precision, precision=args.precision, potential_dtype="float32" if args.precision=="mixed" else "float64",
            random_dtype="float32", com_method="native removal separately for every replica",
            extra_work="native integration requests forces; no energy/full-virial output each step",
            worker_sha256=digest(__file__), common_sha256=digest(ROOT/"benchmark.py"))
        atomic_json(output, result)
        print(f"OpenMM P={args.nbeads}: {result['projected_ns_day']:.3f} ns/day", flush=True)

    main()


def ipi_worker():
    import argparse
    import math
    import os
    from pathlib import Path
    import runpy
    import sys
    import time
    import traceback

    import numpy as np
    import torch



    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--precision", choices=("mixed","double"), required=True)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--nbeads", type=int, required=True)
        parser.add_argument("--method", choices=("main_torch", "main_jax", "pr_torch", "pr_jax"), required=True)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        config = load_config(args.config)
        gpu = args.method.startswith("pr_")
        backend = args.method.split("_")[1]
        config["precision"] = args.precision
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        output, source = args.output.resolve(), args.source.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(output.parent)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        xml = input_xml(config, args.nbeads, args.method, args.precision)
        Path("input.xml").write_text(xml+"\n")
        sys.path.insert(0, str(source))
        from ipi.engine.simulation import Simulation
        import ipi
        assert Path(ipi.__file__).resolve().is_relative_to(source), ipi.__file__
        original_step, original_run = Simulation.run_step, Simulation.run
        if backend == "torch":
            synchronize = torch.cuda.synchronize
        else:
            import jax
            synchronize = lambda: jax.effects_barrier()
            # At each measured boundary explicitly wait for the actual state below.

        tracking = dict(boundaries=[], steps=0, last=-1, checkpoints=[])

        def observed_step(simulation, step):
            if step >= config["warmup_steps"] and (step-config["warmup_steps"]) % config["steps_per_sample"] == 0:
                synchronize()
                if gpu and backend == "jax":
                    jax.block_until_ready(simulation.syslist[0].beads.q)
                tracking["boundaries"].append(time.perf_counter_ns())
            tracking["last"] = step
            result = original_step(simulation, step)
            tracking["steps"] += 1
            return result

        def observed_run(simulation, *positional, **keyword):
            system = simulation.syslist[0]
            assert simulation.safe_stride == config["safe_stride"]
            assert simulation.fflist["water"].batch_size == args.nbeads
            assert simulation.tsteps == total_steps(config)
            assert len(simulation.outputs) == 0
            assert system.beads.nbeads == args.nbeads
            assert system.beads.natoms == 3*config["nwater"]
            assert system.nm.transform_method == "matrix" and system.nm.propagator == "exact"
            assert system.nm.mode == "rpmd" and system.motion.fixcom
            assert math.isclose(float(system.motion.dt), config["dt_fs"]*4.1341373e16*1e-15, rel_tol=1e-13)
            assert math.isclose(float(system.ensemble.temp), config["temperature_K"]*3.1668152e-6, rel_tol=1e-13)
            original_store = simulation.chk.store

            def observed_store():
                result = original_store()
                tracking["checkpoints"].append(tracking["last"]+1)
                return result

            simulation.chk.store = observed_store
            original_run(simulation, *positional, **keyword)
            synchronize()
            if gpu and backend == "jax": jax.block_until_ready(system.beads.q)
            tracking["boundaries"].append(time.perf_counter_ns())
            assert tracking["steps"] == total_steps(config)
            assert len(tracking["boundaries"]) == config["repeats"]+1
            periodic = [step for step in tracking["checkpoints"] if step > 0]
            assert periodic == list(range(config["safe_stride"], total_steps(config)+1, config["safe_stride"]))
            samples = [(b-a)*1e-9 for a,b in zip(tracking["boundaries"], tracking["boundaries"][1:])]
            q, p = system.beads.q, system.beads.p
            assert str(q.dtype) in ("float64", "torch.float64")
            array = (lambda x: x.detach().cpu().numpy()) if gpu and backend == "torch" else np.asarray
            assert np.isfinite(array(q)).all() and np.isfinite(array(p)).all()
            metadata = dict(device="cuda" if gpu else "cpu", numpy=np.__version__, torch=torch.__version__,
                source=source_record(source), batch_size=args.nbeads, safe_stride_input=config["safe_stride"],
                transform="matrix", propagator="exact", finite=True, outputs_during_timing=False,
                checkpointing=True, periodic_checkpoint_steps=periodic,
                setup_seconds=(tracking["boundaries"][0]-started)*1e-9,
                timing_scope="normal i-PI XML/CLI run; checkpoint snapshots included; final file writes excluded",
                extra_work="potential returns energy, force and complete virial each force evaluation",
                worker_sha256=digest(__file__), common_sha256=digest(ROOT/"benchmark.py"),
                input_sha256=digest(Path("input.xml")),
                precision=args.precision, potential_dtype="float32" if args.precision=="mixed" else "float64", potential_backend=backend, potential_kernels=config.get("kernels","array"), potential_compiled=True)
            driver = simulation.fflist["water"].driver
            assert driver.evaluators
            assert driver.fixed_box
            metadata['box_geometry'] = 'cached rectangular lengths and reciprocals'
            assert set(driver.evaluators) == {(config['nwater'],args.nbeads)}
            if gpu:
                replay = system.motion.integrator.step
                assert replay.compiled is not None, getattr(replay,"reason",None)
                if backend == "torch":
                    metadata.update(cuda_graphs=len(replay.segments),
                        compiled_scope="Torch compiled E/F/W between integrator CUDA graph segments",
                        compile_fullgraph_potential=True, compile_dynamic=False)
                else:
                    metadata.update(jax=jax.__version__, compiled_scope="whole MD step including E/F/W and random key updates",
                        whole_step_jit=True)
            else:
                metadata.update(compiled_scope="compiled GPU potential; NumPy integration and transfers",
                    whole_step_jit=False)
            np.savez(output.parent/"final-state.npz", q=array(q), p=array(p))
            result = result_record(config, args.method, args.nbeads, samples, **metadata)
            atomic_json(output, result)
            print(f"{args.method} P={args.nbeads}: {result['projected_ns_day']:.3f} ns/day", flush=True)

        Simulation.run_step, Simulation.run = observed_step, observed_run
        sys.argv = [str(source/"bin/i-pi"), "input.xml"]
        if gpu:
            sys.argv += ["--backend", backend, "--device", "cuda", "--jit"]
        started = time.perf_counter_ns()
        runpy.run_path(sys.argv[0], run_name="__main__")

    main()


if __name__ == "__main__":
    sys.modules["benchmark"] = sys.modules[__name__]
    try:
        if len(sys.argv)>2 and sys.argv[1]=="--worker":
            kind=sys.argv[2]
            del sys.argv[1:3]
            if kind in ("torch","jax"):
                importlib.import_module("pimd_"+kind).main()
            elif kind=="openmm":openmm_worker()
            elif kind=="ipi":ipi_worker()
            else:raise ValueError(kind)
        else:
            raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Benchmark error: {error}",file=sys.stderr)
        raise SystemExit(1)
