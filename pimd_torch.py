"""Functional closed-path q-TIP4P/F NVT/PILE-L update.

Adapted from i-PI a0d47d515c19189647e08623c853b3d7aafc980c
(normalmodes.py, nmtransform.py, thermostats.py, dynamics.py).
The active initialization and update arithmetic is preserved from the validated
benchmark. Matrix transforms, exact propagation, OBABO and global COM removal.
initialize reads the structure; step takes/returns tensors and calls the supplied
energy/force/virial function. Random draws are inside step. The caller manages
compilation, CUDA capture and RNG state. No unused FFT or dummy-force CLI remains.
"""
import math
from pathlib import Path
import re

import torch

DTYPE = torch.float64

def tensor(value, device):
    return torch.as_tensor(value, dtype=DTYPE, device=device)



def read_xyz(path, parameters):
    """Read the fixture's XYZ file: positions/orthorhombic cell in angstrom."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    atoms = [line.split() for line in lines[2 : 2 + n]]
    if len(atoms) != n:
        raise ValueError("Incomplete initial structure")
    match = re.search(r"CELL\(abcABC\):\s+(.+?)\s+cell\{angstrom\}", lines[1])
    if not match or "positions{angstrom}" not in lines[1]:
        raise ValueError("Expected CELL(abcABC), cell{angstrom}, positions{angstrom}")
    a, b, c, alpha, beta, gamma = map(float, match.group(1).split())
    alpha, beta, gamma = [v * math.pi / 180.0 for v in (alpha, beta, gamma)]
    # i-PI's abc2h order, followed by conversion of the complete matrix.
    h = [[0.0] * 3 for _ in range(3)]
    h[0][0] = a
    h[0][1] = b * math.cos(gamma)
    h[0][2] = c * math.cos(beta)
    h[1][1] = b * math.sin(gamma)
    h[1][2] = (b * c * math.cos(alpha) - h[0][1] * h[0][2]) / h[1][1]
    h[2][2] = math.sqrt(c**2 - h[0][2] ** 2 - h[1][2] ** 2)
    device = parameters["device"]
    q = tensor([[float(v) for v in atom[1:4]] for atom in atoms], device)
    q = q.reshape(-1) * parameters["units"]["angstrom"]
    mass = tensor(
        [
            parameters["masses_amu"][atom[0]] * parameters["units"]["amu"]
            for atom in atoms
        ],
        device,
    )
    return q, mass, tensor(h, device) * parameters["units"]["angstrom"]



def b2nm(q, c):
    return c['nm_matrix'] @ q



def nm2b(qnm, c):
    return c['nm_matrix'].T @ qnm



def remove_com(p, weights, half_inv_mtot):
    pcom = p.reshape(-1, 3).sum(dim=0)
    dp = (weights * pcom).reshape(p.shape)
    ecom = (pcom @ pcom) * half_inv_mtot
    return p - dp, ecom



def initialize(path, parameters):
    """Build state from structure and parameters, with native Torch noise."""
    p = parameters
    if (
        p["nbeads"] not in (1, 2, 4, 8, 16, 32)
        or p["transform"] != "matrix"
        or p["propagator"] != "exact"
        or p["splitting"] != "obabo"
        or p["potential"] != "qtip4pf"
        or not p["fixcom"]
    ):
        raise ValueError(
            "Only the documented power-of-two bead matrix/exact/OBABO/qtip4pf/COM-fixed fixture is supported"
        )
    device = p["device"]
    q0, mass, h = read_xyz(path, p)
    count, n = p["nbeads"], mass.numel()
    m3 = mass.repeat_interleave(3).repeat(count, 1)
    sm3 = torch.sqrt(m3)
    temperature = p["temperature_K"] * p["units"]["kelvin"]
    # unit_to_internal uses number * seconds_conversion * femto_prefix.
    dt = p["timestep_fs"] * 4.1341373e16 * 1e-15
    tau = p["tau_fs"] * 4.1341373e16 * 1e-15
    omega_n = temperature * count * 1.0 / 1.0
    omega = omega_n * tensor(
        [2.0 * math.sin(k * math.pi / count) for k in range(count)], device
    )
    weights = tensor([1.0] if count == 1 else [1.0] + [math.sqrt(2.0)] * (count // 2 - 1) + [1.0], device).reshape(-1, 1) / math.sqrt(
        count
    )
    c = dict(
        nbeads=count,
        natoms=n,
        nmodes=count // 2,
        nsin=count - count // 2 - 1,
        fft_weights=weights,
        m3=m3,
        sm3=sm3,
        mass=mass,
        omega=omega,
        temperature=temperature,
        dt=dt,
        fixcom=p["fixcom"],
        device=device,
    )
    matrix = [[(1.0 if k == 0 else
                ((1.0 if j%2 == 0 else -1.0) if k == count//2 and count%2 == 0 else
                 math.sqrt(2.0)*(math.cos(2*math.pi*j*k/count) if k <= count//2 else math.sin(2*math.pi*j*k/count))))
               / math.sqrt(count) for j in range(count)] for k in range(count)]
    c['nm_matrix'] = tensor(matrix, device)
    # NVEIntegrator.qdt = dt*0.5/inmts; here inmts=1.
    qdt = dt * 0.5 / 1
    c["qdt_on_m"] = qdt / m3[0]
    c["weights"] = 3.0 * m3.reshape(-1, 3) / m3.sum()
    c["half_inv_mtot"] = 1.5 / m3.sum()
    # Free-ring-polymer coefficient construction from _propagator_matrices.
    sk = torch.sqrt(torch.ones(count, dtype=DTYPE, device=device)[1:])
    wk = omega[1:]
    angle = wk * qdt / sk
    diagonal = torch.cos(angle)
    p_from_q = -torch.sin(angle) * wk * sk
    q_from_p = torch.sin(angle) / (wk * sk)
    modes = torch.stack(
        [
            torch.stack([diagonal, p_from_q], dim=-1),
            torch.stack([q_from_p, diagonal], dim=-1),
        ],
        dim=-2,
    )
    pq = torch.cat([tensor([[[1.0, 0.0], [qdt, 1.0]]], device), modes], dim=0)
    propagator = torch.zeros((2, 2, count, 3 * n), dtype=DTYPE, device=device)
    propagator[...] = pq.permute(1, 2, 0)[:, :, :, None]
    propagator[0, 1] *= m3
    propagator[1, 0] /= m3
    c["propagator"] = propagator
    # Independent Langevin updates; call shapes/order match PILE-L.
    taus = [tau] + [float(1.0 / (2 * p["pilescale"] * float(w))) for w in omega[1:]]
    c["damping"] = [tensor(math.exp(-(dt * 0.5) / t), device) for t in taus]
    c["damping2"] = [t**2 for t in c["damping"]]
    c["noise_scale"] = [
        torch.sqrt(1.0 * (temperature * count) * (1 - t2)) for t2 in c["damping2"]
    ]
    c["damping_on_sm"] = [t / sm3[i] for i, t in enumerate(c["damping"])]
    generator = torch.Generator(device=device)
    generator.manual_seed(p["torch_seed"])
    pnm = (
        torch.randn((count, 3 * n), generator=generator, dtype=DTYPE, device=device)
        * torch.sqrt(m3 * torch.ones(count, dtype=DTYPE, device=device)[:, None])
        * math.sqrt(count * temperature * 1.0)
    )
    bead_q = q0.repeat(count, 1)
    bead_p, initial_com_energy = remove_com(
        nm2b(pnm, c), c["weights"], c["half_inv_mtot"]
    )
    c["tensors"] = {
        key: c[key]
        for key in [
            "nm_matrix",
            "fft_weights",
            "sm3",
            "propagator",
            "weights",
            "half_inv_mtot",
            "qdt_on_m",
        ]
    }
    c["tensors"].update(
        damping_on_sm=torch.stack(c["damping_on_sm"]),
        damping2=torch.stack(c["damping2"]),
        noise_scale=torch.stack(c["noise_scale"]),
        dt=tensor(dt, device),
        pdt=tensor(dt * 0.5, device),
    )
    state = dict(
        q=bead_q,
        p=bead_p,
        h=h,
        qnm=b2nm(bead_q, c),
        pnm=b2nm(bead_p, c),
        heat=torch.zeros(count, dtype=DTYPE, device=device),
        com_energy=tensor(0.0, device) + initial_com_energy,
        rng=generator.get_state(),
        time=tensor(0.0, device),
    )
    return state, c



def thermostat(pnm, heat_by_mode, c, noise=None):
    p = pnm*c['damping_on_sm']
    heat = (p*p).sum(dim=1)/c['damping2']
    draw = torch.randn_like(p) if noise is None else noise
    p = p+c['noise_scale'][:,None]*draw
    heat = heat-(p*p).sum(dim=1)
    return p*c['sm3'],heat_by_mode+heat*.5


def free_half_step(qnm, pnm, c):
    centroid = qnm[0] + pnm[0] * c["qdt_on_m"]
    if qnm.shape[0] == 1:
        return centroid[None, :], pnm
    a = c["propagator"]
    new_p = pnm[1:] * a[0, 0, 1:] + qnm[1:] * a[0, 1, 1:]
    new_q = pnm[1:] * a[1, 0, 1:] + qnm[1:] * a[1, 1, 1:]
    return (
        torch.cat([centroid[None, :], new_q], dim=0),
        torch.cat([pnm[:1], new_p], dim=0),
    )



def step(state, c, force_function, noise=None):
    """The complete tensor-only MD step, including native Gaussian draws.

    c is coefficients['tensors']. All input tensors are immutable. Two draws per bead
    from Torch's default device generator occur in the original
    PILE-L order. The caller installs and retrieves the generator state; this
    numerical function itself contains no generator state-management calls.
    """
    pnm, heat = thermostat(state["pnm"], state["heat"], c, None if noise is None else noise[0])
    p = (c["nm_matrix"].T @ pnm)
    p, ecom = remove_com(p, c["weights"], c["half_inv_mtot"])
    com_energy = state["com_energy"] + ecom

    # First force kick, preserving i-PI's arithmetic and COM-removal order.
    p = p + state["force"] * c["pdt"]
    p, ecom = remove_com(p, c["weights"], c["half_inv_mtot"])
    com_energy = com_energy + ecom
    pnm = (c["nm_matrix"] @ p)
    qnm, pnm = free_half_step(state["qnm"], pnm, c)
    qnm, pnm = free_half_step(qnm, pnm, c)

    # Read positions/forces before the second kick, just as i-PI does.
    q = (c["nm_matrix"].T @ qnm)
    p = (c["nm_matrix"].T @ pnm)
    potential, force, virial = force_function(q, state["h"])
    p = p + force * c["pdt"]
    p, ecom = remove_com(p, c["weights"], c["half_inv_mtot"])
    com_energy = com_energy + ecom
    pnm = (c["nm_matrix"] @ p)
    pnm, heat = thermostat(pnm, heat, c, None if noise is None else noise[1])
    p = (c["nm_matrix"].T @ pnm)
    p, ecom = remove_com(p, c["weights"], c["half_inv_mtot"])
    com_energy = com_energy + ecom
    # The normal modes are read by kinetic_md at each observation boundary.
    pnm = (c["nm_matrix"] @ p)
    return dict(
        force=force, potential=potential, virial=virial,
        q=q,
        p=p,
        qnm=qnm,
        pnm=pnm,
        h=state["h"],
        heat=heat,
        com_energy=com_energy,
        time=state["time"] + c["dt"],
        rng=state["rng"],
    )





import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from benchmark import ROOT, atomic_json, digest, functional_parameters, load_config, result_record, total_steps


from qtip4pf import _model as pes
from importlib import import_module
KERNELS = os.environ.get("QTIP4PF_KERNELS", "array")
make_evaluator = import_module("qtip4pf.torch").make_evaluator


md = sys.modules[__name__]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("mixed","double"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nbeads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
    started = time.perf_counter()
    state, coefficients = md.initialize(config["structure"], functional_parameters(config, args.nbeads))
    dtype = torch.float32 if args.precision=="mixed" else torch.float64
    evaluate = make_evaluator(config["nwater"],config["box_nm"],args.nbeads,
        dtype="float32" if args.precision=="mixed" else "float64",device="cuda",
        cutoff_nm=config["cutoff_nm"],ewald_tolerance=config["ewald_tolerance"],
        dispersion_correction=config["dispersion_correction"], kernels=KERNELS)
    evaluate = evaluate.set_box((state['h'].to(dtype)*pes.NM_PER_BOHR)[None].expand(args.nbeads,-1,-1))

    def force(q, h):
        energy, forces, virial = evaluate(q.reshape(args.nbeads, -1, 3).to(dtype)*pes.NM_PER_BOHR)
        return ((energy*pes.HARTREE_PER_KJ_MOL).double(),
                (forces.reshape(q.shape)*(pes.HARTREE_PER_KJ_MOL*pes.NM_PER_BOHR)).double(),
                (virial*pes.HARTREE_PER_KJ_MOL).double())

    state["potential"], state["force"], state["virial"] = force(state["q"], state["h"])
    initial = {key:value.clone() for key,value in state.items()}

    def step(s, c):
        updated = md.step(s, c, force)
        for key in ("q", "p", "qnm", "pnm", "heat", "com_energy", "time", "force", "potential", "virial"):
            s[key].copy_(updated[key])

    compiled = torch.compile(step, fullgraph=True, dynamic=False, options={"triton.cudagraphs":False})
    torch.cuda.set_rng_state(state["rng"])
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): compiled(state, coefficients["tensors"])
    torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): compiled(state, coefficients["tensors"])
    for key in state: state[key].copy_(initial[key])
    torch.cuda.set_rng_state(initial["rng"]); torch.cuda.synchronize()
    for _ in range(config["warmup_steps"]): graph.replay()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-started
    samples = []
    for _ in range(config["repeats"]):
        torch.cuda.synchronize(); start = time.perf_counter_ns()
        for _ in range(config["steps_per_sample"]): graph.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns()-start)*1e-9)
    for key in ("q", "p", "force", "potential", "virial"):
        if not bool(torch.isfinite(state[key]).all()): raise RuntimeError(f"Nonfinite final {key}")
    assert abs(state["time"].item()-total_steps(config)*coefficients["dt"]) < 1e-7
    np.savez(output.parent/"final-state.npz", q=state["q"].cpu().numpy(), p=state["p"].cpu().numpy())
    result = result_record(config, "pure_torch", args.nbeads, samples, potential_kernels=KERNELS,
        device="cuda", gpu=torch.cuda.get_device_name(), torch=torch.__version__, torch_cuda=torch.version.cuda,
        batch_size=args.nbeads, transform="matrix", propagator="exact", finite=True,
        box_geometry='cached rectangular lengths and reciprocals',
        setup_seconds=setup_seconds, outputs_during_timing=False, checkpointing=False,
        compile_fullgraph=True, compile_dynamic=False, cuda_graphs=1,
        compiled_scope="complete MD + energy/force/virial + random numbers + live pair cutoff calculation",
        timing_scope="100-step-style Python CUDA-graph replay loop; synchronized block boundaries",
        extra_work="potential returns energy, force and complete virial each force evaluation",
        worker_sha256=digest(__file__), common_sha256=digest(ROOT/"benchmark.py"),
        precision=args.precision, potential_dtype=str(dtype))
    atomic_json(output, result)
    print(f"Pure Torch P={args.nbeads}: {result['projected_ns_day']:.3f} ns/day", flush=True)


if __name__ == "__main__":
    main()
