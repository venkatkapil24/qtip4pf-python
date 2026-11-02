# qtip4pf-python

Torch/JAX q-TIP4P/F with rectangular periodic boxes, ordinary Ewald and shared
Triton kernels. Returns energy, forces and full virials. `pimd_torch.py` and
`pimd_jax.py` contain the functional PILE-L PIMD references, including RNG,
COM removal and fp64 integration.

Use **uv**, Linux/WSL2, an NVIDIA CUDA-13-compatible GPU and a C++17 compiler.
Python, dependencies and i-PI revisions are pinned by the script and lockfile.

```sh
uv run --locked benchmark.py --example torch
uv run --locked benchmark.py --example jax
uv run --locked benchmark.py                     # run and plot
uv run --locked benchmark.py --plot-only         # saved results only
uv run --locked benchmark.py --validate          # independent GPU tests
```

The two inputs in `examples/` use 64 waters, P=4 and batch size 4. The benchmark
uses P=1,2,4,8,16,32, fp32 forces/fp64 integration, 20 warmup steps and five
100-step samples. Compare OpenMM, pure Torch/JAX, i-PI main and the compiled
PR. Figure and separate ASCII caption: `benchmark-results/plots/`.
Use `--precision mixed double` for both precision panels, `--kernels array`
for array kernels, `--smoke` for a short run, or `--resume` to continue.

Standalone: import `make_evaluator` from `qtip4pf.torch` or `qtip4pf.jax`,
then `f = make_evaluator(Nwater, box_nm, batch_size=P, dtype="float32",
kernels="triton")` and `energy, forces, virial = f(positions, cells)`.
Inputs are native arrays, shapes `(P,3*Nwater,3)` and `(P,3,3)`, in nm.
Outputs: kJ/mol, kJ/mol/nm and kJ/mol. Use whole O,H,H molecules; each box
length must exceed twice the cutoff. First energy derivatives are supported;
Hessians are not.

Tests use OpenMM Reference and independent full-virial expressions. Initial
Triton speed checks cover fp32 forces/fp64 integration. Water parameters follow
[WaterPIMD](https://github.com/yunhzou/WaterPIMD/blob/c913385b5805ece5801042942cd19a3df1873e42/qtip4pf.xml);
PIMD arithmetic derives from i-PI (source revision noted in `pimd_torch.py`).
