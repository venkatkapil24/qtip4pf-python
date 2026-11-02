"""i-PI FFDirect adapter. Inputs and outputs use i-PI atomic units."""
from qtip4pf._model import NM_PER_BOHR, HARTREE_PER_KJ_MOL

__DRIVER_NAME__ = "qtip4pf"
__DRIVER_CLASS__ = "Driver"


def _boolean(value):
    if isinstance(value, str):
        if value.lower() not in ("true", "false"): raise ValueError("Expected true or false")
        return value.lower() == "true"
    return bool(value)


class Driver:
    """Select implementation=torch/jax, precision=mixed/double, device=cuda/cpu.

    host_output=true is for original NumPy i-PI. Otherwise returns native fp64
    backend arrays, while potential arithmetic uses the requested precision.
    kernels=array/triton selects the array or shared Triton GPU kernels.
    fixed_box=true stores rectangular-box geometry at initialization for NVT;
    leave false when the simulation changes the box.
    """
    jax_traceable = True
    virial_checked = True

    def __init__(self, implementation="torch", precision="double", device="cuda",
                 host_output=False, compiled=True, fixed_box=False, kernels="array", cutoff_nm=0.6,
                 ewald_tolerance=5e-4, dispersion_correction=True,
                 box_nm=1.24138, **kwargs):
        if precision not in ("mixed", "double"): raise ValueError("precision must be mixed or double")
        if implementation not in ("torch", "jax"): raise ValueError("implementation must be torch or jax")
        if kernels not in ("array", "triton"): raise ValueError("kernels must be array or triton")
        self.kernels = kernels
        self.implementation, self.device = implementation, device
        self.dtype = "float32" if precision == "mixed" else "float64"
        self.backend_arrays = not _boolean(host_output)
        self.compiled = _boolean(compiled)
        self.fixed_box = _boolean(fixed_box)
        self.options = dict(kernels=kernels, device=device, dtype=self.dtype, cutoff_nm=float(cutoff_nm),
            ewald_tolerance=float(ewald_tolerance), dispersion_correction=_boolean(dispersion_correction))
        self.box = [float(box_nm)]*3
        self.evaluators = {}
        if implementation == "torch":
            import torch
            from importlib import import_module
            make_evaluator = import_module("qtip4pf.torch").make_evaluator
            self.xp, self.factory = torch, make_evaluator
        else:
            import jax
            import jax.numpy as jnp
            from importlib import import_module
            make_evaluator = import_module("qtip4pf.jax").make_evaluator
            jax.config.update("jax_enable_x64", True)
            self.xp, self.factory = jnp, make_evaluator

    def compute(self, cell, pos):
        batched = isinstance(cell, list)
        if batched != isinstance(pos,list): raise ValueError("Cells and positions must both be lists for a batch")
        cells, positions = (cell,pos) if batched else ([cell],[pos])
        if len(cells) != len(positions) or not len(cells): raise ValueError("Nonempty matching batches required")
        groups = {}
        for i,(h,q) in enumerate(zip(cells,positions)):
            if q.shape[-1] != 3 or q.shape[0] % 3 or not q.shape[0]: raise ValueError("Expected O,H,H molecules")
            groups.setdefault(q.shape[0]//3,[]).append((i,h,q))
        results = [None]*len(cells)
        for nw, group in groups.items():
            xp = self.xp
            if self.implementation == "torch":
                def cast(x): return xp.as_tensor(x,dtype=getattr(xp,self.dtype),device=self.device)
                stack = lambda v: xp.stack(v)
            else:
                def cast(x): return xp.asarray(x,dtype=getattr(xp,self.dtype))
                stack = lambda v: xp.stack(v)
            if not self.backend_arrays:
                # Original i-PI supplies host arrays. Transfer one complete
                # batch rather than issuing a separate transfer for every bead.
                import numpy as np
                h = cast(np.stack([row[1] for row in group]))*NM_PER_BOHR
                q = cast(np.stack([row[2] for row in group]))*NM_PER_BOHR
            else:
                h = stack([cast(row[1]) for row in group])*NM_PER_BOHR
                q = stack([cast(row[2]) for row in group])*NM_PER_BOHR
            key = nw,len(group)
            if key not in self.evaluators:
                self.evaluators[key] = self.factory(nw,self.box,len(group),compiled=self.compiled,**self.options)
                if self.fixed_box:
                    self.evaluators[key] = self.evaluators[key].set_box(h)
            e,f,w = self.evaluators[key](q) if self.fixed_box else self.evaluators[key](q,h)
            e = e*HARTREE_PER_KJ_MOL
            f = f*(HARTREE_PER_KJ_MOL*NM_PER_BOHR)
            w = w*HARTREE_PER_KJ_MOL
            # i-PI retains fp64 state and accumulation for both precision modes.
            if self.implementation == "torch":
                e,f,w = (x.to(xp.float64) for x in (e,f,w))
                if not self.backend_arrays: e,f,w = (x.detach().cpu().numpy() for x in (e,f,w))
            else:
                e,f,w = (x.astype(xp.float64) for x in (e,f,w))
                if not self.backend_arrays:
                    import numpy as np
                    e,f,w = (np.asarray(x) for x in (e,f,w))
            for j,(i,_,_) in enumerate(group): results[i] = (e[j],f[j],w[j],"{}")
        return results if batched else results[0]

    __call__ = compute
