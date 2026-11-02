"""Torch q-TIP4P/F, ordinary Ewald, direct pairs and analytic backward."""
import torch
from . import _model as parameters
from ._model import analytic

class _Ops:
    sqrt = staticmethod(torch.sqrt)
    floor = staticmethod(torch.floor)
    where = staticmethod(torch.where)
    exp = staticmethod(torch.exp)
    cos = staticmethod(torch.cos)
    sin = staticmethod(torch.sin)
    erf = staticmethod(torch.erf)
    erfc = staticmethod(torch.erfc)
    atan2 = staticmethod(torch.atan2)
    zeros_like = staticmethod(torch.zeros_like)
    broadcast_to = staticmethod(torch.broadcast_to)
    cross = staticmethod(torch.linalg.cross)
    @staticmethod
    def stack(x, axis=0): return torch.stack(x, dim=axis)
    @staticmethod
    def sum(x, axis): return x.sum(dim=axis)
    @staticmethod
    def prod(x, axis): return x.prod(dim=axis)
    @staticmethod
    def diagonal(x, axis1, axis2): return x.diagonal(dim1=axis1, dim2=axis2)
    @staticmethod
    def unbind(x, axis): return x.unbind(dim=axis)
    @staticmethod
    def arange(n, like): return torch.arange(n, device=like.device)
    @staticmethod
    def eye(n, like): return torch.eye(n, dtype=like.dtype, device=like.device)
    @staticmethod
    def index(x, maximum): return x.clamp(min=0, max=maximum).long()
    @staticmethod
    def scatter_add(x, index, values): return x.index_add(0, index, values)



class _Energy(torch.autograd.Function):
    @staticmethod
    def forward(q, h, evaluate):
        return evaluate(q, h)
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        q, h, _ = inputs
        _, f, w = outputs
        ctx.mark_non_differentiable(f, w)
        ctx.save_for_backward(q, h, f, w)
    @staticmethod
    def backward(ctx, de, df, dw):
        q, h, f, w = ctx.saved_tensors
        dh = (f.transpose(-1,-2) @ q - w) / h.diagonal(dim1=-2,dim2=-1)[:,None,:]
        return -f*de[:,None,None], dh*de[:,None,None], None


def make_evaluator(nwater, box_nm, batch_size=1, dtype="float64", device="cuda",
                   cutoff_nm=0.6, ewald_tolerance=5e-4,
                   dispersion_correction=True, compiled=False, kernels="array"):
    """Return evaluate(q,h) -> E,F,W in nm/kJ/mol units.

    q: (B,3*Nwater,3), h: (B,3,3), orthorhombic columns as lattice vectors.
    Molecules must be whole. Direct mode stores all possible half pairs and
    checks their distances and cutoff membership on every call.
    First energy derivatives are available as evaluate.energy.
    """
    if dtype not in ("float32", "float64"): raise ValueError("dtype must be float32 or float64")
    if batch_size < 1: raise ValueError("batch_size must be positive")
    if kernels not in ("array", "triton"): raise ValueError("kernels must be array or triton")
    if kernels == "triton" and not device.startswith("cuda"): raise ValueError("Triton requires CUDA")
    dt = getattr(torch, dtype)
    plan = parameters.make_plan(nwater, box_nm, cutoff_nm, ewald_tolerance,
        dispersion_correction, tensor=lambda v: torch.tensor(v, dtype=dt, device=device))
    nonbonded = None if kernels == "array" else _nonbonded_factory(plan)
    complete_matrices = []
    for atoms in (3*nwater,nwater):
        rows = torch.arange(batch_size*atoms,dtype=torch.int32,device=device)[:,None]
        columns = rows+1+torch.arange(max(1,atoms-1),dtype=torch.int32,device=device)[None,:]
        # Every i<j pair within each structure is retained. The live distance
        # and cutoff test decides whether it contributes on each step.
        complete_matrices.append(torch.where(columns < (rows//atoms+1)*atoms,
                                              columns,batch_size*atoms))
    def evaluate(q, h, inverse_lengths=None, callback=nonbonded):
        assert q.shape == (batch_size, 3*nwater, 3) and h.shape == (batch_size,3,3)
        assert q.dtype == dt and h.dtype == dt
        if inverse_lengths is None:
            inverse_lengths = 1.0/h.diagonal(dim1=-2,dim2=-1)
        return analytic(q,h,plan,*complete_matrices,_Ops,inverse_lengths,nonbonded=callback)
    result = torch.compile(evaluate,fullgraph=True,dynamic=False,
        options={"triton.cudagraphs":False}) if compiled else evaluate
    result.energy = lambda q,h: _Energy.apply(q,h,evaluate)[0]
    result.plan = plan
    def set_box(h):
        """Return a new fixed-box evaluator; call again after any box change.

        Call outside torch.compile/CUDA capture. The returned function stores
        the box and reciprocal lengths and only accepts positions.
        """
        cells = h.detach().cpu().tolist()
        if len(cells) != batch_size: raise ValueError("Box batch size does not match evaluator")
        lengths = torch.tensor(parameters.rectangular_lengths(cells,cutoff_nm),dtype=dt,device=device)
        cell = torch.diag_embed(lengths)
        inverse = 1.0/lengths
        callback = None if nonbonded is None else nonbonded.for_box(cell,inverse)
        def bound(q): return evaluate(q,cell,inverse,callback)
        fixed = torch.compile(bound,fullgraph=True,dynamic=False,
            options={"triton.cudagraphs":False}) if compiled else bound
        fixed.energy = lambda q: _Energy.apply(q,cell,lambda q,h:evaluate(q,h,inverse,callback))[0]
        fixed.set_box = set_box
        fixed.plan = plan
        return fixed
    result.set_box = set_box
    return result


def _nonbonded_factory(plan):
    import triton
    from . import _triton as kernels
    from ._model import COULOMB
    # Torch is imported above; these are native compiler-visible Triton calls.
    def prepare(h,inverse):
        return tuple(x.contiguous() for x in kernels.reciprocal_geometry(plan,h,inverse,_Ops))
    def make(cached=None):
        def calculate(charged,oxygen,h,inverse):
            lengths,wave,a = prepare(h,inverse) if cached is None else cached
            oxygen = oxygen.contiguous()
            inverse = inverse.contiguous()
            b,n,_ = charged.shape
            no,k = oxygen.shape[1],wave.shape[1]
            cf,of,rf = torch.empty_like(charged),torch.empty_like(oxygen),torch.empty_like(charged)
            ce = torch.empty((b,n,10),dtype=charged.dtype,device=charged.device)
            oe = torch.empty((b,no,10),dtype=charged.dtype,device=charged.device)
            ke = torch.empty((b,k,10),dtype=charged.dtype,device=charged.device)
            sums = torch.empty((b,k,2),dtype=charged.dtype,device=charged.device)
            totals = torch.empty((b,12),dtype=charged.dtype,device=charged.device)
            args=dict(C=COULOMB,ALPHA=plan.alpha,SIGMA=plan.params['sigma'],
                EPSILON=plan.params['epsilon'],CUTOFF2=plan.cutoff**2,enable_fp_fusion=False)
            kernels.pairs[(n,b)](charged,plan.charges,lengths,inverse,cf,ce,N=n,
                ELECTROSTATIC=True,BLOCK=min(256,triton.next_power_of_2(n)),num_warps=4,**args)
            kernels.pairs[(no,b)](oxygen,plan.charges,lengths,inverse,of,oe,N=no,
                ELECTROSTATIC=False,BLOCK=min(256,triton.next_power_of_2(no)),num_warps=4,**args)
            kernels.structure_factors[(k,b)](charged,plan.charges,wave,a,sums,ke,
                N=n,K=k,ALPHA=plan.alpha,BLOCK=min(256,triton.next_power_of_2(n)),
                num_warps=4,enable_fp_fusion=False)
            kernels.reciprocal_force[(n,b)](charged,plan.charges,wave,a,sums,rf,
                N=n,K=k,BLOCK=min(1024,triton.next_power_of_2(k)),num_warps=4,enable_fp_fusion=False)
            maximum=max(n,no,k)
            kernels.reduce_terms[(b,12)](ce,oe,ke,totals,NC=n,NO=no,K=k,MAXIMUM=maximum,
                BLOCK=min(1024,triton.next_power_of_2(maximum)),num_warps=4,enable_fp_fusion=False)
            return totals[:,0],totals[:,1],totals[:,2],cf+rf,of,totals[:,3:].reshape(b,3,3)
        return calculate
    result=make()
    result.for_box=lambda h,inverse:make(prepare(h,inverse))
    return result
