"""JAX q-TIP4P/F, ordinary Ewald, direct pairs."""
import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc
from . import _model as parameters
from ._model import analytic

class _Ops:
    sqrt = staticmethod(jnp.sqrt)
    floor = staticmethod(jnp.floor)
    where = staticmethod(jnp.where)
    exp = staticmethod(jnp.exp)
    cos = staticmethod(jnp.cos)
    sin = staticmethod(jnp.sin)
    erf = staticmethod(erf)
    erfc = staticmethod(erfc)
    atan2 = staticmethod(jnp.atan2)
    zeros_like = staticmethod(jnp.zeros_like)
    broadcast_to = staticmethod(jnp.broadcast_to)
    cross = staticmethod(jnp.cross)
    stack = staticmethod(jnp.stack)
    sum = staticmethod(jnp.sum)
    prod = staticmethod(jnp.prod)
    diagonal = staticmethod(jnp.diagonal)
    @staticmethod
    def unbind(x, axis): return tuple(jnp.take(x,i,axis=axis) for i in range(x.shape[axis]))
    @staticmethod
    def arange(n, like): return jnp.arange(n,dtype=jnp.int32)
    @staticmethod
    def eye(n, like): return jnp.eye(n,dtype=like.dtype)
    @staticmethod
    def index(x, maximum): return jnp.clip(x,0,maximum)
    @staticmethod
    def scatter_add(x, index, values): return x.at[index].add(values)


def make_evaluator(nwater, box_nm, batch_size=1, dtype="float64", device="cuda",
                   cutoff_nm=0.6, ewald_tolerance=5e-4,
                   dispersion_correction=True, compiled=False, kernels="array"):
    """Same units, shapes and supported cells as qtip4pf.torch.make_evaluator."""
    if dtype not in ("float32","float64"): raise ValueError("dtype must be float32 or float64")
    if batch_size < 1: raise ValueError("batch_size must be positive")
    if kernels not in ("array", "triton"): raise ValueError("kernels must be array or triton")
    if kernels == "triton" and not device.startswith("cuda"): raise ValueError("Triton requires CUDA")
    # Explicitly required for the double potential and mixed/double integration.
    jax.config.update("jax_enable_x64",True)
    target = jax.devices("gpu" if device.startswith("cuda") else "cpu")[int(device.partition(":")[2] or 0)]
    dt = getattr(jnp,dtype)
    def array(v,dtype=dt): return jax.device_put(jnp.asarray(v,dtype=dtype),target)
    plan = parameters.make_plan(nwater, box_nm, cutoff_nm, ewald_tolerance,
        dispersion_correction, tensor=array)
    nonbonded = None if kernels == "array" else _nonbonded_factory(plan)
    complete_matrices = []
    for atoms in (3*nwater,nwater):
        rows = array(list(range(batch_size*atoms)),jnp.int32)[:,None]
        columns = rows+1+array(list(range(max(1,atoms-1))),jnp.int32)[None,:]
        complete_matrices.append(jnp.where(columns < (rows//atoms+1)*atoms,
                                            columns,batch_size*atoms))
    def evaluate(q,h,inverse_lengths=None,callback=nonbonded):
        assert q.shape == (batch_size,3*nwater,3) and h.shape == (batch_size,3,3)
        assert q.dtype == dt and h.dtype == dt
        if inverse_lengths is None:
            inverse_lengths = 1.0/jnp.diagonal(h,axis1=-2,axis2=-1)
        return analytic(q,h,plan,*complete_matrices,_Ops,inverse_lengths,nonbonded=callback)
    @jax.custom_vjp
    def energy(q,h): return evaluate(q,h)[0]
    def forward(q,h):
        e,f,w = evaluate(q,h)
        return e,(q,h,f,w)
    def backward(saved,de):
        q,h,f,w = saved
        dh = (jnp.swapaxes(f,-1,-2) @ q - w) / jnp.diagonal(h,axis1=-2,axis2=-1)[:,None,:]
        return -f*de[:,None,None],dh*de[:,None,None]
    energy.defvjp(forward,backward)
    result = jax.jit(evaluate) if compiled else evaluate
    result.energy = energy
    result.plan = plan
    def set_box(h):
        """Return a new fixed-box evaluator; call again after any box change.

        Do this outside jax.jit. Reciprocal lengths are stored in the returned
        function; pair distances and cutoff membership remain live on each call.
        """
        cells = jax.device_get(h).tolist()
        if len(cells) != batch_size: raise ValueError("Box batch size does not match evaluator")
        lengths = array(parameters.rectangular_lengths(cells,cutoff_nm))
        cell = lengths[:,:,None]*jnp.eye(3,dtype=dt)[None]
        inverse = 1.0/lengths
        callback = None if nonbonded is None else nonbonded.for_box(cell,inverse)
        def bound(q): return evaluate(q,cell,inverse,callback)
        fixed = jax.jit(bound) if compiled else bound
        @jax.custom_vjp
        def fixed_energy(q): return bound(q)[0]
        def fixed_forward(q):
            e,f,_ = bound(q)
            return e,f
        def fixed_backward(f,de): return (-f*de[:,None,None],)
        fixed_energy.defvjp(fixed_forward,fixed_backward)
        fixed.energy = fixed_energy
        fixed.set_box = set_box
        fixed.plan = plan
        return fixed
    result.set_box = set_box
    return result


def _nonbonded_factory(plan):
    import triton
    from . import _triton as kernels
    from ._model import COULOMB
    import jax_triton as jt
    def prepare(h,inverse):return kernels.reciprocal_geometry(plan,h,inverse,_Ops)
    def make(cached=None):
        def calculate(charged,oxygen,h,inverse):
            lengths,wave,a = prepare(h,inverse) if cached is None else cached
            b,n,_=charged.shape
            no,k=oxygen.shape[1],wave.shape[1]
            def shape(*dims):return jax.ShapeDtypeStruct(dims,charged.dtype)
            args=dict(C=COULOMB,ALPHA=plan.alpha,SIGMA=plan.params['sigma'],
                EPSILON=plan.params['epsilon'],CUTOFF2=plan.cutoff**2,enable_fp_fusion=False)
            cf,ce=jt.triton_call(charged,plan.charges,lengths,inverse,kernel=kernels.pairs,
                out_shape=(shape(b,n,3),shape(b,n,10)),grid=(n,b),N=n,
                ELECTROSTATIC=True,BLOCK=min(256,triton.next_power_of_2(n)),num_warps=4,**args)
            of,oe=jt.triton_call(oxygen,plan.charges,lengths,inverse,kernel=kernels.pairs,
                out_shape=(shape(b,no,3),shape(b,no,10)),grid=(no,b),N=no,
                ELECTROSTATIC=False,BLOCK=min(256,triton.next_power_of_2(no)),num_warps=4,**args)
            sums,ke=jt.triton_call(charged,plan.charges,wave,a,kernel=kernels.structure_factors,
                out_shape=(shape(b,k,2),shape(b,k,10)),grid=(k,b),N=n,K=k,ALPHA=plan.alpha,
                BLOCK=min(256,triton.next_power_of_2(n)),num_warps=4,enable_fp_fusion=False)
            rf=jt.triton_call(charged,plan.charges,wave,a,sums,kernel=kernels.reciprocal_force,
                out_shape=shape(b,n,3),grid=(n,b),N=n,K=k,
                BLOCK=min(1024,triton.next_power_of_2(k)),num_warps=4,enable_fp_fusion=False)
            maximum=max(n,no,k)
            totals=jt.triton_call(ce,oe,ke,kernel=kernels.reduce_terms,out_shape=shape(b,12),
                grid=(b,12),NC=n,NO=no,K=k,MAXIMUM=maximum,
                BLOCK=min(1024,triton.next_power_of_2(maximum)),num_warps=4,enable_fp_fusion=False)
            return totals[:,0],totals[:,1],totals[:,2],cf+rf,of,totals[:,3:].reshape(b,3,3)
        return calculate
    result=make()
    result.for_box=lambda h,inverse:make(prepare(h,inverse))
    return result
