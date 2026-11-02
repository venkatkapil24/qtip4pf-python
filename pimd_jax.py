"""Functional JAX port of the validated Torch OBABO/PILE-L update."""
import jax
import jax.numpy as jnp

def remove_com(p, weights, half_inv_mtot):
    # Separate the fp64 momentum update from the strided column sum: combining
    # them made XLA's small reduction kernel slow in the P=1 GPU profile.
    p = jax.lax.optimization_barrier(p)
    pcom = p.reshape(-1, 3).sum(axis=0)
    dp = (weights * pcom).reshape(p.shape)
    ecom = (pcom @ pcom) * half_inv_mtot
    return p - dp, ecom



def thermostat(pnm, heat_by_mode, c, noise):
    p = pnm*c['damping_on_sm']
    heat = jnp.sum(p*p,axis=1)/c['damping2']
    p = p+c['noise_scale'][:,None]*noise
    heat = heat-jnp.sum(p*p,axis=1)
    return p*c['sm3'],heat_by_mode+heat*.5


def free_half_step(qnm, pnm, c):
    centroid = qnm[0] + pnm[0] * c["qdt_on_m"]
    if qnm.shape[0] == 1:
        return centroid[None, :], pnm
    a = c["propagator"]
    new_p = pnm[1:] * a[0, 0, 1:] + qnm[1:] * a[0, 1, 1:]
    new_q = pnm[1:] * a[1, 0, 1:] + qnm[1:] * a[1, 1, 1:]
    return (
        jnp.concatenate([centroid[None, :], new_q], axis=0),
        jnp.concatenate([pnm[:1], new_p], axis=0),
    )



def deterministic_step(state, c, force_function, noise):
    """Immutable MD update using the two supplied PILE-L Gaussian arrays."""
    pnm, heat = thermostat(state["pnm"], state["heat"], c, noise[0])
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
    pnm, heat = thermostat(pnm, heat, c, noise[1])
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





def step(state,c,force_function):
    key,a,b = jax.random.split(state['rng'],3)
    noise = (jax.random.normal(a,state['pnm'].shape,dtype=jnp.float64),
             jax.random.normal(b,state['pnm'].shape,dtype=jnp.float64))
    # Keep live fp64 draws inside this compiled step, but generate them in
    # parallel before the thermostat sums rather than inside their reductions.
    noise = jax.lax.optimization_barrier(noise)
    updated = deterministic_step(state,c,force_function,noise)
    updated['rng'] = key
    return updated


def initialize(config,beads):
    """Read common physical inputs, build fp64 coefficients, draw JAX momenta."""
    import math
    import numpy as np
    from benchmark import read_structure,MASS_DA,UNITS
    coords,box=read_structure(config['structure'])
    n=len(coords)
    q0=np.asarray(coords).reshape(-1)*(10*UNITS['angstrom'])
    mass=np.array([MASS_DA[a]*UNITS['amu'] for _ in range(n//3) for a in ('O','H','H')])
    m3=np.tile(np.repeat(mass,3),(beads,1));sm3=np.sqrt(m3)
    temp=config['temperature_K']*UNITS['kelvin'];dt=config['dt_fs']*4.1341373e16*1e-15
    tau=config['tau_fs']*4.1341373e16*1e-15
    omega=np.array([2*beads*temp*math.sin(k*math.pi/beads) for k in range(beads)])
    matrix=np.array([[(1. if k==0 else ((1. if j%2==0 else -1.) if k==beads//2 and beads%2==0 else math.sqrt(2)*(math.cos(2*math.pi*j*k/beads) if k<=beads//2 else math.sin(2*math.pi*j*k/beads))))/math.sqrt(beads) for j in range(beads)] for k in range(beads)])
    prop=np.zeros((2,2,beads,3*n));half=dt*.5
    prop[0,0]=1;prop[1,1]=1;prop[1,0,0]=half/m3[0]
    if beads>1:
        w=omega[1:,None];ang=w*half
        prop[0,0,1:]=np.cos(ang);prop[1,1,1:]=np.cos(ang)
        prop[0,1,1:]=-np.sin(ang)*w*m3[1:];prop[1,0,1:]=np.sin(ang)/(w*m3[1:])
    taus=np.r_[tau,1/(2*omega[1:])];damp=np.exp(-half/taus);d2=damp*damp
    coeff=dict(nm_matrix=matrix,sm3=sm3,propagator=prop,
        weights=3*m3.reshape(-1,3)/m3.sum(),half_inv_mtot=1.5/m3.sum(),
        qdt_on_m=half/m3[0],damping_on_sm=damp[:,None]/sm3,
        damping2=d2,noise_scale=np.sqrt(temp*beads*(1-d2)),dt=dt,pdt=half)
    coeff={k:jnp.asarray(v,dtype=jnp.float64) for k,v in coeff.items()}
    key,draw=jax.random.split(jax.random.key(config['seed']))
    pnm=jax.random.normal(draw,(beads,3*n),dtype=jnp.float64)*jnp.asarray(sm3)*math.sqrt(beads*temp)
    q=jnp.broadcast_to(jnp.asarray(q0),(beads,3*n))
    p,ecom=remove_com(coeff['nm_matrix'].T@pnm,coeff['weights'],coeff['half_inv_mtot'])
    state=dict(q=q,p=p,h=jnp.diag(jnp.asarray(box)*(10*UNITS['angstrom'])),
        qnm=coeff['nm_matrix']@q,pnm=coeff['nm_matrix']@p,heat=jnp.zeros(beads,dtype=jnp.float64),
        com_energy=ecom,rng=key,time=jnp.array(0.,dtype=jnp.float64))
    return state,coeff


import argparse
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import sys,time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from benchmark import ROOT,atomic_json,functional_parameters,load_config,result_record,total_steps

from importlib import import_module
KERNELS = os.environ.get("QTIP4PF_KERNELS", "array")
make_evaluator = import_module("qtip4pf.jax").make_evaluator
from qtip4pf import _model as pes


md = sys.modules[__name__]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--precision',choices=('mixed','double'),required=True)
    ap.add_argument('--nbeads',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();c=load_config(a.config);a.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();jax.config.update('jax_default_device',jax.devices('gpu')[0])
    state,coef=md.initialize(c,a.nbeads)
    dtype=jnp.float32 if a.precision=='mixed' else jnp.float64
    evaluate=make_evaluator(c['nwater'],c['box_nm'],a.nbeads,dtype='float32' if a.precision=='mixed' else 'float64',
        device='cuda',cutoff_nm=c['cutoff_nm'],ewald_tolerance=c['ewald_tolerance'],dispersion_correction=c['dispersion_correction'], kernels=KERNELS)
    evaluate=evaluate.set_box(jnp.broadcast_to((state['h'].astype(dtype)*pes.NM_PER_BOHR)[None],(a.nbeads,3,3)))
    def force(q,h):
        e,f,w=evaluate(q.reshape(a.nbeads,-1,3).astype(dtype)*pes.NM_PER_BOHR)
        return ((e*pes.HARTREE_PER_KJ_MOL).astype(jnp.float64),
            (f.reshape(q.shape)*(pes.HARTREE_PER_KJ_MOL*pes.NM_PER_BOHR)).astype(jnp.float64),
            (w*pes.HARTREE_PER_KJ_MOL).astype(jnp.float64))
    state['potential'],state['force'],state['virial']=force(state['q'],state['h'])
    # The loop replaces its state after every call and retains no old arrays.
    # Donation lets XLA reuse those buffers rather than allocate replacements.
    compiled=jax.jit(lambda s:md.step(s,coef,force),donate_argnums=(0,)).lower(state).compile()
    for _ in range(c['warmup_steps']):state=compiled(state)
    jax.block_until_ready(state);setup=time.perf_counter()-started;samples=[]
    for _ in range(c['repeats']):
        jax.block_until_ready(state);t=time.perf_counter_ns()
        for _ in range(c['steps_per_sample']):state=compiled(state)
        jax.block_until_ready(state);samples.append((time.perf_counter_ns()-t)*1e-9)
    for k in ('q','p','force','potential','virial'):
        assert np.isfinite(np.asarray(state[k])).all(),k
    assert abs(float(state['time'])-total_steps(c)*float(coef['dt']))<1e-7
    np.savez(a.output.parent/'final-state.npz',q=np.asarray(state['q']),p=np.asarray(state['p']))
    metadata=dict(precision=a.precision,potential_dtype=str(dtype),device='cuda',gpu=jax.devices('gpu')[0].device_kind,
        jax=jax.__version__,setup_seconds=setup,batch_size=a.nbeads,transform='matrix',propagator='exact',
        finite=True,whole_step_jit=True,state_buffer_reuse=True,gaussian_and_com_fusion_barriers=True,box_geometry='cached rectangular lengths and reciprocals',outputs_during_timing=False,checkpointing=False,
        compiled_scope='complete MD + random draws + live pair cutoff calculation + energy/force/full virial',
        timing_scope='synchronized whole-step jax.jit dispatch loop',extra_work='E/F/W computed and retained each step')
    atomic_json(a.output,result_record(c,'pure_jax',a.nbeads,samples,potential_kernels=KERNELS,**metadata))
    print(f'pure_jax {a.precision} P={a.nbeads}: {samples}',flush=True)


if __name__ == "__main__":
    main()
