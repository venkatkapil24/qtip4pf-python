"""Independent energy/force/strain comparisons, not an MD benchmark launcher."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

HERE = Path(__file__).resolve().parent


def structure(nwater, seed=1):
    rng = np.random.default_rng(seed)
    side = math.ceil(nwater**(1/3))
    coords = []
    for i in range(nwater):
        center = np.array([i%side, (i//side)%side, i//side**2])*0.43+0.35
        center += rng.normal(scale=0.01, size=3)
        r1, r2 = 0.09419*(1+rng.normal(scale=0.03, size=2))
        theta = 1.87448361664+rng.normal(scale=0.03)
        orientation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        molecule = np.array([[0, 0, 0], [r1, 0, 0],
                             [r2*math.cos(theta), r2*math.sin(theta), 0]])
        coords.extend(molecule@orientation+center)
    return np.array(coords)


def context(nwater, box, platform_name="Reference", tail=True, cutoff=.6):
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(nwater):
        residue = topology.addResidue("HOO", chain)
        atoms = [topology.addAtom(name, element, residue) for name, element in
                 [("O", app.element.oxygen), ("H1", app.element.hydrogen),
                  ("H2", app.element.hydrogen), ("M", None)]]
        topology.addBond(atoms[0], atoms[1])
        topology.addBond(atoms[0], atoms[2])
    topology.setPeriodicBoxVectors([mm.Vec3(*v)*unit.nanometer for v in np.diag(box)])
    forcefield = app.ForceField(str(Path(__file__).resolve().parents[1]/"data/qtip4pf.xml"))
    system = forcefield.createSystem(topology, nonbondedMethod=app.Ewald,
        nonbondedCutoff=cutoff*unit.nanometer, ewaldErrorTolerance=5e-4,
        constraints=None, rigidWater=False, removeCMMotion=False)
    for force in system.getForces():
        if isinstance(force, mm.CustomBondForce):
            force.setForceGroup(0)
        elif isinstance(force, mm.HarmonicAngleForce):
            force.setForceGroup(1)
        elif isinstance(force, mm.NonbondedForce):
            force.setForceGroup(2)
            force.setReciprocalSpaceForceGroup(3)
            force.setUseDispersionCorrection(tail)
    integrator = mm.VerletIntegrator(0.0005)
    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {"Precision": "double", "DeviceIndex": "0"} if platform_name == "CUDA" else {}
    ctx = mm.Context(system, integrator, platform, properties)
    return ctx, integrator


def reference(ctx, q, box, groups=-1):
    lengths = np.asarray(box)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*v)*unit.nanometer for v in np.diag(lengths)])
    extended = np.zeros((q.shape[0]//3, 4, 3))
    extended[:, :3] = q.reshape(-1, 3, 3)
    ctx.setPositions(extended.reshape(-1, 3)*unit.nanometer)
    ctx.computeVirtualSites()
    state = ctx.getState(getEnergy=True, getForces=True, groups=groups)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    return e, f.reshape(-1, 4, 3)[:, :3].reshape(-1, 3)


def analytical_virial(ctx, no_tail_ctx, q, box, alpha, kmax, cutoff=.9):
    """Independent full tensor: OpenMM direct forces plus analytic Ewald stress.

    Periodic image corrections convert sum F_i outer r_i into the direct-space
    strain derivative. OpenMM supplies direct forces; NumPy supplies only the
    image correction and reciprocal stress, independently of the Torch code.
    Reciprocal stress follows differentiation at fixed fractional coordinates:
    sum_k E_k [I-2(1/k^2+1/(4 alpha^2)) k outer k].
    """
    _, direct_force = reference(ctx, q, box, (1<<0)|(1<<1)|(1<<2))
    water = q.reshape(-1, 3, 3)
    m = .73612*water[:, 0]+.13194*(water[:, 1]+water[:, 2])
    sites = np.stack((water[:, 1], water[:, 2], m), axis=1).reshape(-1, 3)
    charges = np.tile([.5564, .5564, -1.1128], len(water))
    direct_virial = direct_force.T@q
    for points, electrostatic in ((sites, True), (water[:,0], False)):
        i,j = np.triu_indices(len(points),1)
        if not len(i):
            continue
        raw = points[i]-points[j]
        image = np.floor(raw/box+.5)*box
        delta = raw-image
        distance = np.linalg.norm(delta,axis=1)
        active = distance<cutoff
        if electrostatic:
            active &= (i//3 != j//3)
        d=distance[active];v=delta[active];im=image[active]
        if electrostatic:
            from scipy.special import erfc
            coefficient=138.93545764438198*charges[i[active]]*charges[j[active]]*(
                erfc(alpha*d)/d+2*alpha/np.sqrt(np.pi)*np.exp(-(alpha*d)**2))/d**2
        else:
            sr6=(.31589/d)**6
            coefficient=24*.7749*sr6*(2*sr6-1)/d**2
        direct_virial -= np.einsum('ni,nj->ij',coefficient[:,None]*v,im)
    integers = np.array([(x,y,z) for x in range(kmax[0])
                         for y in range(1-kmax[1],kmax[1])
                         for z in range(1-kmax[2],kmax[2])
                         if (x,y,z) > (0,0,0)])
    k = 2*np.pi*integers/box
    k2 = np.sum(k*k, axis=1)
    structure_factor = charges @ np.exp(1j*(sites@k.T))
    weights = 4*np.pi*138.93545764438198/np.prod(box)*np.exp(-k2/(4*alpha**2))/k2*np.abs(structure_factor)**2
    reciprocal = np.eye(3)*weights.sum()-2*np.einsum(
        'k,ki,kj->ij', weights*(1/k2+1/(4*alpha**2)), k, k)
    tail = reference(ctx, q, box)[0]-reference(no_tail_ctx, q, box)[0]
    return direct_virial+reciprocal+np.eye(3)*tail, tail




import importlib,json,math,os
from pathlib import Path
import numpy as np
import openmm
import pytest
import torch
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64',True)
HERE=Path(__file__).resolve().parent.parent
ROWS=[]

@pytest.fixture(autouse=True)
def independent_compile_cases():
    # Each test deliberately constructs distinct static shapes/options. Reset
    # Dynamo's eight-specialization limit between tests, not during timed MD.
    torch._dynamo.reset()



def reciprocal_parameters(box,cutoff=.6,tolerance=.0005):
    """Independent specification of OpenMM's initialization, not production helpers."""
    alpha=math.sqrt(-math.log(2*tolerance))/cutoff
    maxima=[]
    for length in box:
        def err(k):return .05*math.sqrt(length*alpha)*k*math.exp(-(k*math.pi/(length*alpha))**2)
        # Select the falling tail of the truncation bound; OpenMM requires odd limits.
        peak=max(1,int(length*alpha/(math.pi*math.sqrt(2))))
        k=peak
        while err(k)>tolerance:k+=1
        maxima.append(k if k%2 else k+1)
    return alpha,tuple(maxima)


def arrays(backend,q,h,dtype):
    if backend=='torch':
        dt=getattr(torch,dtype)
        return torch.tensor(q,dtype=dt,device='cuda'),torch.tensor(h,dtype=dt,device='cuda')
    return jax.device_put(jnp.asarray(q,dtype=getattr(jnp,dtype)),jax.devices('gpu')[0]),jax.device_put(jnp.asarray(h,dtype=getattr(jnp,dtype)),jax.devices('gpu')[0])


def host(xs):
    return [x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x) for x in xs]


def evaluate(backend,dtype,q,h,nwater,box,tail=True,compiled=True):
    mod=importlib.import_module('qtip4pf.'+backend)
    f=mod.make_evaluator(nwater,box,len(q),dtype=dtype,device='cuda',
        cutoff_nm=.6,dispersion_correction=tail,compiled=compiled)
    x,y=arrays(backend,q,h,dtype)
    return f,host(f(x,y))


@pytest.mark.parametrize('backend',['torch','jax'])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('nwater',[1,2,64])
def test_external_energy_force_and_complete_virial(backend,dtype,nwater):
    if nwater==64:
        lines=(HERE/'data/water64.xyz').read_text().splitlines()
        base=np.array([[float(v) for v in l.split()[1:4]] for l in lines[2:194]])/10
        box=np.array([1.24138]*3)
    else:
        base=structure(nwater,seed=29);box=np.array([2.4,2.5,2.6])
    q=np.stack([base,base+np.sin(np.arange(base.size)).reshape(base.shape)*.0002])
    boxes=np.stack([box,box*[1.002,1.001,1.003]])
    h=np.array([np.diag(b) for b in boxes])
    f,out=evaluate(backend,dtype,q,h,nwater,box)
    x,y=arrays(backend,q,h,dtype)
    fixed=f.set_box(y)
    fixed_out=host(fixed(x))
    ctx,integ=context(nwater,box,tail=True)
    nt,nint=context(nwater,box,tail=False)
    alpha,kmax=reciprocal_parameters(box)
    # Numeric tolerances allow fp32 cancellation of reciprocal/self/exclusion
    # energies and pair force accumulation; no fp64 work is substituted.
    atol=(2e-4*nwater,.03,.03) if dtype=='float32' else (2e-8,2e-7,2e-7)
    rtol=3e-5 if dtype=='float32' else 2e-10
    for b in range(2):
        rq=q[b].astype(dtype).astype('float64'); rb=boxes[b].astype(dtype).astype('float64')
        e,ff=reference(ctx,rq,rb);w,tail=analytical_virial(ctx,nt,rq,rb,alpha,kmax,cutoff=.6)
        expected=(np.array(e),ff,w)
        differences=[float(np.max(np.abs(x[b]-y))) for x,y in zip(out,expected)]
        ROWS.append(dict(backend=backend,dtype=dtype,nwater=nwater,batch=b,energy_abs=differences[0],force_max_abs=differences[1],virial_max_abs=differences[2]))
        for values in (out,fixed_out):
            for x,y,tol in zip(values,expected,atol):np.testing.assert_allclose(x[b],y,rtol=rtol,atol=tol)
    # Reuse the same compiled executable with changed coordinates and live cells.
    moved=q.reshape(2,nwater,3,3).copy();moved[:,0]+=boxes[:,None,:]*[1,-1,2]
    x,y=arrays(backend,moved.reshape(q.shape),h,dtype)
    translated=host(f(x,y))
    fixed_translated=host(fixed(x))
    for b in range(2):
        rq=moved.reshape(q.shape)[b].astype(dtype).astype('float64'); rb=boxes[b].astype(dtype).astype('float64')
        ee,ff=reference(ctx,rq,rb); ww,_=analytical_virial(ctx,nt,rq,rb,alpha,kmax,cutoff=.6)
        for values in (translated,fixed_translated):
            for x,y,tol in zip(values,(ee,ff,ww),atol):np.testing.assert_allclose(x[b],y,rtol=rtol,atol=tol)
    # fp32 translated coordinates themselves are quantized differently. Compare
    # each against the external reference at its actual represented coordinates;
    # preserve the observed translation difference as evidence, not a false bug.
    ROWS[-1]['translation_differences']=[float(np.max(np.abs(x-y))) for x,y in zip(translated,out)]
    if dtype=='float64':
        for x,y,tol in zip(translated,out,atol):np.testing.assert_allclose(x,y,rtol=rtol,atol=tol*2)
    # Batch partition: independently constructed single calls retain order/results.
    for b in range(2):
        _,single=evaluate(backend,dtype,q[b:b+1],h[b:b+1],nwater,box)
        for x,y,tol in zip(single,out,atol):np.testing.assert_allclose(x[0],y[b],rtol=rtol,atol=tol)
    # Dispersion is checked independently by toggling the OpenMM System term.
    _,no_tail=evaluate(backend,dtype,q[:1],h[:1],nwater,box,tail=False)
    tail=reference(ctx,q[0],box)[0]-reference(nt,q[0],box)[0]
    np.testing.assert_allclose(out[0][0]-no_tail[0][0],tail,atol=atol[0],rtol=rtol)
    np.testing.assert_allclose(out[2][0]-no_tail[2][0],np.eye(3)*tail,atol=atol[2],rtol=rtol)
    del ctx,integ,nt,nint


@pytest.mark.parametrize('backend',['torch','jax'])
def test_backward_and_energy_derivatives(backend):
    box=np.array([2.4,2.5,2.6]);q=structure(2,seed=13)[None];h=np.diag(box)[None]
    f,out=evaluate(backend,'float64',q,h,2,box,compiled=False)
    x,y=arrays(backend,q,h,'float64')
    if backend=='torch':
        x.requires_grad_(True);y.requires_grad_(True)
        dq,dh=torch.autograd.grad(f.energy(x,y).sum(),(x,y));dq,dh=host((dq,dh))
    else:
        dq,dh=host(jax.grad(lambda a,b:f.energy(a,b).sum(),argnums=(0,1))(x,y))
    np.testing.assert_allclose(-dq,out[1],rtol=2e-10,atol=2e-8)
    np.testing.assert_allclose(np.swapaxes(out[1],-1,-2)@q-dh@np.swapaxes(h,-1,-2),out[2],rtol=2e-10,atol=2e-8)
    fixed=f.set_box(y)
    if backend=='torch':
        fixed_dq=host(torch.autograd.grad(fixed.energy(x).sum(),x))[0]
    else:fixed_dq=np.asarray(jax.grad(lambda a:fixed.energy(a).sum())(x))
    np.testing.assert_allclose(-fixed_dq,out[1],rtol=2e-10,atol=2e-8)
    ctx,integ=context(2,box)
    # External energy derivative for one nonzero coordinate.
    step=1e-6;plus=q[0].copy();minus=plus.copy();plus[1,0]+=step;minus[1,0]-=step
    fd=-(reference(ctx,plus,box)[0]-reference(ctx,minus,box)[0])/(2*step)
    np.testing.assert_allclose(out[1][0,1,0],fd,rtol=2e-7,atol=2e-5)
    # Independent diagonal strain derivative includes explicit volume dependence.
    for a in range(3):
        scale=np.ones(3);scale[a]+=step;ep=reference(ctx,q[0]*scale,box*scale)[0]
        scale[a]-=2*step;em=reference(ctx,q[0]*scale,box*scale)[0]
        np.testing.assert_allclose(out[2][0,a,a],-(ep-em)/(2*step),rtol=2e-7,atol=2e-5)
    del ctx,integ


def teardown_module():
    out=Path(os.environ.get('QTIP4PF_VALIDATION_REPORT',str(HERE/'validation.json')))
    out.write_text(json.dumps(dict(rows=ROWS,openmm=openmm.__version__,torch=torch.__version__,jax=jax.__version__),indent=2)+'\n')

@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('fixed_box',[False,True])
def test_torch_cuda_capture_rebuilds_neighbors(dtype,fixed_box):
    """Move waters across the cutoff after capture; compare to OpenMM."""
    from qtip4pf.torch import make_evaluator
    box=np.array([2.4]*3);q=np.stack([structure(2,13),structure(2,29)])
    h=np.broadcast_to(np.diag(box),(2,3,3)).copy()
    q=q.reshape(2,2,3,3);q[:,1,:,0]+=.7;q=q.reshape(2,6,3)
    x,y=arrays('torch',q,h,dtype)
    f=make_evaluator(2,box,2,dtype=dtype,compiled=True)
    if fixed_box:
        f=f.set_box(y)
        call=lambda:f(x)
    else:call=lambda:f(x,y)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):call()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=call()
    # Start outside the .6 nm cutoff; move water two INTO it after capture.
    # A frozen list would omit these newly interacting pairs entirely.
    changed=q.reshape(2,2,3,3).copy();changed[:,1,:,0]-=.7
    x.copy_(torch.tensor(changed.reshape(q.shape),dtype=x.dtype,device=x.device))
    if not fixed_box:y.mul_(1.001)
    graph.replay();torch.cuda.synchronize();out=host(output)
    ctx,integ=context(2,box)
    actual_q=x.cpu().numpy().astype('float64');actual_box=y[0].diag().cpu().numpy().astype('float64')
    for b in range(2):
        e,ff=reference(ctx,actual_q[b],actual_box)
        np.testing.assert_allclose(out[0][b],e,atol=.001 if dtype=='float32' else 2e-8,rtol=3e-5 if dtype=='float32' else 2e-10)
        np.testing.assert_allclose(out[1][b],ff,atol=.03 if dtype=='float32' else 2e-7,rtol=3e-5 if dtype=='float32' else 2e-10)
    del ctx,integ


@pytest.mark.parametrize('backend',['torch','jax'])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_setting_new_rectangular_box_refreshes_geometry(backend,dtype):
    """A new box uses new reciprocals; both versions match an external oracle."""
    box=np.array([2.4,2.5,2.6]);q=structure(2,seed=29)[None]
    mod=importlib.import_module('qtip4pf.'+backend)
    evaluator=mod.make_evaluator(2,box,1,dtype=dtype,compiled=True)
    ctx,integ=context(2,box);nt,nint=context(2,box,tail=False)
    alpha,kmax=reciprocal_parameters(box)
    atol=(.001,.03,.03) if dtype=='float32' else (2e-8,2e-7,2e-7)
    rtol=3e-5 if dtype=='float32' else 2e-10
    for lengths in (box,box*[1.03,1.04,1.05]):
        h=np.diag(lengths)[None];x,y=arrays(backend,q,h,dtype)
        fixed=evaluator.set_box(y)
        # A whole molecule moves by integer lattice vectors; its original
        # coordinates and the cached periodic wrapping must remain consistent.
        moved=q.reshape(1,2,3,3).copy();moved[:,1]+=lengths*[2,-3,1]
        x,_=arrays(backend,moved.reshape(q.shape),h,dtype)
        actual=host(fixed(x));rq=moved.reshape(q.shape)[0].astype(dtype).astype('float64')
        rb=lengths.astype(dtype).astype('float64')
        e,ff=reference(ctx,rq,rb);ww,_=analytical_virial(ctx,nt,rq,rb,alpha,kmax,cutoff=.6)
        for a,b,t in zip(actual,(e,ff,ww),atol):np.testing.assert_allclose(a[0],b,rtol=rtol,atol=t)
    # Reject an actual tilted cell instead of silently using its diagonal.
    skew=np.diag(box)[None];skew[0,0,1]=.1
    _,y=arrays(backend,q,skew,dtype)
    with pytest.raises(ValueError,match='rectangular'):evaluator.set_box(y)
    del ctx,integ,nt,nint


import importlib
from pathlib import Path
import numpy as np
import pytest
import ipi
from ipi.engine.beads import Beads
from ipi.engine.cell import Cell
from ipi.engine.forcefields import FFDirect
from ipi.utils.array_backend import set_backend,to_numpy,xp
# Independent literals from pinned i-PI UnitMap; never import the tested
# adapter's conversion constants as their own expected values.
NM_PER_BOHR=1/(10*1.8897261)
HARTREE_PER_KJ_MOL=1000*.00000038087989

HERE=Path(__file__).resolve().parent.parent

@pytest.mark.parametrize('kernels',['array','triton'])
@pytest.mark.parametrize('host_output',[False,True])
@pytest.mark.parametrize('backend',['torch','jax'])
@pytest.mark.parametrize('precision',['mixed','double'])
@pytest.mark.parametrize('fixed_box',[False,True])
def test_real_ffdirect_batch(backend,precision,host_output,fixed_box,kernels):
    set_backend('numpy' if host_output else backend,device='cpu' if host_output else 'cuda')
    box=np.array([2.4,2.5,2.6]);q=np.stack([structure(2,13),structure(2,29)])
    # Adapter box_nm is cubic; these noncubic live cells are permitted and keep
    # the initialized cubic Ewald grid. Use cubic reference here for comparison.
    box=np.array([2.4]*3)
    beads=Beads(6,2);beads.q_[:]=xp.asarray(q.reshape(2,-1)/NM_PER_BOHR)
    cell=Cell(h=np.diag(box)/NM_PER_BOHR)
    ff=FFDirect(name='testwater',pes='custom',pes_path=str(HERE/'qtip4pf/ffdirect.py'),batch_size=2,
        pars=dict(implementation=backend,precision=precision,kernels=kernels,device='cuda',compiled='true',fixed_box=str(fixed_box).lower(),host_output=str(host_output).lower(),box_nm='2.4',cutoff_nm='.6'))
    requests=[ff.queue(beads[b],cell) for b in range(2)]
    ff.poll()
    assert all(r['status']=='Done' for r in requests)
    ctx,integ=context(2,box)
    nt,nint=context(2,box,tail=False)
    alpha,kmax=reciprocal_parameters(box)
    tol=.03 if precision=='mixed' else 2e-7
    for b,r in enumerate(requests):
        e,f,w,_=r['result']
        ee,ff_expected=reference(ctx,q[b],box)
        np.testing.assert_allclose(float(to_numpy(e))/HARTREE_PER_KJ_MOL,ee,atol=.001 if precision=='mixed' else 2e-8,rtol=3e-5)
        np.testing.assert_allclose(to_numpy(f).reshape(6,3)/(HARTREE_PER_KJ_MOL*NM_PER_BOHR),ff_expected,atol=tol,rtol=3e-5 if precision=='mixed' else 2e-10)
        ww,_=analytical_virial(ctx,nt,q[b],box,alpha,kmax,cutoff=.6)
        np.testing.assert_allclose(to_numpy(w)/HARTREE_PER_KJ_MOL,ww,atol=tol,rtol=3e-5 if precision=='mixed' else 2e-10)
        ff.release(r)
    assert len(ff.requests)==0
    ff.stop();del ctx,integ,nt,nint


import os,subprocess,sys
import pytest

@pytest.mark.parametrize('backend',['torch','jax'])
def test_only_selected_framework_required(backend):
    other='jax' if backend=='torch' else 'torch'
    script=f'''
import importlib.abc,sys
class BlockOther(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="{other}" or fullname.startswith("{other}."):
            raise ModuleNotFoundError("Other framework intentionally unavailable",name=fullname)
sys.meta_path.insert(0,BlockOther())
from qtip4pf.{backend} import make_evaluator
f=make_evaluator(1,[1.3]*3,dtype="float64",compiled=True)
if "{backend}"=="torch":
    import torch as xp
    q=xp.tensor([[[0.,0.,0.],[.09419,0.,0.],[-.028,.09,0.]]],dtype=xp.float64,device="cuda")
    h=xp.eye(3,dtype=q.dtype,device=q.device)[None]*1.3
else:
    import jax.numpy as xp
    q=xp.asarray([[[0.,0.,0.],[.09419,0.,0.],[-.028,.09,0.]]],dtype=xp.float64)
    h=xp.eye(3,dtype=q.dtype)[None]*1.3
result=f(q,h)
print(result)
assert "{other}" not in sys.modules
'''
    # Keep a fresh compiler subprocess from spawning a large worker pool while
    # the parent validation process already has Torch and JAX pools alive.
    env=dict(os.environ,TORCHINDUCTOR_COMPILE_THREADS='1')
    subprocess.run([sys.executable,'-c',script],check=True,timeout=180,env=env)


import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("water_launcher", PACKAGE / "benchmark.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class BenchmarkLauncher(unittest.TestCase):
    def source_fixture(self, root):
        archive = root / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, content in {"ipi/__init__.py": b"version = 1\n", "bin/i-pi": b"print('ipi')\n", "pyproject.toml": b"[project]\nname='ipi'\n"}.items():
                member = tarfile.TarInfo("source/" + name)
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
        return dict(commit="a" * 40, url=archive.as_uri(), archive_sha256=launcher.sha256(archive))

    def test_verified_cache_rejects_edited_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pin = self.source_fixture(root)
            destination = root / "sources/main"
            first = launcher.fetch_source(destination, pin, root / "downloads")
            self.assertEqual(launcher.fetch_source(destination, pin, root / "downloads"), first)
            (destination / "ipi/__init__.py").write_text("version = 2\n")
            with self.assertRaisesRegex(RuntimeError, "modified"):
                launcher.fetch_source(destination, pin, root / "downloads")

    def test_wrong_archive_checksum_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pin = self.source_fixture(root)
            pin["archive_sha256"] = "0" * 64
            destination = root / "sources/pr"
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                launcher.fetch_source(destination, pin, root / "downloads")
            self.assertFalse(destination.exists())

    def test_expected_matrix_and_commits(self):
        self.assertEqual(len(launcher.METHODS) * len(launcher.PRECISIONS) * len(launcher.REPLICAS), 84)
        pins = launcher.SOURCES
        self.assertEqual(pins["main"]["commit"], "93162d280d15fd954da6dc3580a5ed3503e207ac")
        self.assertEqual(pins["pr"]["commit"], "1d3682e90584b2e072feb95c9aaad510a3367475")
        for pin in pins.values():
            self.assertEqual(len(pin["archive_sha256"]), 64)
