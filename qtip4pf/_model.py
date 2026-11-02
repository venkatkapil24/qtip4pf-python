"""Parameters from the pinned OpenMM qtip4pf.xml; nm and kJ/mol.

https://github.com/yunhzou/WaterPIMD/blob/c913385b5805ece5801042942cd19a3df1873e42/qtip4pf.xml
"""
import math
from typing import NamedTuple

PARAMETERS = dict(D=485.72056, a=22.87, r0=0.09419,
                  theta0=1.87448361664, ktheta=367.6,
                  sigma=0.31589, epsilon=0.7749,
                  qh=0.5564, qm=-1.1128, gamma=0.73612)
# OpenMM 8.6.1 SimTKOpenMMRealType.h definitions (2019 SI constants).
COULOMB = 1.0/(4*math.pi*(1e-6*8.8541878128e-12 /
                           (1.602176634e-19**2*6.02214076e23)))
# i-PI UnitMap: angstrom=1.8897261 Bohr; j/mol=0.00000038087989 Ha.
NM_PER_BOHR = 1.0/(10*1.8897261)
HARTREE_PER_KJ_MOL = 1000*0.00000038087989


def rectangular_lengths(cells, cutoff):
    """Validate a box-setting operation; permit only roundoff off the diagonal."""
    lengths = []
    for cell in cells:
        if len(cell) != 3 or any(len(row) != 3 for row in cell):
            raise ValueError("Expected batched 3 by 3 rectangular cells")
        diagonal = [float(cell[i][i]) for i in range(3)]
        if not all(math.isfinite(x) and x > 2*cutoff for x in diagonal):
            raise ValueError("Box lengths must be finite and strictly larger than twice the cutoff")
        tolerance = 1e-12*max(diagonal)
        if any(not math.isfinite(float(cell[i][j])) or abs(float(cell[i][j])) > tolerance
               for i in range(3) for j in range(3) if i != j):
            raise ValueError("Only rectangular cells are supported")
        lengths.append(diagonal)
    return lengths


class EwaldPlan(NamedTuple):
    nwater: int
    cutoff: float
    tolerance: float
    alpha: float
    kmax: tuple
    indices: object
    charge_pairs: object
    oxygen_pairs: object
    charges: object
    params: dict
    self_energy: float
    exclusion_coefficients: object
    dispersion_coefficient: float
    pair_chunk: int
    reciprocal_chunk: int
    parameter_mode: str


def _openmm_kmax(width, alpha, tolerance):
    def error(n):
        return tolerance-0.05*math.sqrt(width*alpha)*n*math.exp(-(n*math.pi/(width*alpha))**2)
    n, value = 10, error(10)
    if value > 0:
        while value > 0 and n > 0:
            n -= 1
            value = error(n)
        n += 1
    else:
        while value < 0:
            n += 1
            value = error(n)
    return n+1 if n % 2 == 0 else n


def make_plan(nwater, reference_box_nm, cutoff_nm=0.9, ewald_tolerance=5e-4,
              dispersion_correction=True, tensor=None, parameter_mode="reference",
              pair_chunk=32768, reciprocal_chunk=256):
    """Initialize exactly the Ewald alpha and integer limits chosen by OpenMM.

    These parameters stay FIXED when the live cell changes, just as in an
    OpenMM Context. reference_box_nm specifies three positive box lengths.
    XML parameters retain their full precision, as in OpenMM Reference. This
    deliberately does not reproduce CUDA Ewald's inconsistent energy sum.
    """
    if not isinstance(nwater, int) or nwater < 1:
        raise ValueError("nwater must be a positive integer")
    box = tuple(float(x) for x in reference_box_nm)
    if len(box) != 3 or not all(math.isfinite(x) and x > 0 for x in box):
        raise ValueError("reference_box_nm must contain three positive finite lengths")
    if not math.isfinite(cutoff_nm) or cutoff_nm <= 0 or 2*cutoff_nm >= min(box):
        raise ValueError("cutoff_nm must be positive and strictly below half the shortest box length")
    if not 0 < ewald_tolerance < 0.5:
        raise ValueError("ewald_tolerance must be between 0 and 0.5")
    if parameter_mode != "reference":
        raise ValueError("Only the consistent OpenMM Reference equations are supported")
    if not isinstance(pair_chunk, int) or not isinstance(reciprocal_chunk, int) or min(pair_chunk, reciprocal_chunk) < 1:
        raise ValueError("chunk sizes must be positive integers")
    p = PARAMETERS.copy()
    p["a2"] = p["a"]**2
    alpha = math.sqrt(-math.log(2*ewald_tolerance))/cutoff_nm
    kmax = tuple(_openmm_kmax(x, alpha, ewald_tolerance) for x in box)
    integers = [(x, y, z) for x in range(kmax[0])
                for y in range(1-kmax[1], kmax[1])
                for z in range(1-kmax[2], kmax[2])
                if x > 0 or (x == 0 and y > 0) or (x == 0 and y == 0 and z > 0)]
    indices = tensor(integers)
    charge_pairs = oxygen_pairs = None
    charges = tensor([p["qh"], p["qh"], p["qm"]]*nwater)
    q2 = 2*p["qh"]**2+p["qm"]**2
    self_energy = -COULOMB*alpha/math.sqrt(math.pi)*q2*nwater
    excluded = [COULOMB*p["qh"]*p["qh"], COULOMB*p["qh"]*p["qm"],
                COULOMB*p["qh"]*p["qm"]]
    excluded = tensor(excluded)
    # OpenMM's finite-system tail coefficient counts all FOUR particles,
    # including virtual sites, and uses the ORIGINAL XML LJ parameters.
    particles = 4*nwater
    count = nwater*(nwater+1)/2
    interactions = particles*(particles+1)/2
    sigma, epsilon = PARAMETERS["sigma"], PARAMETERS["epsilon"]
    tail = 8*particles*particles*math.pi*count/interactions*epsilon*(
        sigma**12/(9*cutoff_nm**9)-sigma**6/(3*cutoff_nm**3))
    return EwaldPlan(nwater, cutoff_nm, ewald_tolerance, alpha, kmax,
                     indices, charge_pairs, oxygen_pairs, charges, p,
                     self_energy, excluded, tail if dispersion_correction else 0.0,
                     pair_chunk, reciprocal_chunk, parameter_mode)



import math

def _outer_sum(force, displacement, dimensions, xp):
    # Explicit reductions avoid turning this into many small batched GEMMs.
    return xp.stack([
        xp.sum((force[..., a] * displacement[..., b]), axis=dimensions)
        for a in range(3) for b in range(3)
    ], axis=-1).reshape(force.shape[0], 3, 3)


def _pair_terms(points, cell, inverse_lengths, matrix, plan, electrostatic, xp):
    """Half-list pair energies, forces and -dU/dstrain; no pair truncation."""
    batch, atoms, _ = points.shape
    flat = points.reshape(-1, 3)
    total = flat.shape[0]
    target = xp.index(matrix, total-1)
    source = xp.arange(total, like=points)[:, None]
    valid = matrix < total
    if electrostatic:
        valid = valid & (source // 3 != target // 3)
    delta = (flat[:, None, :] - flat[target]).reshape(batch, atoms, -1, 3)
    lengths = xp.diagonal(cell, axis1=-2, axis2=-1)[:, None, None, :]
    delta = delta - xp.floor(delta * inverse_lengths[:, None, None, :] + 0.5) * lengths
    r2 = xp.sum((delta * delta), axis=-1)
    valid = valid.reshape(batch, atoms, -1) & (r2 < plan.cutoff**2)
    r2 = xp.where(valid, r2, 1.0)
    r = xp.sqrt(r2)
    if electrostatic:
        charge = xp.broadcast_to(plan.charges, (batch, atoms)).reshape(-1)
        qq = (charge[:, None] * charge[target]).reshape(r.shape)
        ar = plan.alpha * r
        erfc = xp.erfc(ar)
        value = COULOMB * qq * erfc / r
        coefficient = COULOMB * qq * (
            erfc / r + 2 * plan.alpha / math.sqrt(math.pi) * xp.exp(-ar*ar)
        ) / r2
    else:
        sr6 = (plan.params["sigma"]**2 / r2)**3
        value = 4 * plan.params["epsilon"] * sr6 * (sr6-1)
        coefficient = 24 * plan.params["epsilon"] * sr6 * (2*sr6-1) / r2
    pair_force = xp.where(valid, coefficient, 0.0)[..., None] * delta
    energy = xp.sum(xp.where(valid, value, 0.0), axis=(1, 2))
    central = xp.sum(pair_force, axis=2).reshape(total, 3)
    # Invalid slots contribute exactly zero. Route those additions back to
    # their source rows instead of contending on the clamped final atom.
    reaction_target = xp.where(valid.reshape(matrix.shape), target,
                               xp.broadcast_to(source, target.shape))
    reaction = xp.scatter_add(xp.zeros_like(flat), reaction_target.reshape(-1), -pair_force.reshape(-1, 3))
    force = (central + reaction).reshape(points.shape)
    virial = _outer_sum(pair_force, delta, (1, 2), xp)
    return energy, force, virial


def analytic(q, h, plan, charge_matrix, oxygen_matrix, xp, inverse_lengths, nonbonded=None):
    """Explicit derivatives of the unchanged q-TIP4P/F energy expression."""
    batch = q.shape[0]
    p = plan.params
    water = q.reshape(batch, plan.nwater, 3, 3)
    oxygen, h1, h2 = xp.unbind(water, 2)
    v1, v2 = h1-oxygen, h2-oxygen
    r1 = xp.sqrt(xp.sum((v1*v1), axis=-1))
    r2 = xp.sqrt(xp.sum((v2*v2), axis=-1))
    u1, u2 = v1/r1[..., None], v2/r2[..., None]
    dr = xp.stack((r1, r2), axis=-1) - p["r0"]
    stretch = p["D"]*xp.sum((p["a2"]*dr**2-p["a"]**3*dr**3+
                        (7/12)*p["a"]**4*dr**4), axis=(1, 2))
    radial = p["D"]*(2*p["a2"]*dr-3*p["a"]**3*dr**2+
                      (7/3)*p["a"]**4*dr**3)
    fh1 = -radial[..., 0, None]*u1
    fh2 = -radial[..., 1, None]*u2
    sine = xp.sqrt(xp.sum((xp.cross(u1, u2)**2), axis=-1))
    cosine = xp.sum((u1*u2), axis=-1)
    theta = xp.atan2(sine, cosine)
    bend = 0.5*p["ktheta"]*xp.sum(((theta-p["theta0"])**2), axis=1)
    angle = p["ktheta"]*(theta-p["theta0"])/sine
    fh1 = fh1 + (angle/r1)[..., None]*(u2-cosine[..., None]*u1)
    fh2 = fh2 + (angle/r2)[..., None]*(u1-cosine[..., None]*u2)
    force = xp.stack((-fh1-fh2, fh1, fh2), axis=2)
    virial = _outer_sum(fh1, v1, (1,), xp) + _outer_sum(fh2, v2, (1,), xp)
    m = p["gamma"]*oxygen + ((1-p["gamma"])/2)*(h1+h2)
    charged = xp.stack((h1, h2, m), axis=2).reshape(batch, -1, 3)
    if nonbonded is None:
        direct, charge_force, pair_virial = _pair_terms(charged, h, inverse_lengths, charge_matrix, plan, True, xp)
        lj, oxygen_force, lj_virial = _pair_terms(oxygen, h, inverse_lengths, oxygen_matrix, plan, False, xp)
        virial = virial + pair_virial + lj_virial

        # E_k = 4*pi*C/V exp(-k^2/(4 alpha^2))/k^2 |sum q exp(i k.r)|^2.
        # At fixed fractional coordinates the phases are strain-invariant:
        # W_k = E_k [I - 2(1/k^2 + 1/(4 alpha^2)) k outer k].
        lengths = xp.diagonal(h, axis1=-2, axis2=-1)
        volume = xp.prod(lengths, axis=-1)
        k = 2*math.pi*plan.indices[None, :, :]*inverse_lengths[:, None, :]
        k2 = xp.sum((k*k), axis=-1)
        phase = sum(charged[..., a, None]*k[:, None, :, a] for a in range(3))
        cos, sin = xp.cos(phase), xp.sin(phase)
        real = xp.sum((cos*plan.charges[None, :, None]), axis=1)
        imag = xp.sum((sin*plan.charges[None, :, None]), axis=1)
        coefficient = (4*math.pi*COULOMB/volume[:, None]) * xp.exp(
            -k2/(4*plan.alpha**2)
        )/k2
        ek = coefficient*(real*real+imag*imag)
        reciprocal = xp.sum(ek, axis=1)
        phase_force = 2*plan.charges[None, :, None]*coefficient[:, None, :]*(
            real[:, None, :]*sin-imag[:, None, :]*cos
        )
        reciprocal_force = xp.stack([
            xp.sum((phase_force*k[:, None, :, a]), axis=-1) for a in range(3)
        ], axis=-1)
        stress_weight = 2*ek*(1/k2+1/(4*plan.alpha**2))
        reciprocal_virial = xp.stack([
            xp.sum(((ek if a == b else 0.0)-stress_weight*k[..., a]*k[..., b]), axis=-1)
            for a in range(3) for b in range(3)
        ], axis=-1).reshape(batch, 3, 3)
        charge_force = charge_force + reciprocal_force
        virial = virial + reciprocal_virial

    else:
        direct, lj, reciprocal, charge_force, oxygen_force, nonbonded_virial = nonbonded(
            charged, oxygen, h, inverse_lengths)
        virial = virial + nonbonded_virial
        volume = xp.prod(xp.diagonal(h,axis1=-2,axis2=-1),axis=-1)

    # Remove reciprocal interactions for the three intramolecular charge pairs.
    displacement = xp.stack((h1-h2, h1-m, h2-m), axis=2)
    distance2 = xp.sum((displacement*displacement), axis=-1)
    distance = xp.sqrt(distance2)
    ar = plan.alpha*distance
    qq = plan.exclusion_coefficients[None, None, :]
    exclusion = -xp.sum((qq*xp.erf(ar)/distance), axis=(1, 2))
    exclusion_force = (qq*(2*plan.alpha/math.sqrt(math.pi)*xp.exp(-ar*ar)
                          -xp.erf(ar)/distance)/distance2)[..., None]*displacement
    x12, x1m, x2m = xp.unbind(exclusion_force, 2)
    excluded = xp.stack((x12+x1m, -x12+x2m, -x1m-x2m), axis=2)
    charge_force = charge_force.reshape(batch, plan.nwater, 3, 3)+excluded
    virial = virial + _outer_sum(exclusion_force, displacement, (1, 2), xp)
    fc1, fc2, fcm = xp.unbind(charge_force, 2)
    force = force + xp.stack((p["gamma"]*fcm+oxygen_force,
        fc1+((1-p["gamma"])/2)*fcm, fc2+((1-p["gamma"])/2)*fcm), axis=2)
    tail = plan.dispersion_coefficient/volume
    virial = virial + xp.eye(3, like=q)*tail[:, None, None]
    energy = stretch+bend+direct+lj+reciprocal+exclusion+plan.self_energy+tail
    return energy, force.reshape(q.shape), virial
