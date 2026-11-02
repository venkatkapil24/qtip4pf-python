"""Shared CUDA Triton kernels. No Torch/JAX imports and no force atomics.

All nine strain-virial components are returned. Reciprocal kernels consume the
validated half-space wavevector list; they never reconstruct a different grid.
"""
import math
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def reciprocal_geometry(plan,h,inverse,xp):
    """Geometry only: can be cached outside MD when the box is set."""
    from ._model import COULOMB
    lengths = xp.diagonal(h,axis1=-2,axis2=-1)
    wave = 2*math.pi*plan.indices[None,:,:]*inverse[:,None,:]
    k2 = xp.sum(wave*wave,axis=-1)
    volume = xp.prod(lengths,axis=-1)
    coefficient = (4*math.pi*COULOMB/volume[:,None])*xp.exp(-k2/(4*plan.alpha**2))/k2
    return lengths,wave,coefficient


@triton.jit
def pairs(X, Charge, Length, Inverse, Force, EW, N: tl.constexpr,
          C: tl.constexpr, ALPHA: tl.constexpr, SIGMA: tl.constexpr,
          EPSILON: tl.constexpr, CUTOFF2: tl.constexpr,
          ELECTROSTATIC: tl.constexpr, BLOCK: tl.constexpr):
    i, b = tl.program_id(0), tl.program_id(1)
    xi = tl.load(X+(b*N+i)*3)
    yi = tl.load(X+(b*N+i)*3+1)
    zi = tl.load(X+(b*N+i)*3+2)
    lx, ly, lz = tl.load(Length+b*3), tl.load(Length+b*3+1), tl.load(Length+b*3+2)
    ix, iy, iz = tl.load(Inverse+b*3), tl.load(Inverse+b*3+1), tl.load(Inverse+b*3+2)
    dtype: tl.constexpr = X.dtype.element_ty
    fx, fy, fz, energy = tl.full((),0,dtype), tl.full((),0,dtype), tl.full((),0,dtype), tl.full((),0,dtype)
    xx, xy, xz = tl.full((),0,dtype), tl.full((),0,dtype), tl.full((),0,dtype)
    yy, yz, zz = tl.full((),0,dtype), tl.full((),0,dtype), tl.full((),0,dtype)
    for start in range(0,N,BLOCK):
        j = start+tl.arange(0,BLOCK)
        dx = xi-tl.load(X+(b*N+j)*3,mask=j<N,other=0)
        dy = yi-tl.load(X+(b*N+j)*3+1,mask=j<N,other=0)
        dz = zi-tl.load(X+(b*N+j)*3+2,mask=j<N,other=0)
        dx -= tl.floor(dx*ix+0.5)*lx
        dy -= tl.floor(dy*iy+0.5)*ly
        dz -= tl.floor(dz*iz+0.5)*lz
        r2 = dx*dx+dy*dy+dz*dz
        valid = (j<N)&(r2<CUTOFF2)&(j!=i)
        if ELECTROSTATIC:
            valid &= (j//3 != i//3)
        r2 = tl.where(valid,r2,1.0)
        if ELECTROSTATIC:
            r = tl.extra.cuda.libdevice.sqrt(r2)
            qq = tl.load(Charge+i)*tl.load(Charge+j,mask=j<N,other=0)
            ar = ALPHA*r
            complementary = tl.extra.cuda.libdevice.erfc(ar)
            value = C*qq*complementary/r
            coefficient = C*qq*(complementary/r+2*ALPHA/1.772453850905516*tl.extra.cuda.libdevice.exp(-ar*ar))/r2
        else:
            sr6 = (SIGMA*SIGMA/r2)*(SIGMA*SIGMA/r2)*(SIGMA*SIGMA/r2)
            value = 4*EPSILON*sr6*(sr6-1)
            coefficient = 24*EPSILON*sr6*(2*sr6-1)/r2
        coefficient = tl.where(valid,coefficient,0.0)
        vx, vy, vz = coefficient*dx, coefficient*dy, coefficient*dz
        fx += tl.sum(vx,0)
        fy += tl.sum(vy,0)
        fz += tl.sum(vz,0)
        energy += 0.5*tl.sum(tl.where(valid,value,0.0),0)
        xx += 0.5*tl.sum(vx*dx,0)
        xy += 0.5*tl.sum(vx*dy,0)
        xz += 0.5*tl.sum(vx*dz,0)
        yy += 0.5*tl.sum(vy*dy,0)
        yz += 0.5*tl.sum(vy*dz,0)
        zz += 0.5*tl.sum(vz*dz,0)
    tl.store(Force+(b*N+i)*3,fx)
    tl.store(Force+(b*N+i)*3+1,fy)
    tl.store(Force+(b*N+i)*3+2,fz)
    base = (b*N+i)*10
    tl.store(EW+base,energy)
    tl.store(EW+base+1,xx)
    tl.store(EW+base+2,xy)
    tl.store(EW+base+3,xz)
    tl.store(EW+base+4,xy)
    tl.store(EW+base+5,yy)
    tl.store(EW+base+6,yz)
    tl.store(EW+base+7,xz)
    tl.store(EW+base+8,yz)
    tl.store(EW+base+9,zz)


@triton.jit
def structure_factors(X, Charge, Wave, Coefficient, S, EW,
                      N: tl.constexpr, K: tl.constexpr, ALPHA: tl.constexpr,
                      BLOCK: tl.constexpr):
    k, b = tl.program_id(0), tl.program_id(1)
    kx = tl.load(Wave+(b*K+k)*3)
    ky = tl.load(Wave+(b*K+k)*3+1)
    kz = tl.load(Wave+(b*K+k)*3+2)
    dtype: tl.constexpr = X.dtype.element_ty
    real, imag = tl.full((),0,dtype), tl.full((),0,dtype)
    for start in range(0,N,BLOCK):
        j = start+tl.arange(0,BLOCK)
        x = tl.load(X+(b*N+j)*3,mask=j<N,other=0)
        y = tl.load(X+(b*N+j)*3+1,mask=j<N,other=0)
        z = tl.load(X+(b*N+j)*3+2,mask=j<N,other=0)
        q = tl.load(Charge+j,mask=j<N,other=0)
        phase = x*kx+y*ky+z*kz
        real += tl.sum(q*tl.extra.cuda.libdevice.cos(phase),0)
        imag += tl.sum(q*tl.extra.cuda.libdevice.sin(phase),0)
    tl.store(S+(b*K+k)*2,real)
    tl.store(S+(b*K+k)*2+1,imag)
    energy = tl.load(Coefficient+b*K+k)*(real*real+imag*imag)
    k2 = kx*kx+ky*ky+kz*kz
    weight = 2*energy*(1/k2+1/(4*ALPHA*ALPHA))
    base = (b*K+k)*10
    tl.store(EW+base,energy)
    tl.store(EW+base+1,energy-weight*kx*kx)
    tl.store(EW+base+2,-weight*kx*ky)
    tl.store(EW+base+3,-weight*kx*kz)
    tl.store(EW+base+4,-weight*kx*ky)
    tl.store(EW+base+5,energy-weight*ky*ky)
    tl.store(EW+base+6,-weight*ky*kz)
    tl.store(EW+base+7,-weight*kx*kz)
    tl.store(EW+base+8,-weight*ky*kz)
    tl.store(EW+base+9,energy-weight*kz*kz)


@triton.jit
def reciprocal_force(X, Charge, Wave, Coefficient, S, Force,
                     N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    i, b = tl.program_id(0), tl.program_id(1)
    x, y, z = tl.load(X+(b*N+i)*3), tl.load(X+(b*N+i)*3+1), tl.load(X+(b*N+i)*3+2)
    qi = tl.load(Charge+i)
    dtype: tl.constexpr = X.dtype.element_ty
    fx, fy, fz = tl.full((),0,dtype), tl.full((),0,dtype), tl.full((),0,dtype)
    for start in range(0,K,BLOCK):
        k = start+tl.arange(0,BLOCK)
        kx = tl.load(Wave+(b*K+k)*3,mask=k<K,other=0)
        ky = tl.load(Wave+(b*K+k)*3+1,mask=k<K,other=0)
        kz = tl.load(Wave+(b*K+k)*3+2,mask=k<K,other=0)
        real = tl.load(S+(b*K+k)*2,mask=k<K,other=0)
        imag = tl.load(S+(b*K+k)*2+1,mask=k<K,other=0)
        a = tl.load(Coefficient+b*K+k,mask=k<K,other=0)
        phase = x*kx+y*ky+z*kz
        weight = 2*qi*a*(real*tl.extra.cuda.libdevice.sin(phase)-imag*tl.extra.cuda.libdevice.cos(phase))
        fx += tl.sum(weight*kx,0)
        fy += tl.sum(weight*ky,0)
        fz += tl.sum(weight*kz,0)
    tl.store(Force+(b*N+i)*3,fx)
    tl.store(Force+(b*N+i)*3+1,fy)
    tl.store(Force+(b*N+i)*3+2,fz)


@triton.jit
def reduce_terms(CE, OE, KE, Out, NC: tl.constexpr, NO: tl.constexpr,
                 K: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    b, component = tl.program_id(0), tl.program_id(1)
    column = tl.where(component<3,0,component-2)
    dtype: tl.constexpr = CE.dtype.element_ty
    value = tl.full((),0,dtype)
    for start in range(0,MAXIMUM,BLOCK):
        j = start+tl.arange(0,BLOCK)
        c = tl.load(CE+(b*NC+j)*10+column,mask=(j<NC)&((component==0)|(component>=3)),other=0)
        o = tl.load(OE+(b*NO+j)*10+column,mask=(j<NO)&((component==1)|(component>=3)),other=0)
        k = tl.load(KE+(b*K+j)*10+column,mask=(j<K)&((component==2)|(component>=3)),other=0)
        value += tl.sum(c+o+k,0)
    tl.store(Out+b*12+component,value)
