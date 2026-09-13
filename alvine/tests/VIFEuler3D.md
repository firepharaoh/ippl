# VIF3D Forward Euler

Select `--integrator euler` on `VortexInFourier3D`. This follows the particle
update in `ChatGPT Image Sep 13, 2026, 12_26_30 AM.png`, despite that image's
SFSL title. It does not change SFSL's separate IMEX Euler implementation.

The numbered comments in `EulerStep()` and `computeRK4ParticleRHS()` match
the eight image steps:

1. Preserve the old particle positions and strengths, Gamma=omega*dVp.
2. Type-1 scatter gives q=omega_hat/k^2 with the existing normalization.
3. Biot-Savart gives u_hat=i*k cross q.
4. Compute grad(u)_hat=i*k_j*u_i_hat and D_hat=-nu*k^4*q.
5. Type-2 gather evaluates velocity, gradients, and D at the old positions.
6. Form k_Gamma=grad(u)*Gamma+dVp*D without modifying Gamma.
7. Update Gamma_new=Gamma_old+dt*k_Gamma.
8. Update R_new=R_old+dt*u_old, wrap periodic positions, and migrate particles.

Steps 7 and 8 share a kernel and exactly one dt. No second RHS evaluation,
implicit denominator, exponential diffusion, or source-updated velocity is
used. Optional remeshing follows the existing VIF schedule, not every step
unless requested. Existing filters, projection, and zero/Nyquist handling
are unchanged. Shared helpers retain their historical RK4 names, including
their internal checkpoint labels, but Euler calls them only once per step.

This is first-order forward Euler for the particle ODE. There is no guarantee
of inviscid energy/enstrophy conservation. For the isolated diffusion operator,
stability requires nu*dt*k_max_squared <= 2 (a necessary constraint, not a
sufficient nonlinear stability test). Adaptive LCFL and final-time clipping
are supported, but LCFL does not automatically impose this diffusion bound.

The update reuses existing registered base/RHS particle attributes, which
migrate with the particles. Kokkos kernels use the configured execution and
memory space, with no new device allocations or full-state host copies.
It uses one scatter/operator/gather pass instead of four RK4 stages. Operation
ordering and timestep approximation differ from leapfrog/RK4, so numerical
results are not expected to be bitwise equal across methods or MPI layouts.

## IFFT Remeshing

VIF3D remeshing (all integrators) reconstructs physical vorticity and velocity
using IFFT, then samples their cell centers directly. Gamma=omega*dVp and all
scalar/vector particle attributes are synchronized. The final Type-1 scatter
is retained without another filter application. RHS gathers at off-grid
particles are unchanged; only the six remesh Type-2 gathers are removed.

When remeshing is enabled, nx=ny=nz and np=nx*ny*nz are required. A different
particle lattice is rejected before initialization rather than approximated
with nearest-cell interpolation. With remeshing disabled, different particle
counts remain supported. Supporting different remesh lattices through IFFT
would require a separate spectral resampling grid.

The same reconstruction kernels handle phase and q-to-physical-vorticity
conversion. Device grid views are read directly after particle migration;
no host copies or new persistent device buffers are introduced. Replacing
NUFFT evaluation with IFFT changes floating-point ordering and removes
Type-2 approximation error at these lattice points, but does not guarantee
conservation or stability of the full evolution. Existing filters, projection,
particle lifecycle, and zero/Nyquist conventions are retained.

## Regression

In the configured MPI/Kokkos/heFFTe build:

```sh
cmake --build build_openmp --target VortexInFourier3D VIFEuler3DTest -j 4
ctest --test-dir build_openmp -R '^vif_euler_3d_' --output-on-failure
```

Tests run on one and two ranks. Analytic TGV checks both particle updates
from the same old state, with nu=0 and nu=0.2, using 8^3 and 16^3 particles
on an 8^3 grid. Scalar and vector strengths must remain synchronized after
migration. A transverse shear tests the explicit amplitude (1-nu*dt)^steps,
with and without remeshing. A clipped adaptive step checks consistent dt use.
A t=0 IFFT remesh verifies analytic TGV velocity/vorticity and synchronized
attributes, and a mismatched lattice must be rejected.
