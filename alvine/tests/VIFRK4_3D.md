# VIF3D classical RK4

Select --integrator rk4 on VortexInFourier3D. Euler and RK4 call
evaluateParticleTimeRHS3D for the identical coupled particle equations:

    R' = u(R)
    Gamma' = grad(u)(R)*Gamma + particleVolume*nu*Laplacian(omega)(R)

Gamma is particle strength, not physical vorticity. Each stage scatters its
own positions and strengths, computes spectral velocity, gradients and
diffusion, and gathers them at those positions. Intermediate states are
constructed from the saved initial state with scales 1/2, 1/2, 1.
The final combination uses weights (1,2,2,1)/6 for both positions and strengths.
One timestep is selected in the first stage and retained throughout the step.
Remeshing follows the final combination and the configured schedule.
RK4 applies the same configured RHS filters in all four stages, including
the first stage after remeshing. The remesh filter-skip convention used by
other integrators is not applied to RK4.

This is fully explicit fourth-order integration of the spatial particle ODE.
Viscosity remains subject to explicit diffusion stability limits; LCFL alone
does not enforce them. Filtering, NUFFT errors and remeshing can limit observed
temporal convergence. RK4 does not guarantee inviscid energy or enstrophy
conservation.

Existing registered particle buffers migrate with positions at every stage.
The configured Kokkos execution and memory spaces are unchanged; no new
production host copies or allocations are introduced. Independent base states
and velocity slopes are retained through explicit Kokkos deep copies within
the particle memory space. Four spectral RHS
evaluations and intermediate MPI migrations dominate cost relative to Euler.
The shared helper preserves arithmetic ordering within each RHS; different
integrators, MPI layouts and execution backends need not agree bitwise.

Build and test in the existing configured OpenMP build:

    cmake --build build --target VortexInFourier3D VIFEuler3DTest VIFRK4_3DTest -j 4
    OMP_NUM_THREADS=2 OMP_PROC_BIND=false ctest --test-dir build -R '^vif_(euler|rk4)_3d_' --output-on-failure

One- and two-rank tests check analytic shear strength and position polynomials,
two particle densities, synchronized scalar/vector strengths, repeated steps,
remeshing, clipped adaptive timesteps, fourth-order shear decay, and nonlinear
TGV refinement against a finer temporal reference. Nonlinear refinement uses
a global strength norm and permits spatial/NUFFT error floors.
The shear order test uses an eight-cell grid and 10/20/40 steps to keep all
resolved diffusion modes stable. Nonlinear refinement uses nu=0.02
for the same reason. Storage independence and consistent post-remesh filtering
are checked explicitly.
