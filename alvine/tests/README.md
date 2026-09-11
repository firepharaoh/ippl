# SFSL3D Time Integration Regressions

The `--integrator euler` path follows the source split in
`SFSL_pipeline_A4_remade.pdf`. Comments in `spectral/SFSLEuler3D.hpp` identify
the diagram's steps. Leapfrog still uses its previous algorithm. RK4 now
uses the grid-only Strang adaptation described below.

Stretching is evaluated on the IFFT grid and transformed back with FFT.
For stored `q = omega_hat / k^2`, the source update is

```
q_new = (q_old + dt * stretching_hat / k^2) / (1 + nu * dt * k^2)
```

The existing zero-mode, Nyquist, filtering, and Biot-Savart projection
conventions are retained. Velocity is rebuilt from the updated vorticity.
IFFT fields directly supply velocity and updated strengths to cell-center
particles before Euler transport. The existing type-1 NUFFT scatters those
advected particles before they are discarded and replaced on the lattice.
No CIC is used. Direct sampling requires `nx = ny = nz` and
`np = nx * ny * nz`.

This is a first-order time split, with backward Euler for diffusion. It
does not promise conservation of energy/enstrophy in inviscid nonlinear
flows or remove transport/filter/projection errors. Diffusion contracts
each nonzero represented mode by `1 / (1 + nu * dt * k^2)`.

All grid kernels use the configured Kokkos execution/memory space. Three
persistent complex source fields are allocated for Euler; the nine existing
gradient buffers are reused as IFFT scratch. FFTs and particle migration
remain distributed. Adaptive LCFL transfers only a reduced scalar maximum
to MPI. The new FFT/source operations change floating-point ordering, so
bitwise agreement with old runs or across MPI layouts is not expected.

## Build and Run

In an existing configured build with the normal MPI/Kokkos/heFFTe toolchain:

```sh
cmake -S . -B build_openmp -DIPPL_ENABLE_TESTS=ON
cmake --build build_openmp --target SpectralFSL3D SFSLEuler3DTest -j 4
ctest --test-dir build_openmp -R '^sfsl_euler_3d_' --output-on-failure
```

The tests run the production code on an 8^3 grid with one and two MPI ranks:

- TGV initialization has the expected physical vorticity even with viscosity
  enabled: initialization must not perform a viscous timestep.
- TGV stretching matches its independently differentiated analytic field;
  the original k^2=3 modes and generated k^2=8 modes receive their respective
  implicit diffusion factors.
- Three full timesteps of a transverse shear preserve the updated strengths
  through transport, scatter, destruction, and IFFT resampling. Its energy
  and enstrophy decay by the square of the expected amplitude factor.
- Zero viscosity, disabled stretching, active LCFL, and final-time clipping
  exercise the operator switches and consistent selection of dt.

The tolerance is 2e-7, allowing the existing type-1 NUFFT tolerance while
detecting missing diffusion, extra diffusion, wrong k^2 recovery, FFT
normalization/phase errors, or discarded source updates.

## PDF Coupled RK4 with Strang Diffusion

Select `--integrator strang` for the nonlinear treatment in
`SFSL_RK4_full_pipeline_A4.pdf`:

```
D(dt/2) -> RK4(advection + stretching, dt) -> D(dt/2)
```

The first exact diffusion half-step multiplies stored `q=omega_hat/k^2`
by `exp(-nu*k^2*dt/2)`. IFFT samples refresh lattice particle strengths.
Each classical RK4 stage then evolves BOTH `R` and `Gamma=omega*particleVolume`
from their saved post-diffusion base states. Each RHS scatters that stage's
particles with Type-1 NUFFT, constructs velocity and nine spectral gradients,
and gathers them at the stage positions using Type-2 NUFFT. The slopes are
`k_R=u` and `k_Gamma=grad(u)*Gamma`; no additional volume factor or dt belongs
in the RHS. The final combination uses weights `(1,2,2,1)/6` for both states.
Only after scattering this combination is the second diffusion half applied.
There are no separate stretching half-steps in this path.

SFSL transfer conventions replace the diagram's generic grid interpolation:
no CIC, direct IFFT sampling only on cell centers, NUFFT at off-grid stages.
The completed spectral state remains authoritative and supplies a new lattice,
rather than retaining the diagram's off-grid particles for the next timestep.
As for Euler, a cubic grid and one particle per grid cell are required.

LCFL/final-time clipping selects one dt before either diffusion half. Stage
RHS calls do not adapt dt, filter, project the stored vorticity, or apply
viscosity. The normal solenoidal projection and configured filter act on the
completed step. Existing zero/Nyquist conventions are unchanged.

Overall formal temporal order is two with diffusion and four for the isolated
coupled nonlinear ODE; transfer, projection, and filtering errors can limit
observed convergence. Exact diffusion is contractive but does not guarantee
nonlinear stability or inviscid energy/enstrophy conservation. The nonlinear
RK evaluations and MPI/NUFFT operation order differ from the `rk4` path;
bitwise agreement between those paths or different rank layouts is not expected.

Kernels use the configured Kokkos execution and memory space. Existing
registered particle bases/slopes are reused and migrate together; there are
no full-state host copies. Three spectral source buffers support LCFL, but
the six grid-stretching RK scratch fields are not allocated for `strang`.
Gathering nine gradients at four stages adds distributed NUFFT work.

## Legacy Grid-Split RK4 Option

Select this path with `--integrator rk4`. It adapts the three blocks in
`SFSL_RK4_full_pipeline_A4.pdf` to preserve SFSL grid-based stretching:

```
D(dt/2) -> S_RK4(dt/2) -> A_RK4(dt) -> S_RK4(dt/2) -> D(dt/2)
```

- `D(h)` multiplies stored vorticity modes `q` by `exp(-nu*k^2*h)`.
  It neither damps velocity independently nor scatters/gathers particles.
- `S_RK4(h)` integrates `dq/dt = FFT[(omega . grad)u] / k^2` on the fixed
  spectral grid. Each of four stages rebuilds Biot-Savart velocity and all
  gradients from that stage's q, computes the product on the IFFT grid,
  and FFTs it back. RK slopes contain no timestep factor.
- `A_RK4(dt)` starts lattice particles with IFFT-sampled, source-updated
  strengths. These strengths remain constant during the advection subflow.
  Each temporary position is formed from the saved initial position. Type-1
  NUFFT scatters the current stage, Biot-Savart rebuilds velocity, and Type-2
  NUFFT evaluates velocity at off-grid stage positions. The particle
  positions, saved initial positions, and RK slopes migrate together.

The additional symmetric `S/2 -> A -> S/2` decomposition is intentional:
the PDF's single coupled particle RK4 would require a particle stretching
RHS. Here stretching stays on the grid and is never gathered to particles.
This is a second-order composition, including when nu=0; it must not be
reported as a fourth-order integrator for the full PDE. Each nonlinear
subflow individually uses classical RK4.

One dt is selected from the initial state using LCFL and the remaining final
time before either diffusion half-step. No stage changes dt. Filtering and
the existing vorticity projection act on the completed composition. The
individual source/advection subflows need not preserve div(omega), so their
Biot-Savart evaluations skip the otherwise default projection. VIF and Euler
retain their default projected solves. Applying projection within these split
subflows would change the operators being composed.

After the second diffusion half-step, recompute velocity and recreate the
cell-center particles by IFFT sampling. Never scatter the old particle
strengths at this point: that would discard the second source/diffusion halves.
Diagnostics run after the physical time increment. Stage tracing refreshes
velocity without filtering/projecting q and does not perform extra scatters.

As for Euler, this path requires a cubic grid with one particle per cell.
No CIC is used. Direct IFFT sampling is restricted to lattice particles;
off-grid RK stage velocities use the existing Type-2 NUFFT.

The spectral RK4 implementation adds six persistent complex fields (base
state and weighted slope sum), in addition to the three source fields shared
with Euler. It reuses these fields between both stretching halves and all
steps. Kernels execute in the configured Kokkos memory/execution space;
there are no full-field host copies. The extra FFTs and different ordering
of filtering, projection, and diffusion change runtime and floating-point
results. Exact diffusion removes its explicit stability bound, but advection,
stretching, aliasing, and transfer errors still limit stability and accuracy.

```sh
cmake --build build_openmp --target SpectralFSL3D SFSLEuler3DTest SFSLStrang3DTest -j 4
ctest --test-dir build_openmp -R '^sfsl_(euler|strang)_3d_' --output-on-failure
```

`SFSLStrang3DTest` exercises `--integrator strang`: analytic TGV particle RHS,
strength-volume scaling, repeated noncumulative and synchronized temporary
stages across migration, exact half-step diffusion, full shear decay and
particle reset, inviscid/disabled-source cases, LCFL and final-time consistency,
fourth-order refinement of legacy `rk4` isolated stretching, and second-order refinement
of the complete noncommuting TGV composition on a 16^3 grid. The MPI tests run
on one and two ranks. The full-composition refinement test deliberately uses
a finer grid than the individual operator tests to reduce the transfer-error
floor.
