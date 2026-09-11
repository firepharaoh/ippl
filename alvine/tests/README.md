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

## Strang Splitting with RK4 Substeps

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

`SFSLStrang3DTest` checks exact half-step diffusion, full shear decay and
particle reset, inviscid/disabled-source cases, LCFL and final-time consistency,
fourth-order refinement of isolated stretching, and second-order refinement
of the complete noncommuting TGV composition on a 16^3 grid. The MPI tests run
on one and two ranks. The full-composition refinement test deliberately uses
a finer grid than the individual operator tests to reduce the transfer-error
floor.
