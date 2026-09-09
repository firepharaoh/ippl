# SFSL3D Euler Pipeline Regression

The `--integrator euler` path follows the source split in
`SFSL_pipeline_A4_remade.pdf`. Comments in `spectral/SFSLEuler3D.hpp` identify
the diagram's steps. Leapfrog and RK4 still use their previous algorithms.

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
