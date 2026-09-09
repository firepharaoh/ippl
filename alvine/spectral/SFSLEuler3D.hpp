#ifndef IPPL_ALVINE_SFSL_EULER_3D_HPP
#define IPPL_ALVINE_SFSL_EULER_3D_HPP

// Member definitions included in SpectralFSL3DManager's public section. Kernel
// launchers must remain public for CUDA extended-lambda compilation.
    bool useSpectralEuler3D() const {
        return this->time_integrator_m == "euler";
    }

    void forwardEulerGridModes3D(ComplexField_t& modes) {
        // Step 8: the forward FFT already includes 1/Ngrid. Remove the
        // cell-center phase to recover the Fourier-series convention of q.
        this->spectralFft_mp->transform(ippl::FORWARD, modes);
        auto values = modes.getView();
        const auto local = modes.getLayout().getLocalNDIndex();
        const auto nr = this->nr_m;
        const int ng = modes.getNghost();
        const T pi = T(std::acos(-1.0));
        Kokkos::parallel_for(
            "sfsl_euler_remove_cell_center_phase",
            ippl::getRangePolicy(values, ng),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int indices[3] = {i - ng + local[0].first(),
                                        j - ng + local[1].first(),
                                        k - ng + local[2].first()};
                T phase = 0;
                for (unsigned d = 0; d < Dim; ++d) {
                    const int m = indices[d] <= nr[d] / 2
                                      ? indices[d] : indices[d] - nr[d];
                    phase += pi * T(m) / T(nr[d]);
                }
                values(i, j, k) *= Kokkos::complex<T>(Kokkos::cos(phase),
                                                      -Kokkos::sin(phase));
            });
        Kokkos::fence();
    }

    void computeEulerGridSource3D() {
        euler_sx_m = Kokkos::complex<T>(0, 0);
        euler_sy_m = Kokkos::complex<T>(0, 0);
        euler_sz_m = Kokkos::complex<T>(0, 0);
        if (!use_stretching_m && !this->adaptive_lcfl_m) {
            return;
        }

        // Steps 5-6: differentiate velocity spectrally, then IFFT all nine
        // derivatives and physical vorticity onto the same cell-center grid.
        // These derivative buffers are scratch until the next gradient build.
        this->computeSpectralVelocityGradientModes3D();
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        for (auto* modes : {&this->duxdx_hat_m, &this->duxdy_hat_m, &this->duxdz_hat_m,
                            &this->duydx_hat_m, &this->duydy_hat_m, &this->duydz_hat_m,
                            &this->duzdx_hat_m, &this->duzdy_hat_m, &this->duzdz_hat_m}) {
            this->applyCellCenteredIfftPhase3D(*modes);
            this->spectralFft_mp->transform(ippl::BACKWARD, *modes);
        }

        auto omega = this->fcontainer_m->getOmegaField().getView();
        auto g00 = this->duxdx_hat_m.getView();
        auto g01 = this->duxdy_hat_m.getView();
        auto g02 = this->duxdz_hat_m.getView();
        auto g10 = this->duydx_hat_m.getView();
        auto g11 = this->duydy_hat_m.getView();
        auto g12 = this->duydz_hat_m.getView();
        auto g20 = this->duzdx_hat_m.getView();
        auto g21 = this->duzdy_hat_m.getView();
        auto g22 = this->duzdz_hat_m.getView();
        auto sx = euler_sx_m.getView();
        auto sy = euler_sy_m.getView();
        auto sz = euler_sz_m.getView();
        const int ng = euler_sx_m.getNghost();
        const bool stretching = use_stretching_m;

        // Step 7: the nonlinear product is evaluated on the grid. S is a
        // physical-vorticity RHS, without dt or particle-volume factors.
        Kokkos::parallel_for(
            "sfsl_euler_grid_stretching", ippl::getRangePolicy(sx, ng),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                if (stretching) {
                    const auto w = omega(i, j, k);
                    sx(i, j, k) = w[0] * g00(i, j, k).real()
                                  + w[1] * g01(i, j, k).real()
                                  + w[2] * g02(i, j, k).real();
                    sy(i, j, k) = w[0] * g10(i, j, k).real()
                                  + w[1] * g11(i, j, k).real()
                                  + w[2] * g12(i, j, k).real();
                    sz(i, j, k) = w[0] * g20(i, j, k).real()
                                  + w[1] * g21(i, j, k).real()
                                  + w[2] * g22(i, j, k).real();
                }
            });
        Kokkos::fence();

        // Select dt before either source operator. The global grid maximum
        // supplies the deformation constraint without a particle gather.
        if (this->adaptive_lcfl_m) {
            T localMax = 0;
            Kokkos::parallel_reduce(
                "sfsl_euler_grid_deformation", ippl::getRangePolicy(sx, ng),
                KOKKOS_LAMBDA(const int i, const int j, const int k, T& maximum) {
                    const T s01 = Kokkos::abs(T(0.5) * (g01(i,j,k).real() + g10(i,j,k).real()));
                    const T s02 = Kokkos::abs(T(0.5) * (g02(i,j,k).real() + g20(i,j,k).real()));
                    const T s12 = Kokkos::abs(T(0.5) * (g12(i,j,k).real() + g21(i,j,k).real()));
                    const T c0 = Kokkos::abs(g00(i,j,k).real()) + s01 + s02;
                    const T c1 = Kokkos::abs(g11(i,j,k).real()) + s01 + s12;
                    const T c2 = Kokkos::abs(g22(i,j,k).real()) + s02 + s12;
                    maximum = Kokkos::max(maximum, Kokkos::max(c0, Kokkos::max(c1, c2)));
                }, Kokkos::Max<T>(localMax));
            const double localValue = double(localMax);
            double globalMax = 0;
            MPI_Allreduce(&localValue, &globalMax, 1, MPI_DOUBLE, MPI_MAX,
                          ippl::Comm->getCommunicator());
            this->dt_m = globalMax > 0
                ? std::min(activeMaximumTimestep3D(), this->lcfl_m / globalMax)
                : activeMaximumTimestep3D();
        }

        // Step 8: transform S back to Fourier space; it is never gathered.
        if (use_stretching_m) {
            forwardEulerGridModes3D(euler_sx_m);
            forwardEulerGridModes3D(euler_sy_m);
            forwardEulerGridModes3D(euler_sz_m);
        }
    }

    void updateEulerSpectralVorticity3D() {
        auto qx = this->omega_x_hat_m.getView();
        auto qy = this->omega_y_hat_m.getView();
        auto qz = this->omega_z_hat_m.getView();
        auto sx = euler_sx_m.getView();
        auto sy = euler_sy_m.getView();
        auto sz = euler_sz_m.getView();
        const auto local = this->omega_x_hat_m.getLayout().getLocalNDIndex();
        const auto nr = this->nr_m;
        const Vector_t<T, Dim> lengths = this->rmax_m - this->rmin_m;
        const int ng = this->omega_x_hat_m.getNghost();
        const T twoPi = T(2) * T(std::acos(-1.0));
        const T dt = T(this->dt_m);
        const T nu = T(this->viscosity_m);

        // Steps 9-10: q = omega_hat/k^2, so the stretching increment is
        // dt*S_hat/k^2. Backward-Euler diffusion acts on that UPDATED state.
        // Keep the shared scatter/Biot-Savart zero/Nyquist convention.
        Kokkos::parallel_for(
            "sfsl_euler_spectral_source_split", ippl::getRangePolicy(qx, ng),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int indices[3] = {i - ng + local[0].first(),
                                        j - ng + local[1].first(),
                                        k - ng + local[2].first()};
                T k2 = 0;
                for (unsigned d = 0; d < Dim; ++d) {
                    const int m = indices[d] <= nr[d] / 2
                                      ? indices[d] : indices[d] - nr[d];
                    const T kd = indices[d] == nr[d] / 2 ? T(0) : twoPi * T(m) / lengths[d];
                    k2 += kd * kd;
                }
                if (k2 == T(0)) {
                    qx(i,j,k) = qy(i,j,k) = qz(i,j,k) = Kokkos::complex<T>(0,0);
                    return;
                }
                const T denominator = T(1) + nu * dt * k2;
                qx(i,j,k) = (qx(i,j,k) + dt * sx(i,j,k) / k2) / denominator;
                qy(i,j,k) = (qy(i,j,k) + dt * sy(i,j,k) / k2) / denominator;
                qz(i,j,k) = (qz(i,j,k) + dt * sz(i,j,k) / k2) / denominator;
            });
        Kokkos::fence();
    }

    void sampleEulerLatticeFromGrid3D() {
        // Steps 12-13: direct IFFT cell-center sampling, with no CIC or type-2
        // NUFFT. Also give particles the UPDATED strengths before transport,
        // so the next type-1 scatter retains the spectral source update.
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto omega = pc->omega.getView();
        auto ox = pc->omega_x.getView();
        auto oy = pc->omega_y.getView();
        auto oz = pc->omega_z.getView();
        auto P = pc->P.getView();
        auto u = pc->u.getView();
        auto wg = this->fcontainer_m->getOmegaField().getView();
        auto ug = this->fcontainer_m->getUField().getView();
        const auto local = this->fcontainer_m->getFL().getLocalNDIndex();
        const auto xmin = this->rmin_m;
        const auto dx = this->hr_m;
        const int ng = this->fcontainer_m->getOmegaField().getNghost();
        const T volume = particleVolume3D();
        Kokkos::parallel_for(
            "sfsl_euler_sample_ifft_lattice", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                const int i = int(Kokkos::floor((R(p)[0] - xmin[0]) / dx[0])) - local[0].first() + ng;
                const int j = int(Kokkos::floor((R(p)[1] - xmin[1]) / dx[1])) - local[1].first() + ng;
                const int k = int(Kokkos::floor((R(p)[2] - xmin[2]) / dx[2])) - local[2].first() + ng;
                for (unsigned d = 0; d < Dim; ++d) {
                    omega(p)[d] = wg(i,j,k)[d] * volume;
                    P(p)[d] = ug(i,j,k)[d];
                    u(p)[d] = P(p)[d];
                }
                ox(p) = omega(p)[0];
                oy(p) = omega(p)[1];
                oz(p) = omega(p)[2];
            });
        Kokkos::fence();
    }

    void advectSpectralEuler3D() {
        // Steps 1-4: the preceding final scatter (or pre_run) prepared q^n
        // and u^n. Retain the existing type-1 NUFFT for off-grid transport.
        computeEulerGridSource3D();
        updateEulerSpectralVorticity3D();

        // Step 11: use final source-updated vorticity in Biot-Savart. Its
        // existing solenoidal projection is retained; no velocity-only decay.
        this->computeSpectralVelocityModes3D();
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        sampleEulerLatticeFromGrid3D();

        // Step 14: Euler transport carries the source-updated strengths.
        // Scatter before discarding virtual particles, then recreate the
        // next cell-center lattice from the transported spectral state.
        pushVirtualParticlesForward3D();
        wrapParticlePositions3D(this->pcontainer_m->R);
        this->pcontainer_m->update();
        this->rebuildNUFFTPlans3D();
        scatterAndSolveCurrentParticles3D();
        clearVirtualParticles3D();
        resetVirtualParticlesToGridFromSpectralModes3D();
    }

#endif
