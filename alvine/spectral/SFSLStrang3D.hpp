#ifndef IPPL_ALVINE_SFSL_STRANG_3D_HPP
#define IPPL_ALVINE_SFSL_STRANG_3D_HPP

// Included in the public section of SpectralFSL3DManager. CUDA extended
// lambdas require public kernel launchers.
    void applyStrangSpectralDiffusion3D(const T duration) {
        if (this->viscosity_m == 0.0 || duration == T(0)) {
            return;
        }
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("sfsl3dStrangDiffusion");
        IpplTimings::startTimer(timer);
        auto qx = this->omega_x_hat_m.getView();
        auto qy = this->omega_y_hat_m.getView();
        auto qz = this->omega_z_hat_m.getView();
        const auto local = this->omega_x_hat_m.getLayout().getLocalNDIndex();
        const auto nr = this->nr_m;
        const Vector_t<T, Dim> lengths = this->rmax_m - this->rmin_m;
        const int ng = this->omega_x_hat_m.getNghost();
        const T twoPi = T(2) * T(std::acos(-1.0));
        const T nu = T(this->viscosity_m);

        // PDF blocks 1 and 3: exact diffusion of VORTICITY over a half-step.
        // Since q=omega_hat/k^2, the same exponential multiplies q directly.
        // Match the existing zero/Nyquist wave-number convention.
        Kokkos::parallel_for(
            "sfsl_strang_spectral_diffusion", ippl::getRangePolicy(qx, ng),
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
                const T factor = Kokkos::exp(-nu * k2 * duration);
                qx(i,j,k) *= factor;
                qy(i,j,k) *= factor;
                qz(i,j,k) *= factor;
            });
        Kokkos::fence();
        IpplTimings::stopTimer(timer);
    }

    void accumulateStrangStretchingStage3D(ComplexField_t& qField,
                                           ComplexField_t& baseField,
                                           ComplexField_t& sumField,
                                           ComplexField_t& sourceField,
                                           const T duration, const T nextScale,
                                           const T weight, const bool finalStage) {
        auto q = qField.getView();
        auto base = baseField.getView();
        auto sum = sumField.getView();
        auto source = sourceField.getView();
        const auto local = qField.getLayout().getLocalNDIndex();
        const auto nr = this->nr_m;
        const Vector_t<T, Dim> lengths = this->rmax_m - this->rmin_m;
        const int ng = qField.getNghost();
        const T twoPi = T(2) * T(std::acos(-1.0));
        Kokkos::parallel_for(
            "sfsl_strang_rk4_spectral_stage", ippl::getRangePolicy(q, ng),
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
                    q(i,j,k) = sum(i,j,k) = Kokkos::complex<T>(0,0);
                    return;
                }
                // The grid FFT returns physical S_hat; dq/dt=S_hat/k^2.
                // Every temporary state is based on q_base, never cumulative.
                const auto rhs = source(i,j,k) / k2;
                sum(i,j,k) += weight * rhs;
                q(i,j,k) = base(i,j,k) + duration
                    * (finalStage ? sum(i,j,k) / T(6) : nextScale * rhs);
            });
        Kokkos::fence();
    }

    void advanceStrangStretchingRK4_3D(const T duration, const std::string& label) {
        if (!use_stretching_m) return;
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("sfsl3dStrangStretching");
        IpplTimings::startTimer(timer);
        std::array<ComplexField_t*, 3> q = {
            &this->omega_x_hat_m, &this->omega_y_hat_m, &this->omega_z_hat_m};
        std::array<ComplexField_t*, 3> source = {&euler_sx_m, &euler_sy_m, &euler_sz_m};
        for (unsigned d = 0; d < Dim; ++d) {
            Kokkos::deep_copy(strang_base_m[d].getView(), q[d]->getView());
            strang_sum_m[d] = Kokkos::complex<T>(0,0);
        }
        const T scales[4] = {T(0.5), T(0.5), T(1), T(0)};
        const T weights[4] = {T(1), T(2), T(2), T(1)};
        for (int stage = 0; stage < 4; ++stage) {
            // PDF block 2, SFSL adaptation: reconstruct this stage's velocity,
            // IFFT omega and grad(u), multiply on the grid, and FFT S back.
            // No particle RHS or stretching gather, no viscosity/filter here.
            this->computeSpectralVelocityModes3D(false);
            computeEulerGridSource3D(false);
            logRK4StageSpectralState3D(label + "_k" + std::to_string(stage + 1));
            for (unsigned d = 0; d < Dim; ++d) {
                accumulateStrangStretchingStage3D(*q[d], strang_base_m[d],
                    strang_sum_m[d], *source[d], duration, scales[stage],
                    weights[stage], stage == 3);
            }
        }
        IpplTimings::stopTimer(timer);
    }

    void setStrangAdvectionStage3D(
        typename ParticleContainer_t::particle_position_type& derivative, const T scale) {
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto base = pc->rk4_R0.getView();
        auto rhs = derivative.getView();
        const T interval = scale * T(this->dt_m);
        Kokkos::parallel_for("sfsl_strang_advection_stage", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                for (unsigned d = 0; d < Dim; ++d) {
                    R(p)[d] = base(p)[d] + interval * rhs(p)[d];
                }
            });
        Kokkos::fence();
        wrapParticlePositions3D(pc->R);
        // Registered R0 and RK slopes migrate with each particle.
        pc->update();
        this->rebuildNUFFTPlans3D();
    }

    void evaluateStrangAdvectionRHS3D(const std::string& label) {
        // Particle strengths are constant throughout this transport subflow.
        // Re-scatter the CURRENT stage positions for a fresh stage velocity.
        this->spectralScatter3D(false);
        this->computeSpectralVelocityModes3D(false);
        this->spectralGather3D();
        logRK4StageSpectralState3D(label);
    }

    void advanceStrangAdvectionRK4_3D() {
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("sfsl3dStrangAdvection");
        IpplTimings::startTimer(timer);
        auto pc = this->pcontainer_m;
        this->computeSpectralVelocityModes3D(false);
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        // Cell-center particles receive the first diffusion/stretching results.
        // Direct IFFT sampling here; later RK positions are off-grid and need
        // type-2 velocity evaluation. Never sample a grid cell by floor there.
        sampleEulerLatticeFromGrid3D();
        pc->rk4_R0 = pc->R;
        pc->rk4_k1 = pc->P;
        logRK4StageSpectralState3D("advection_k1");

        setStrangAdvectionStage3D(pc->rk4_k1, T(0.5));
        evaluateStrangAdvectionRHS3D("advection_k2");
        pc->rk4_k2 = pc->P;
        setStrangAdvectionStage3D(pc->rk4_k2, T(0.5));
        evaluateStrangAdvectionRHS3D("advection_k3");
        pc->rk4_k3 = pc->P;
        setStrangAdvectionStage3D(pc->rk4_k3, T(1));
        evaluateStrangAdvectionRHS3D("advection_k4");
        pc->rk4_k4 = pc->P;

        auto R = pc->R.getView();
        auto Rold = pc->R_old.getView();
        auto base = pc->rk4_R0.getView();
        auto k1 = pc->rk4_k1.getView();
        auto k2 = pc->rk4_k2.getView();
        auto k3 = pc->rk4_k3.getView();
        auto k4 = pc->rk4_k4.getView();
        const T sixthDt = T(this->dt_m) / T(6);
        Kokkos::parallel_for("sfsl_strang_finalize_advection", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                Rold(p) = base(p);
                for (unsigned d = 0; d < Dim; ++d) {
                    R(p)[d] = base(p)[d] + sixthDt
                        * (k1(p)[d] + T(2)*k2(p)[d] + T(2)*k3(p)[d] + k4(p)[d]);
                }
            });
        Kokkos::fence();
        wrapParticlePositions3D(pc->R);
        pc->update();
        this->rebuildNUFFTPlans3D();
        this->spectralScatter3D(false);
        logRK4StageSpectralState3D("after_advection");
        IpplTimings::stopTimer(timer);
    }

    void advectForwardRK4_3D() {
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("sfsl3dStrangRK4");
        IpplTimings::startTimer(timer);
        // Choose ONE dt from the input state, before either diffusion half.
        // The source helper's default adaptation is disabled in all RK stages.
        this->computeSpectralVelocityModes3D();
        if (this->adaptive_lcfl_m) computeEulerGridSource3D(true);
        const T halfDt = T(0.5) * T(this->dt_m);
        logRK4StageSpectralState3D("start");

        // PDF block 1: D(dt/2). No scatter/resampling is needed to diffuse q.
        applyStrangSpectralDiffusion3D(halfDt);
        logRK4StageSpectralState3D("after_first_diffusion");

        // PDF block 2, grid-only SFSL adaptation: S(dt/2) A(dt) S(dt/2).
        // This additional symmetric split keeps stretching in Fourier space.
        // Each nonlinear subflow uses RK4; the FULL composition is order two.
        advanceStrangStretchingRK4_3D(halfDt, "first_stretching");
        advanceStrangAdvectionRK4_3D();
        advanceStrangStretchingRK4_3D(halfDt, "second_stretching");

        // PDF block 3: D(dt/2) on the final nonlinear state, exactly once.
        applyStrangSpectralDiffusion3D(halfDt);
        logRK4StageSpectralState3D("after_second_diffusion");

        // Filter only the completed timestep, not each RK evaluation. Split
        // subflows may have div(omega)!=0; retain the existing projection at
        // the completed composition, rather than altering the split operators.
        if (this->useShapeFunctionFilter()) this->applyShapeFunctionToSpectralVorticityModes3D();
        this->applyConfiguredSpectralFilter3D(this->omega_x_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_y_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_z_hat_m);
        this->computeSpectralVelocityModes3D();

        // Complete SFSL lifecycle AFTER the last spectral source update.
        // Do not re-scatter stale particle strengths after the second half.
        clearVirtualParticles3D();
        resetVirtualParticlesToGridFromSpectralModes3D();
        logRK4StageSpectralState3D("after_final_lattice_reset");
        IpplTimings::stopTimer(timer);
    }

#endif
