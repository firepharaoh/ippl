#ifndef IPPL_ALVINE_SFSL_COUPLED_STRANG_3D_HPP
#define IPPL_ALVINE_SFSL_COUPLED_STRANG_3D_HPP

// Included in the manager's public section for CUDA extended lambdas.
    void evaluateCoupledStrangRHS3D(
        typename ParticleContainer_t::particle_position_type& velocityRHS,
        typename ParticleContainer_t::particle_position_type& strengthRHS,
        const std::string& label) {
        // PDF block 2: every stage scatters its OWN positions and strengths.
        // No filtering, diffusion, or persistent particle update in the RHS.
        this->spectralScatter3D(false);
        this->computeSpectralVelocityModes3D(false);
        this->spectralGather3D(false);
        if (use_stretching_m) {
            this->computeSpectralVelocityGradientModes3D();
            this->spectralGatherGradientModes3D(false);
        }
        auto pc = this->pcontainer_m;
        velocityRHS = pc->P;
        auto w = pc->omega.getView();
        auto rhs = strengthRHS.getView();
        auto xx = pc->duxdx.getView();
        auto xy = pc->duxdy.getView();
        auto xz = pc->duxdz.getView();
        auto yx = pc->duydx.getView();
        auto yy = pc->duydy.getView();
        auto yz = pc->duydz.getView();
        auto zx = pc->duzdx.getView();
        auto zy = pc->duzdy.getView();
        auto zz = pc->duzdz.getView();
        const bool stretching = use_stretching_m;
        Kokkos::parallel_for("sfsl_coupled_strang_rhs", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                // w is Gamma=omega*particleVolume, so grad(u)*Gamma already
                // has strength/time units. Do not multiply by volume again.
                rhs(p)[0] = stretching ? w(p)[0]*xx(p)+w(p)[1]*xy(p)+w(p)[2]*xz(p) : T(0);
                rhs(p)[1] = stretching ? w(p)[0]*yx(p)+w(p)[1]*yy(p)+w(p)[2]*yz(p) : T(0);
                rhs(p)[2] = stretching ? w(p)[0]*zx(p)+w(p)[1]*zy(p)+w(p)[2]*zz(p) : T(0);
            });
        Kokkos::fence();
        logRK4StageSpectralState3D(label);
    }

    void setCoupledStrangStage3D(
        typename ParticleContainer_t::particle_position_type& kR,
        typename ParticleContainer_t::particle_position_type& kW, const T scale) {
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto w = pc->omega.getView();
        auto wx = pc->omega_x.getView();
        auto wy = pc->omega_y.getView();
        auto wz = pc->omega_z.getView();
        auto R0 = pc->rk4_R0.getView();
        auto w0 = pc->rk4_omega0.getView();
        auto rRHS = kR.getView();
        auto wRHS = kW.getView();
        const T h = scale * T(this->dt_m);
        Kokkos::parallel_for("sfsl_coupled_strang_stage", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                for (unsigned d=0; d<Dim; ++d) {
                    R(p)[d] = R0(p)[d] + h*rRHS(p)[d];
                    w(p)[d] = w0(p)[d] + h*wRHS(p)[d];
                }
                wx(p) = w(p)[0]; wy(p) = w(p)[1]; wz(p) = w(p)[2];
            });
        Kokkos::fence();
        wrapParticlePositions3D(pc->R);
        // Registered base states and all slopes migrate with the particles.
        pc->update();
        this->rebuildNUFFTPlans3D();
    }

    void advanceCoupledNonlinearRK4_3D() {
        auto pc = this->pcontainer_m;
        pc->rk4_R0 = pc->R;
        pc->rk4_omega0 = pc->omega;
        evaluateCoupledStrangRHS3D(pc->rk4_k1, pc->rk4_omega_k1, "coupled_k1");
        setCoupledStrangStage3D(pc->rk4_k1, pc->rk4_omega_k1, T(0.5));
        evaluateCoupledStrangRHS3D(pc->rk4_k2, pc->rk4_omega_k2, "coupled_k2");
        setCoupledStrangStage3D(pc->rk4_k2, pc->rk4_omega_k2, T(0.5));
        evaluateCoupledStrangRHS3D(pc->rk4_k3, pc->rk4_omega_k3, "coupled_k3");
        setCoupledStrangStage3D(pc->rk4_k3, pc->rk4_omega_k3, T(1));
        evaluateCoupledStrangRHS3D(pc->rk4_k4, pc->rk4_omega_k4, "coupled_k4");

        auto R = pc->R.getView();
        auto Rold = pc->R_old.getView();
        auto w = pc->omega.getView();
        auto wx = pc->omega_x.getView();
        auto wy = pc->omega_y.getView();
        auto wz = pc->omega_z.getView();
        auto R0 = pc->rk4_R0.getView();
        auto w0 = pc->rk4_omega0.getView();
        auto r1 = pc->rk4_k1.getView(); auto r2 = pc->rk4_k2.getView();
        auto r3 = pc->rk4_k3.getView(); auto r4 = pc->rk4_k4.getView();
        auto w1 = pc->rk4_omega_k1.getView(); auto w2 = pc->rk4_omega_k2.getView();
        auto w3 = pc->rk4_omega_k3.getView(); auto w4 = pc->rk4_omega_k4.getView();
        const T h = T(this->dt_m)/T(6);
        Kokkos::parallel_for("sfsl_coupled_strang_combine", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                Rold(p) = R0(p);
                for (unsigned d=0; d<Dim; ++d) {
                    R(p)[d] = R0(p)[d] + h*(r1(p)[d]+T(2)*r2(p)[d]+T(2)*r3(p)[d]+r4(p)[d]);
                    w(p)[d] = w0(p)[d] + h*(w1(p)[d]+T(2)*w2(p)[d]+T(2)*w3(p)[d]+w4(p)[d]);
                }
                wx(p) = w(p)[0]; wy(p) = w(p)[1]; wz(p) = w(p)[2];
            });
        Kokkos::fence();
        wrapParticlePositions3D(pc->R);
        pc->update();
        this->rebuildNUFFTPlans3D();
    }

    void advanceCoupledStrang3D() {
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("sfsl3dCoupledStrang");
        IpplTimings::startTimer(timer);
        this->computeSpectralVelocityModes3D();
        if (this->adaptive_lcfl_m) computeEulerGridSource3D(true);
        const T halfDt = T(this->dt_m)/T(2);
        logRK4StageSpectralState3D("start");

        // PDF block 1: exact spectral diffusion, then refresh lattice strengths.
        // SFSL already retains the input modes; no redundant initial scatter.
        applyStrangSpectralDiffusion3D(halfDt);
        this->computeSpectralVelocityModes3D(false);
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        sampleEulerLatticeFromGrid3D();
        logRK4StageSpectralState3D("after_first_diffusion");

        // PDF block 2: ONE coupled RK4 solve for (R,Gamma), without viscosity.
        advanceCoupledNonlinearRK4_3D();

        // PDF block 3: scatter the combined state, not the fourth RK stage.
        this->spectralScatter3D(false);
        logRK4StageSpectralState3D("after_coupled_combination");
        applyStrangSpectralDiffusion3D(halfDt);
        logRK4StageSpectralState3D("after_second_diffusion");
        if (this->useShapeFunctionFilter()) this->applyShapeFunctionToSpectralVorticityModes3D();
        this->applyConfiguredSpectralFilter3D(this->omega_x_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_y_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_z_hat_m);
        this->computeSpectralVelocityModes3D();
        // SFSL convention: final spectral state is authoritative. Recreate the
        // next lattice from it, rather than retain the PDF's off-grid particles.
        clearVirtualParticles3D();
        resetVirtualParticlesToGridFromSpectralModes3D();
        logRK4StageSpectralState3D("after_final_lattice_reset");
        IpplTimings::stopTimer(timer);
    }

#endif
