#ifndef IPPL_SPECTRAL_FSL_3D_MANAGER_H
#define IPPL_SPECTRAL_FSL_3D_MANAGER_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/AlvineManager.h"
#include "fields/FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "particles/ParticleContainer.hpp"
#include "test_cases/TaylorGreen3D.hpp"
#include "Utility/IpplTimings.h"
#include "VtkDump.hpp"

template <typename T>
class SpectralFSL3DManager : public AlvineManager<T, 3> {
public:
    static constexpr unsigned Dim = 3;

    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;

private:
    int diagnostics_freq_m = 1;
    double final_time_m = -1.0;
    bool use_stretching_m = true;
    bool leapfrog_history_valid_m = false;

public:
    SpectralFSL3DManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_,
                         std::string& solver_, int dump_freq_,
                         double dt_ = 0.05,
                         std::string method_ = "sfsl3d",
                         int spectral_filter_ = 0,
                         double viscosity_ = 0.0,
                         std::string time_integrator_ = "euler",
                         Vector_t<double, Dim> rmin_ = 0.0,
                         Vector_t<double, Dim> rmax_ = 1.0,
                         Vector_t<double, Dim> origin_ = 0.0,
                         int diagnostics_freq_ = 1)
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_, dt_, method_,
                                spectral_filter_, viscosity_, time_integrator_)
        , diagnostics_freq_m(diagnostics_freq_) {
        this->rmin_m = rmin_;
        this->rmax_m = rmax_;
        this->origin_m = origin_;
    }

    ~SpectralFSL3DManager() override {}

    void setFinalTime(const double finalTime) {
        final_time_m = finalTime;
    }

    void setUseStretching(const bool enabled) {
        use_stretching_m = enabled;
    }

    void pre_run() override {
        for (unsigned d = 0; d < Dim; ++d) {
            this->domain_m[d] = ippl::Index(this->nr_m[d]);
        }

        const Vector_t<double, Dim> dr = this->rmax_m - this->rmin_m;
        this->hr_m = dr / this->nr_m;
        this->it_m = 0;
        this->time_m = 0.0;
        this->decomp_m.fill(true);
        // BOUNDARY: SFSL 3D uses the same fully periodic TGV domain as VIF 3D.
        this->isAllPeriodic_m = true;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m,
            this->origin_m, this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields();
        this->initNUFFT3D();

        resetVirtualParticlesToGridFromTGV3D();
        scatterAndSolveCurrentParticles3D();
        logDiagnostics3D();
    }

    void advance() override {
        advectForward();
    }

    void run(int nt) {
        for (int it = 0; it < nt; ++it) {
            if (reachedFinalTime3D()) {
                break;
            }
            prepareTimestepForFinalTime3D();
            this->pre_step();
            this->advance();
            this->post_step();
        }
    }

    void post_step() override {
        Inform m("Step: ");
        this->time_m += this->dt_m;
        this->it_m++;

        if (this->dump_freq_m > 0 && this->it_m % this->dump_freq_m == 0) {
            this->dump();
        }

        m << this->it_m << " Done" << endl;
    }

    void dump() override {
        static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("vtkDump");
        IpplTimings::startTimer(dumpTimer);

        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        alvine::vtk::writeVectorField3D("data/SpectralFSL3D", "omega",
                                        this->fcontainer_m->getOmegaField(), this->rmin_m,
                                        this->hr_m, this->it_m);
        alvine::vtk::writeVectorField3D("data/SpectralFSL3D", "velocity",
                                        this->fcontainer_m->getUField(), this->rmin_m,
                                        this->hr_m, this->it_m);

        IpplTimings::stopTimer(dumpTimer);
    }

    void par2grid() override {}

    void grid2par() override {}

public:
    bool shouldLogDiagnostics3D() const {
        return diagnostics_freq_m > 0
               && static_cast<int>(this->it_m) % diagnostics_freq_m == 0;
    }

    bool reachedFinalTime3D() const {
        return final_time_m >= 0.0 && this->time_m + 1e-14 >= final_time_m;
    }

    double activeMaximumTimestep3D() const {
        if (final_time_m < 0.0) {
            return this->dt_max_m;
        }
        const double remaining = final_time_m - this->time_m;
        return std::min(this->dt_max_m, std::max(0.0, remaining));
    }

    void prepareTimestepForFinalTime3D() {
        this->dt_m = activeMaximumTimestep3D();
    }

    unsigned particlesPerDirection3D() const {
        const double n = std::round(std::cbrt(static_cast<double>(this->np_m)));
        return static_cast<unsigned>(n);
    }

    T particleVolume3D() const {
        const unsigned nxp = particlesPerDirection3D();
        const T dxp = T(this->rmax_m[0] - this->rmin_m[0]) / T(nxp);
        const T dyp = T(this->rmax_m[1] - this->rmin_m[1]) / T(nxp);
        const T dzp = T(this->rmax_m[2] - this->rmin_m[2]) / T(nxp);
        return dxp * dyp * dzp;
    }

    void scatterAndSolveCurrentParticles3D() {
        this->spectralScatter3D(false);
        if (this->useShapeFunctionFilter()) {
            this->applyShapeFunctionToSpectralVorticityModes3D();
        }
        this->applyConfiguredSpectralFilter3D(this->omega_x_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_y_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_z_hat_m);
        this->computeSpectralVelocityModes3D();
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
    }

    void logDiagnostics3D() {
        if (!shouldLogDiagnostics3D()) {
            return;
        }
        refreshParticleGradientForDiagnostics3D();
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        this->logTaylorGreenDiagnostics3D();
        this->logSpectralDiagnostics3D();
    }

    void refreshParticleGradientForDiagnostics3D() {
        this->computeSpectralVelocityGradientModes3D();
        this->spectralGatherGradientModes3D();
    }

    void resetVirtualParticlesToGridFromTGV3D() {
        createGridLatticeParticles3D();
        leapfrog_history_valid_m = false;

        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto Rold = pc->R_old.getView();
        auto omega = pc->omega.getView();
        auto omegaX = pc->omega_x.getView();
        auto omegaY = pc->omega_y.getView();
        auto omegaZ = pc->omega_z.getView();
        const auto nlocal = pc->getLocalNum();
        const T particleVolume = particleVolume3D();

        Kokkos::parallel_for(
            "initialize_sfsl3d_tgv_virtual_particles",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                const T x = R(p)[0];
                const T y = R(p)[1];
                const T z = R(p)[2];
                const auto vort = TaylorGreen3D<T>::vorticity(x, y, z) * particleVolume;
                omega(p) = vort;
                omegaX(p) = vort[0];
                omegaY(p) = vort[1];
                omegaZ(p) = vort[2];
                Rold(p) = R(p);
            });
        Kokkos::fence();
    }

    void resetVirtualParticlesToGridFromSpectralModes3D() {
        createGridLatticeParticles3D();
        sampleGridParticlesFromSpectralModes3D(particleVolume3D());
        leapfrog_history_valid_m = false;
    }

    void createGridLatticeParticles3D() {
        auto* FL = &this->fcontainer_m->getFL();
        auto local = FL->getLocalNDIndex();

        const unsigned nxpGlobal = particlesPerDirection3D();
        const unsigned nypGlobal = nxpGlobal;
        const unsigned nzpGlobal = nxpGlobal;
        if (nxpGlobal * nypGlobal * nzpGlobal != this->np_m) {
            throw std::runtime_error("SFSL 3D requires np to be a perfect cube.");
        }

        const int localStartX = local[0].first();
        const int localEndX = local[0].last();
        const int localStartY = local[1].first();
        const int localEndY = local[1].last();
        const int localStartZ = local[2].first();
        const int localEndZ = local[2].last();

        const double dxp = (this->rmax_m[0] - this->rmin_m[0]) / nxpGlobal;
        const double dyp = (this->rmax_m[1] - this->rmin_m[1]) / nypGlobal;
        const double dzp = (this->rmax_m[2] - this->rmin_m[2]) / nzpGlobal;

        int ixStart = static_cast<int>(std::ceil((localStartX * this->hr_m[0] - 0.5 * dxp) / dxp));
        int ixEnd = static_cast<int>(std::floor(((localEndX + 1) * this->hr_m[0] - 0.5 * dxp) / dxp));
        int iyStart = static_cast<int>(std::ceil((localStartY * this->hr_m[1] - 0.5 * dyp) / dyp));
        int iyEnd = static_cast<int>(std::floor(((localEndY + 1) * this->hr_m[1] - 0.5 * dyp) / dyp));
        int izStart = static_cast<int>(std::ceil((localStartZ * this->hr_m[2] - 0.5 * dzp) / dzp));
        int izEnd = static_cast<int>(std::floor(((localEndZ + 1) * this->hr_m[2] - 0.5 * dzp) / dzp));

        ixStart = std::max(0, ixStart);
        iyStart = std::max(0, iyStart);
        izStart = std::max(0, izStart);
        ixEnd = std::min(static_cast<int>(nxpGlobal - 1), ixEnd);
        iyEnd = std::min(static_cast<int>(nypGlobal - 1), iyEnd);
        izEnd = std::min(static_cast<int>(nzpGlobal - 1), izEnd);

        const bool hasLocalLattice =
            ixEnd >= ixStart && iyEnd >= iyStart && izEnd >= izStart;
        const unsigned nxpLocal = hasLocalLattice ? ixEnd - ixStart + 1 : 0;
        const unsigned nypLocal = hasLocalLattice ? iyEnd - iyStart + 1 : 0;
        const unsigned nzpLocal = hasLocalLattice ? izEnd - izStart + 1 : 0;
        const size_type nlocalNew = nxpLocal * nypLocal * nzpLocal;

        auto pc = this->pcontainer_m;
        pc->create(nlocalNew);

        auto R = pc->R.getView();
        auto Rold = pc->R_old.getView();
        const auto nlocal = pc->getLocalNum();
        const T xmin = T(this->rmin_m[0]);
        const T ymin = T(this->rmin_m[1]);
        const T zmin = T(this->rmin_m[2]);
        const T dx = T(dxp);
        const T dy = T(dyp);
        const T dz = T(dzp);

        Kokkos::parallel_for(
            "place_sfsl3d_grid_lattice_particles",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                const unsigned ixLocal = p % nxpLocal;
                const unsigned iyLocal = (p / nxpLocal) % nypLocal;
                const unsigned izLocal = p / (nxpLocal * nypLocal);
                const unsigned ixGlobal = ixStart + ixLocal;
                const unsigned iyGlobal = iyStart + iyLocal;
                const unsigned izGlobal = izStart + izLocal;

                R(p)[0] = xmin + (T(ixGlobal) + T(0.5)) * dx;
                R(p)[1] = ymin + (T(iyGlobal) + T(0.5)) * dy;
                R(p)[2] = zmin + (T(izGlobal) + T(0.5)) * dz;
                Rold(p) = R(p);
            });
        Kokkos::fence();
        pc->update();
        this->rebuildNUFFTPlans3D();
    }

    void clearVirtualParticles3D() {
        auto pc = this->pcontainer_m;
        const size_type nlocal = pc->getLocalNum();
        if (nlocal == 0) {
            return;
        }

        Kokkos::View<bool*> invalid("sfsl3d_invalid_virtual_particles", nlocal);
        Kokkos::parallel_for(
            "mark_sfsl3d_virtual_particles_invalid",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                invalid(p) = true;
            });
        Kokkos::fence();
        pc->destroy(invalid, nlocal);
    }

    void sampleGridParticlesFromSpectralModes3D(const T particleVolume) {
        if (!this->nufftType2_mp) {
            throw std::runtime_error("SFSL 3D type-2 sampling called before initNUFFT3D.");
        }

        auto pc = this->pcontainer_m;
        pc->omega_x = 0.0;
        pc->omega_y = 0.0;
        pc->omega_z = 0.0;

        auto oxModes = this->omega_x_hat_m.deepCopy();
        auto oyModes = this->omega_y_hat_m.deepCopy();
        auto ozModes = this->omega_z_hat_m.deepCopy();
        recoverPhysicalVorticityModesForSampling3D(oxModes, oyModes, ozModes);

        this->nufftType2_mp->transform(pc->R, pc->omega_x, oxModes);
        this->nufftType2_mp->transform(pc->R, pc->omega_y, oyModes);
        this->nufftType2_mp->transform(pc->R, pc->omega_z, ozModes);

        auto omega = pc->omega.getView();
        auto omegaX = pc->omega_x.getView();
        auto omegaY = pc->omega_y.getView();
        auto omegaZ = pc->omega_z.getView();
        const auto nlocal = pc->getLocalNum();

        Kokkos::parallel_for(
            "pack_sfsl3d_type2_vorticity_samples",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                omega(p)[0] = omegaX(p) * particleVolume;
                omega(p)[1] = omegaY(p) * particleVolume;
                omega(p)[2] = omegaZ(p) * particleVolume;
                omegaX(p) = omega(p)[0];
                omegaY(p) = omega(p)[1];
                omegaZ(p) = omega(p)[2];
            });
        Kokkos::fence();
    }

    void recoverPhysicalVorticityModesForSampling3D(typename AlvineManager<T, Dim>::ComplexField_t& oxModes,
                                                    typename AlvineManager<T, Dim>::ComplexField_t& oyModes,
                                                    typename AlvineManager<T, Dim>::ComplexField_t& ozModes) {
        auto ox = oxModes.getView();
        auto oy = oyModes.getView();
        auto oz = ozModes.getView();
        auto& layout = oxModes.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = oxModes.getNghost();
        const int Nx = this->nr_m[0];
        const int Ny = this->nr_m[1];
        const int Nz = this->nr_m[2];
        const T Lx = this->rmax_m[0] - this->rmin_m[0];
        const T Ly = this->rmax_m[1] - this->rmin_m[1];
        const T Lz = this->rmax_m[2] - this->rmin_m[2];
        const T twoPi = T(2.0 * std::acos(-1.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "recover_sfsl3d_physical_vorticity_modes",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ox.extent(0)) - nghost,
                         static_cast<int>(ox.extent(1)) - nghost,
                         static_cast<int>(ox.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();
                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const int mz = (gz <= Nz / 2) ? gz : gz - Nz;
                const bool notMidX = (gx != Nx / 2);
                const bool notMidY = (gy != Ny / 2);
                const bool notMidZ = (gz != Nz / 2);
                const T kx = notMidX * twoPi * mx / Lx;
                const T ky = notMidY * twoPi * my / Ly;
                const T kz = notMidZ * twoPi * mz / Lz;
                const T k2 = kx * kx + ky * ky + kz * kz;
                ox(i, j, k) *= k2;
                oy(i, j, k) *= k2;
                oz(i, j, k) *= k2;
            });
        Kokkos::fence();
    }

    void advectForward() {
        static IpplTimings::TimerRef solveTimer = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("spectralGather");
        static IpplTimings::TimerRef pushTimer = IpplTimings::getTimer("sfsl3dPushParticles");
        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("spectralScatter");

        auto pc = this->pcontainer_m;

        IpplTimings::startTimer(solveTimer);
        this->computeSpectralVelocityModes3D();
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
        if (use_stretching_m || this->adaptive_lcfl_m) {
            this->computeSpectralVelocityGradientModes3D();
        }
        if (this->viscosity_m > 0.0) {
            this->viscosity_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_z_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->computeSpectralViscosityModes3D();
        }
        IpplTimings::stopTimer(solveTimer);

        IpplTimings::startTimer(gatherTimer);
        this->spectralGather3D();
        if (use_stretching_m || this->adaptive_lcfl_m) {
            this->spectralGatherGradientModes3D();
        }
        if (this->viscosity_m > 0.0) {
            this->spectralGatherViscosity3D();
        }
        IpplTimings::stopTimer(gatherTimer);

        if (this->adaptive_lcfl_m) {
            updateLCFLTimestep3D();
        }

        if (use_stretching_m) {
            this->applyParticleVortexStretching3D();
        }
        if (this->viscosity_m > 0.0) {
            this->applyParticleViscosity3D();
        }

        IpplTimings::startTimer(pushTimer);
        pushVirtualParticlesForward3D();
        if (!this->useRK4()) {
            wrapParticlePositions3D(pc->R);
            pc->update();
            this->rebuildNUFFTPlans3D();
        }
        IpplTimings::stopTimer(pushTimer);

        IpplTimings::startTimer(scatterTimer);
        scatterAndSolveCurrentParticles3D();
        IpplTimings::stopTimer(scatterTimer);

        logDiagnostics3D();
        clearVirtualParticles3D();
        resetVirtualParticlesToGridFromSpectralModes3D();
    }

    void pushVirtualParticlesForward3D() {
        if (this->useRK4()) {
            pushVirtualParticlesForwardRK4_3D();
            return;
        }

        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto Rold = pc->R_old.getView();
        auto P = pc->P.getView();
        const auto nlocal = pc->getLocalNum();
        const T dt = T(this->dt_m);
        const bool useLeapfrogPush = this->useLeapFrog() && leapfrog_history_valid_m;

        Kokkos::parallel_for(
            "push_sfsl3d_virtual_particles",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                const auto Rn = R(p);
                if (useLeapfrogPush) {
                    R(p)[0] = Rold(p)[0] + T(2.0) * P(p)[0] * dt;
                    R(p)[1] = Rold(p)[1] + T(2.0) * P(p)[1] * dt;
                    R(p)[2] = Rold(p)[2] + T(2.0) * P(p)[2] * dt;
                } else {
                    R(p)[0] += P(p)[0] * dt;
                    R(p)[1] += P(p)[1] * dt;
                    R(p)[2] += P(p)[2] * dt;
                }
                Rold(p) = Rn;
            });
        Kokkos::fence();
        leapfrog_history_valid_m = true;
    }

    void pushVirtualParticlesForwardRK4_3D() {
        auto pc = this->pcontainer_m;
        const T dt = T(this->dt_m);

        pc->rk4_R0 = pc->R;
        pc->rk4_k1 = pc->P;

        pc->R = pc->rk4_R0 + (T(0.5) * dt) * pc->rk4_k1;
        wrapParticlePositions3D(pc->R);
        pc->update();
        this->rebuildNUFFTPlans3D();
        this->spectralGather3D();
        pc->rk4_k2 = pc->P;

        pc->R = pc->rk4_R0 + (T(0.5) * dt) * pc->rk4_k2;
        wrapParticlePositions3D(pc->R);
        pc->update();
        this->rebuildNUFFTPlans3D();
        this->spectralGather3D();
        pc->rk4_k3 = pc->P;

        pc->R = pc->rk4_R0 + dt * pc->rk4_k3;
        wrapParticlePositions3D(pc->R);
        pc->update();
        this->rebuildNUFFTPlans3D();
        this->spectralGather3D();
        pc->rk4_k4 = pc->P;

        pc->R_old = pc->rk4_R0;
        pc->R = pc->rk4_R0
                + (dt / T(6.0)) * (pc->rk4_k1 + T(2.0) * pc->rk4_k2
                                   + T(2.0) * pc->rk4_k3 + pc->rk4_k4);
        wrapParticlePositions3D(pc->R);
        pc->update();
        this->rebuildNUFFTPlans3D();
        leapfrog_history_valid_m = false;
    }

    void updateLCFLTimestep3D() {
        if (!this->adaptive_lcfl_m) {
            return;
        }
        const double deformationOneNorm = this->computeParticleDeformationOneNorm3D();
        const double lcflDt = deformationOneNorm > 0.0
            ? this->lcfl_m / deformationOneNorm
            : this->dt_m;
        this->dt_m = std::min(activeMaximumTimestep3D(), lcflDt);
    }

    void wrapParticlePositions3D(typename ParticleContainer_t::particle_position_type& positions) {
        auto R = positions.getView();
        const auto nlocal = this->pcontainer_m->getLocalNum();
        const T xmin = this->rmin_m[0];
        const T ymin = this->rmin_m[1];
        const T zmin = this->rmin_m[2];
        const T Lx = this->rmax_m[0] - this->rmin_m[0];
        const T Ly = this->rmax_m[1] - this->rmin_m[1];
        const T Lz = this->rmax_m[2] - this->rmin_m[2];

        Kokkos::parallel_for(
            "wrap_sfsl3d_particle_positions",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                R(p)[0] = xmin + (R(p)[0] - xmin) - Lx * Kokkos::floor((R(p)[0] - xmin) / Lx);
                R(p)[1] = ymin + (R(p)[1] - ymin) - Ly * Kokkos::floor((R(p)[1] - ymin) / Ly);
                R(p)[2] = zmin + (R(p)[2] - zmin) - Lz * Kokkos::floor((R(p)[2] - zmin) / Lz);
            });
        Kokkos::fence();
    }
};

#endif
