#ifndef IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H
#define IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/AlvineManager.h"
#include "FieldSolver.hpp"
#include "fields/FieldContainer.hpp"
#include "particles/ParticleContainer.hpp"
#include "test_cases/TaylorGreen3D.hpp"
#include "VtkDump.hpp"

template <typename T>
class VortexInFourier3DManager : public AlvineManager<T, 3> {
public:
    static constexpr unsigned Dim = 3;

    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;

private:
    int remesh_freq_m = 0;
    bool bootstrap_next_push_m = false;

public:
    VortexInFourier3DManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_,
                             std::string& solver_, int dump_freq_,
                             double dt_ = 0.05,
                             std::string method_ = "vif3d",
                             int spectral_filter_ = 0,
                             double viscosity_ = 0.0,
                             std::string time_integrator_ = "leapfrog",
                             Vector_t<double, Dim> rmin_ = 0.0,
                             Vector_t<double, Dim> rmax_ = 1.0,
                             Vector_t<double, Dim> origin_ = 0.0,
                             int remesh_freq_ = 0)
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_, dt_, method_,
                                spectral_filter_, viscosity_, time_integrator_)
        , remesh_freq_m(remesh_freq_) {
        this->rmin_m = rmin_;
        this->rmax_m = rmax_;
        this->origin_m = origin_;
    }

    ~VortexInFourier3DManager() override {}

    void pre_run() override {
        for (unsigned d = 0; d < Dim; ++d) {
            this->domain_m[d] = ippl::Index(this->nr_m[d]);
        }

        const Vector_t<double, Dim> dr = this->rmax_m - this->rmin_m;
        this->hr_m = dr / this->nr_m;
        this->it_m = 0;
        this->time_m = 0.0;
        this->decomp_m.fill(true);
        this->isAllPeriodic_m = true;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m,
            this->origin_m, this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields();

        initializeParticles();
        this->initNUFFT3D();
    }

    void advance() override {
        if (this->useRK4()) {
            RK4Step();
        } else {
            LeapFrogStep();
        }
    }

    void remeshParticles3D() {
        reconstructFieldsForRemeshing3D();
        remeshParticlesFromGrid3D();
        bootstrap_next_push_m = true;
    }

    void reconstructFieldsForRemeshing3D() {
        this->spectralScatter3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_x_hat_m);
            this->Hou_Li_filter(this->omega_y_hat_m);
            this->Hou_Li_filter(this->omega_z_hat_m);
        }

        this->computeSpectralVelocityModes3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
            this->Hou_Li_filter(this->uz_hat_m);
        }

        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
    }

    void remeshParticlesFromGrid3D() {
        auto* FL = &this->fcontainer_m->getFL();
        auto local = FL->getLocalNDIndex();

        const unsigned nxp_global = particlesPerDirection3D();
        const unsigned nyp_global = nxp_global;
        const unsigned nzp_global = nxp_global;

        if (nxp_global * nyp_global * nzp_global != this->np_m) {
            throw std::runtime_error("VIF 3D remeshing requires np to be a perfect cube.");
        }

        const double xmin_global = this->rmin_m[0];
        const double xmax_global = this->rmax_m[0];
        const double ymin_global = this->rmin_m[1];
        const double ymax_global = this->rmax_m[1];
        const double zmin_global = this->rmin_m[2];
        const double zmax_global = this->rmax_m[2];

        const double dxp = (xmax_global - xmin_global) / nxp_global;
        const double dyp = (ymax_global - ymin_global) / nyp_global;
        const double dzp = (zmax_global - zmin_global) / nzp_global;

        const int local_start_x = local[0].first();
        const int local_end_x   = local[0].last();
        const int local_start_y = local[1].first();
        const int local_end_y   = local[1].last();
        const int local_start_z = local[2].first();
        const int local_end_z   = local[2].last();

        const double xmin_local = xmin_global + local_start_x * this->hr_m[0];
        const double xmax_local = xmin_global + (local_end_x + 1) * this->hr_m[0];
        const double ymin_local = ymin_global + local_start_y * this->hr_m[1];
        const double ymax_local = ymin_global + (local_end_y + 1) * this->hr_m[1];
        const double zmin_local = zmin_global + local_start_z * this->hr_m[2];
        const double zmax_local = zmin_global + (local_end_z + 1) * this->hr_m[2];

        int ix_start =
            static_cast<int>(std::ceil((xmin_local - xmin_global - 0.5 * dxp) / dxp));
        int ix_end =
            static_cast<int>(std::floor((xmax_local - xmin_global - 0.5 * dxp) / dxp));
        int iy_start =
            static_cast<int>(std::ceil((ymin_local - ymin_global - 0.5 * dyp) / dyp));
        int iy_end =
            static_cast<int>(std::floor((ymax_local - ymin_global - 0.5 * dyp) / dyp));
        int iz_start =
            static_cast<int>(std::ceil((zmin_local - zmin_global - 0.5 * dzp) / dzp));
        int iz_end =
            static_cast<int>(std::floor((zmax_local - zmin_global - 0.5 * dzp) / dzp));

        ix_start = std::max(0, ix_start);
        iy_start = std::max(0, iy_start);
        iz_start = std::max(0, iz_start);
        ix_end   = std::min(static_cast<int>(nxp_global - 1), ix_end);
        iy_end   = std::min(static_cast<int>(nyp_global - 1), iy_end);
        iz_end   = std::min(static_cast<int>(nzp_global - 1), iz_end);

        if (ix_end < ix_start || iy_end < iy_start || iz_end < iz_start) {
            return;
        }

        const unsigned nxp_local = ix_end - ix_start + 1;
        const unsigned nyp_local = iy_end - iy_start + 1;
        const unsigned nzp_local = iz_end - iz_start + 1;
        size_type lattice_local = nxp_local * nyp_local * nzp_local;

        auto pc = this->pcontainer_m;
        const size_type old_nlocal = pc->getLocalNum();
        if (old_nlocal > 0) {
            Kokkos::View<bool*> invalid("vif_3d_remesh_invalid_particles", old_nlocal);
            Kokkos::parallel_for(
                "mark_vif_3d_remesh_particles_invalid",
                old_nlocal,
                KOKKOS_LAMBDA(const size_t p) {
                    invalid(p) = true;
                });
            Kokkos::fence();
            pc->destroy(invalid, old_nlocal);
        }
        pc->create(lattice_local);

        const size_type nlocal = pc->getLocalNum();

        const int nghost = this->fcontainer_m->getOmegaField().getNghost();

        auto R_view       = pc->R.getView();
        auto R_old_view   = pc->R_old.getView();
        auto P_view       = pc->P.getView();
        auto omega_view   = pc->omega.getView();
        auto omega_x_view = pc->omega_x.getView();
        auto omega_y_view = pc->omega_y.getView();
        auto omega_z_view = pc->omega_z.getView();

        auto omega_grid = this->fcontainer_m->getOmegaField().getView();
        auto u_grid     = this->fcontainer_m->getUField().getView();

        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> hr   = this->hr_m;
        const double particle_volume = dxp * dyp * dzp;

        Kokkos::parallel_for(
            "remesh_vif_3d_particles_from_spectral_grid",
            nlocal,
            KOKKOS_LAMBDA(const int p) {
                const unsigned ix_local = p % nxp_local;
                const unsigned iy_local = (p / nxp_local) % nyp_local;
                const unsigned iz_local = p / (nxp_local * nyp_local);

                const unsigned ix_global = ix_start + ix_local;
                const unsigned iy_global = iy_start + iy_local;
                const unsigned iz_global = iz_start + iz_local;

                const double x = xmin_global + (ix_global + 0.5) * dxp;
                const double y = ymin_global + (iy_global + 0.5) * dyp;
                const double z = zmin_global + (iz_global + 0.5) * dzp;

                int grid_i = static_cast<int>(Kokkos::floor((x - rmin[0]) / hr[0]));
                int grid_j = static_cast<int>(Kokkos::floor((y - rmin[1]) / hr[1]));
                int grid_k = static_cast<int>(Kokkos::floor((z - rmin[2]) / hr[2]));

                grid_i = grid_i < local_start_x ? local_start_x : grid_i;
                grid_i = grid_i > local_end_x ? local_end_x : grid_i;
                grid_j = grid_j < local_start_y ? local_start_y : grid_j;
                grid_j = grid_j > local_end_y ? local_end_y : grid_j;
                grid_k = grid_k < local_start_z ? local_start_z : grid_k;
                grid_k = grid_k > local_end_z ? local_end_z : grid_k;

                const int li = grid_i - local_start_x + nghost;
                const int lj = grid_j - local_start_y + nghost;
                const int lk = grid_k - local_start_z + nghost;

                R_view(p)[0] = x;
                R_view(p)[1] = y;
                R_view(p)[2] = z;
                R_old_view(p) = R_view(p);

                const auto omega_value = omega_grid(li, lj, lk) * particle_volume;
                omega_view(p)   = omega_value;
                omega_x_view(p) = omega_value[0];
                omega_y_view(p) = omega_value[1];
                omega_z_view(p) = omega_value[2];
                P_view(p) = u_grid(li, lj, lk);
            });

        Kokkos::fence();
    }

    void computeSpectralParticleVelocity(bool diagnostics) {
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("spectralScatter");
        static IpplTimings::TimerRef grid2parTimer = IpplTimings::getTimer("spectralGather");

        IpplTimings::startTimer(par2gridTimer);
        this->spectralScatter3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_x_hat_m);
            this->Hou_Li_filter(this->omega_y_hat_m);
            this->Hou_Li_filter(this->omega_z_hat_m);
        }
        IpplTimings::stopTimer(par2gridTimer);

        IpplTimings::startTimer(SolveTimer);
        this->computeSpectralVelocityModes3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
            this->Hou_Li_filter(this->uz_hat_m);
        }
        IpplTimings::stopTimer(SolveTimer);

        IpplTimings::startTimer(PTimer);
        if (diagnostics) {
            this->logSpectralDiagnostics3D();
            this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
            this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
            this->logTaylorGreenDiagnostics3D();
        }
        IpplTimings::stopTimer(PTimer);

        IpplTimings::startTimer(grid2parTimer);
        this->spectralGather3D();
        IpplTimings::stopTimer(grid2parTimer);
    }

    void RK4Step() {
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("rk4PushPosition");
        static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        const T dt = this->dt_m;

        pc->rk4_R0 = pc->R;

        computeSpectralParticleVelocity(true);
        pc->rk4_k1 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R = pc->rk4_R0 + (0.5 * dt) * pc->rk4_k1;
        IpplTimings::stopTimer(RTimer);
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeSpectralParticleVelocity(false);
        pc->rk4_k2 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R = pc->rk4_R0 + (0.5 * dt) * pc->rk4_k2;
        IpplTimings::stopTimer(RTimer);
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeSpectralParticleVelocity(false);
        pc->rk4_k3 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R = pc->rk4_R0 + dt * pc->rk4_k3;
        IpplTimings::stopTimer(RTimer);
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeSpectralParticleVelocity(false);
        pc->rk4_k4 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R_old = pc->rk4_R0;
        pc->R = pc->rk4_R0 + (dt / 6.0) *
                              (pc->rk4_k1 + 2.0 * pc->rk4_k2 + 2.0 * pc->rk4_k3 + pc->rk4_k4);
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        if (this->viscosity_m > 0.0) {
            this->spectralScatter3D();
            if (this->useHouLiFilter()) {
                this->Hou_Li_filter(this->omega_x_hat_m);
                this->Hou_Li_filter(this->omega_y_hat_m);
                this->Hou_Li_filter(this->omega_z_hat_m);
            }
            this->viscosity_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_z_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->computeSpectralViscosityModes3D();
            this->spectralGatherViscosity3D();
            this->applyParticleViscosity3D();
        }
    }

    void LeapFrogStep() {
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("spectralScatter");
        static IpplTimings::TimerRef grid2parTimer = IpplTimings::getTimer("spectralGather");
        static IpplTimings::TimerRef gradientGatherTimer =
            IpplTimings::getTimer("spectralGradientGather");
        static IpplTimings::TimerRef stretchingTimer =
            IpplTimings::getTimer("vortexStretching");

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

        IpplTimings::startTimer(par2gridTimer);
        this->spectralScatter3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_x_hat_m);
            this->Hou_Li_filter(this->omega_y_hat_m);
            this->Hou_Li_filter(this->omega_z_hat_m);
        }
        IpplTimings::stopTimer(par2gridTimer);

        IpplTimings::startTimer(SolveTimer);
        this->computeSpectralVelocityModes3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
            this->Hou_Li_filter(this->uz_hat_m);
        }
        this->computeSpectralVelocityGradientModes3D();
        IpplTimings::stopTimer(SolveTimer);

        IpplTimings::startTimer(PTimer);
        this->logSpectralDiagnostics3D();
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        this->logTaylorGreenDiagnostics3D();
        IpplTimings::stopTimer(PTimer);

        IpplTimings::startTimer(gradientGatherTimer);
        this->spectralGatherGradientModes3D();
        IpplTimings::stopTimer(gradientGatherTimer);

        IpplTimings::startTimer(stretchingTimer);
        this->applyParticleVortexStretching3D();
        IpplTimings::stopTimer(stretchingTimer);

        IpplTimings::startTimer(grid2parTimer);
        this->spectralGather3D();
        IpplTimings::stopTimer(grid2parTimer);

        if (this->viscosity_m > 0.0) {
            this->viscosity_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_z_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->computeSpectralViscosityModes3D();
            this->spectralGatherViscosity3D();
            this->applyParticleViscosity3D();
        }

        IpplTimings::startTimer(RTimer);
        if (this->it_m == 0 || bootstrap_next_push_m) {
            pc->R_old = pc->R;
            pc->R = pc->R + pc->P * this->dt_m;
            bootstrap_next_push_m = false;
        } else {
            typename ippl::ParticleBase<
                ippl::ParticleSpatialLayout<T, Dim>>::particle_position_type R_old_temp =
                pc->R_old;

            pc->R_old = pc->R;
            pc->R = R_old_temp + 2 * pc->P * this->dt_m;
        }
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        if (remesh_freq_m > 0 && (this->it_m + 1) % remesh_freq_m == 0) {
            remeshParticles3D();
        }
    }

    void par2grid() override {}

    void grid2par() override {}

    void dump() override {
        static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("vtkDump");
        IpplTimings::startTimer(dumpTimer);

        this->spectralScatter3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_x_hat_m);
            this->Hou_Li_filter(this->omega_y_hat_m);
            this->Hou_Li_filter(this->omega_z_hat_m);
        }
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());

        alvine::vtk::writeVectorField3D("data/VortexInFourier3D", "omega",
                                        this->fcontainer_m->getOmegaField(), this->rmin_m,
                                        this->hr_m, this->it_m);

        IpplTimings::stopTimer(dumpTimer);
    }

    void initializeParticles() {
        auto& FL = this->fcontainer_m->getFL();
        const auto local = FL.getLocalNDIndex();

        const unsigned nxp_global = static_cast<unsigned>(std::round(std::cbrt(this->np_m)));
        const unsigned nyp_global = nxp_global;
        const unsigned nzp_global = nxp_global;

        const double xmin_global = this->rmin_m[0];
        const double ymin_global = this->rmin_m[1];
        const double zmin_global = this->rmin_m[2];
        const double xmax_global = this->rmax_m[0];
        const double ymax_global = this->rmax_m[1];
        const double zmax_global = this->rmax_m[2];

        const double dxp = (xmax_global - xmin_global) / nxp_global;
        const double dyp = (ymax_global - ymin_global) / nyp_global;
        const double dzp = (zmax_global - zmin_global) / nzp_global;

        const double x_low = this->rmin_m[0] + local[0].first() * this->hr_m[0];
        const double x_high = this->rmin_m[0] + (local[0].last() + 1) * this->hr_m[0];
        const double y_low = this->rmin_m[1] + local[1].first() * this->hr_m[1];
        const double y_high = this->rmin_m[1] + (local[1].last() + 1) * this->hr_m[1];
        const double z_low = this->rmin_m[2] + local[2].first() * this->hr_m[2];
        const double z_high = this->rmin_m[2] + (local[2].last() + 1) * this->hr_m[2];

        int ix_start = static_cast<int>(std::ceil((x_low - xmin_global - 0.5 * dxp) / dxp));
        int ix_end = static_cast<int>(std::floor((x_high - xmin_global - 0.5 * dxp) / dxp));
        int iy_start = static_cast<int>(std::ceil((y_low - ymin_global - 0.5 * dyp) / dyp));
        int iy_end = static_cast<int>(std::floor((y_high - ymin_global - 0.5 * dyp) / dyp));
        int iz_start = static_cast<int>(std::ceil((z_low - zmin_global - 0.5 * dzp) / dzp));
        int iz_end = static_cast<int>(std::floor((z_high - zmin_global - 0.5 * dzp) / dzp));

        ix_start = std::max(0, ix_start);
        iy_start = std::max(0, iy_start);
        iz_start = std::max(0, iz_start);
        ix_end = std::min(static_cast<int>(nxp_global - 1), ix_end);
        iy_end = std::min(static_cast<int>(nyp_global - 1), iy_end);
        iz_end = std::min(static_cast<int>(nzp_global - 1), iz_end);

        const unsigned nxp_local = ix_end >= ix_start ? ix_end - ix_start + 1 : 0;
        const unsigned nyp_local = iy_end >= iy_start ? iy_end - iy_start + 1 : 0;
        const unsigned nzp_local = iz_end >= iz_start ? iz_end - iz_start + 1 : 0;
        const size_type nlocal = nxp_local * nyp_local * nzp_local;

        auto pc = this->pcontainer_m;
        pc->create(nlocal);

        auto R = pc->R.getView();
        auto omega = pc->omega.getView();
        const T particleVolume = T(dxp * dyp * dzp);

        Kokkos::parallel_for(
            "init_taylor_green_3d_particles",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                const unsigned ix_local = p % nxp_local;
                const unsigned iy_local = (p / nxp_local) % nyp_local;
                const unsigned iz_local = p / (nxp_local * nyp_local);

                const unsigned ix_global = ix_start + ix_local;
                const unsigned iy_global = iy_start + iy_local;
                const unsigned iz_global = iz_start + iz_local;

                const T x = xmin_global + (ix_global + T(0.5)) * dxp;
                const T y = ymin_global + (iy_global + T(0.5)) * dyp;
                const T z = zmin_global + (iz_global + T(0.5)) * dzp;

                R(p)[0] = x;
                R(p)[1] = y;
                R(p)[2] = z;

                omega(p) = TaylorGreen3D<T>::vorticity(x, y, z) * particleVolume;
            });
        Kokkos::fence();
    }

private:
    unsigned particlesPerDirection3D() const {
        const double n = std::round(std::cbrt(static_cast<double>(this->np_m)));
        return static_cast<unsigned>(n);
    }
};

#endif
