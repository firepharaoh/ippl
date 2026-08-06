#ifndef IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H
#define IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H

#include <algorithm>
#include <cmath>
#include <functional>
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
    bool skip_next_rhs_filter_after_remesh_m = false;
    int last_remesh_step_m = -1;

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
        skip_next_rhs_filter_after_remesh_m = true;
        last_remesh_step_m = static_cast<int>(this->it_m + 1);
    }

    void reconstructFieldsForRemeshing3D() {
        this->spectralScatter3D(false);
        const double omegaNormUnfiltered = this->computeSpectralVorticityModeNorm2Raw3D();

        double omegaNormAfterShape = omegaNormUnfiltered;
        if (this->useShapeFunctionFilter()) {
            this->applyShapeFunctionToSpectralVorticityModes3D();
            omegaNormAfterShape = this->computeSpectralVorticityModeNorm2Raw3D();
        }

        double omegaNormAfterHouLi = omegaNormAfterShape;
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_x_hat_m);
            this->Hou_Li_filter(this->omega_y_hat_m);
            this->Hou_Li_filter(this->omega_z_hat_m);
            omegaNormAfterHouLi = this->computeSpectralVorticityModeNorm2Raw3D();
        }

        const double omegaNormBeforeProjection = omegaNormAfterHouLi;
        this->computeSpectralVelocityModes3D();
        const double omegaNormAfterProjection = this->computeSpectralVorticityModeNorm2Raw3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
            this->Hou_Li_filter(this->uz_hat_m);
        }

        const double energyAfterProjection = this->computeSpectralEnergy3D();
        const double enstrophyAfterProjection = this->computeSpectralEnstrophy3D();

        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());

        const double gridEnstrophy = computeRemeshGridVorticityEnstrophy3D();
        logRemeshSpectralProbe3D(omegaNormUnfiltered, omegaNormAfterShape,
                                 omegaNormAfterHouLi, omegaNormBeforeProjection,
                                 omegaNormAfterProjection, energyAfterProjection,
                                 enstrophyAfterProjection, gridEnstrophy);
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

        logRemeshParticleGridAlignment3D(
            nlocal, nxp_local, nyp_local, ix_start, iy_start, iz_start, dxp, dyp, dzp,
            xmin_global, ymin_global, zmin_global, local_start_x, local_start_y,
            local_start_z, local_end_x, local_end_y, local_end_z);

        this->spectralScatter3D(false);
        const double postRemeshOmegaNormUnfiltered =
            this->computeSpectralVorticityModeNorm2Raw3D();
        double postRemeshOmegaNormAfterShape = postRemeshOmegaNormUnfiltered;
        // The reconstructed grid field was already produced from filtered modes.
        // Applying a spectral filter again immediately after assigning remeshed
        // particles compounds attenuation at every remesh event.
        const double postRemeshOmegaNormBeforeProjection =
            this->computeSpectralVorticityModeNorm2Raw3D();
        this->computeSpectralVelocityModes3D();
        const double postRemeshOmegaNormAfterProjection =
            this->computeSpectralVorticityModeNorm2Raw3D();
        const double postRemeshEnergy = this->computeSpectralEnergy3D();
        const double postRemeshEnstrophy = this->computeSpectralEnstrophy3D();

        logPostRemeshSpectralProbe3D(postRemeshOmegaNormUnfiltered,
                                     postRemeshOmegaNormAfterShape,
                                     postRemeshOmegaNormBeforeProjection,
                                     postRemeshOmegaNormAfterProjection,
                                     postRemeshEnergy, postRemeshEnstrophy);
    }

    double computeRemeshGridVorticityEnstrophy3D() {
        auto omegaGrid = this->fcontainer_m->getOmegaField().getView();
        const int nghost = this->fcontainer_m->getOmegaField().getNghost();
        const double cellVolume = this->hr_m[0] * this->hr_m[1] * this->hr_m[2];

        double localSum = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "compute_remesh_grid_vorticity_enstrophy_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(omegaGrid.extent(0)) - nghost,
                         static_cast<int>(omegaGrid.extent(1)) - nghost,
                         static_cast<int>(omegaGrid.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& lsum) {
                const auto omega = omegaGrid(i, j, k);
                lsum += omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
            },
            localSum);

        double globalSum = 0.0;
        ippl::Comm->allreduce(localSum, globalSum, 1, std::plus<double>());
        return 0.5 * cellVolume * globalSum;
    }

    void logRemeshSpectralProbe3D(const double omegaNormUnfiltered,
                                  const double omegaNormAfterShape,
                                  const double omegaNormAfterHouLi,
                                  const double omegaNormBeforeProjection,
                                  const double omegaNormAfterProjection,
                                  const double energyAfterProjection,
                                  const double enstrophyAfterProjection,
                                  const double gridEnstrophy) const {
        if (ippl::Comm->rank() != 0) {
            return;
        }

        Inform m("vif3d_remesh_probe ");
        m << "step = " << (this->it_m + 1)
          << ", phase = reconstruct"
          << ", omegaNorm2Unfiltered = " << omegaNormUnfiltered
          << ", omegaNorm2AfterShape = " << omegaNormAfterShape
          << ", shapeRelLoss = " << relativeLoss(omegaNormUnfiltered, omegaNormAfterShape)
          << ", omegaNorm2AfterHouLi = " << omegaNormAfterHouLi
          << ", houLiRelLoss = " << relativeLoss(omegaNormAfterShape, omegaNormAfterHouLi)
          << ", omegaNorm2BeforeProjection = " << omegaNormBeforeProjection
          << ", omegaNorm2AfterProjection = " << omegaNormAfterProjection
          << ", projectionRelLoss = "
          << relativeLoss(omegaNormBeforeProjection, omegaNormAfterProjection)
          << ", spectralEnergyAfterProjection = " << energyAfterProjection
          << ", spectralEnstrophyAfterProjection = " << enstrophyAfterProjection
          << ", reconstructedGridEnstrophy = " << gridEnstrophy << endl;
    }

    void logPostRemeshSpectralProbe3D(const double omegaNormUnfiltered,
                                      const double omegaNormAfterShape,
                                      const double omegaNormBeforeProjection,
                                      const double omegaNormAfterProjection,
                                      const double energyAfterProjection,
                                      const double enstrophyAfterProjection) const {
        if (ippl::Comm->rank() != 0) {
            return;
        }

        Inform m("vif3d_remesh_probe ");
        m << "step = " << (this->it_m + 1)
          << ", phase = post_particle_assignment"
          << ", omegaNorm2Unfiltered = " << omegaNormUnfiltered
          << ", omegaNorm2AfterShape = " << omegaNormAfterShape
          << ", shapeRelLoss = " << relativeLoss(omegaNormUnfiltered, omegaNormAfterShape)
          << ", omegaNorm2BeforeProjection = " << omegaNormBeforeProjection
          << ", omegaNorm2AfterProjection = " << omegaNormAfterProjection
          << ", projectionRelLoss = "
          << relativeLoss(omegaNormBeforeProjection, omegaNormAfterProjection)
          << ", spectralEnergyAfterProjection = " << energyAfterProjection
          << ", spectralEnstrophyAfterProjection = " << enstrophyAfterProjection << endl;
    }

    void logRemeshParticleGridAlignment3D(
        const size_type nlocal, const unsigned nxp_local, const unsigned nyp_local,
        const int ix_start, const int iy_start, const int iz_start,
        const double dxp, const double dyp, const double dzp,
        const double xmin_global, const double ymin_global, const double zmin_global,
        const int local_start_x, const int local_start_y, const int local_start_z,
        const int local_end_x, const int local_end_y, const int local_end_z) const {
        const Vector_t<double, Dim> rmin = this->rmin_m;
        const Vector_t<double, Dim> hr = this->hr_m;

        double localMaxCenterError = 0.0;
        double localMismatchCount = 0.0;
        Kokkos::parallel_reduce(
            "measure_remesh_particle_grid_alignment_3d",
            nlocal,
            KOKKOS_LAMBDA(const size_t p, double& maxError, double& mismatchCount) {
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

                const double centerX = rmin[0] + (grid_i + 0.5) * hr[0];
                const double centerY = rmin[1] + (grid_j + 0.5) * hr[1];
                const double centerZ = rmin[2] + (grid_k + 0.5) * hr[2];
                const double err = Kokkos::max(Kokkos::abs(x - centerX),
                                   Kokkos::max(Kokkos::abs(y - centerY),
                                               Kokkos::abs(z - centerZ)));

                maxError = Kokkos::max(maxError, err);
                if (grid_i != static_cast<int>(ix_global)
                    || grid_j != static_cast<int>(iy_global)
                    || grid_k != static_cast<int>(iz_global)) {
                    mismatchCount += 1.0;
                }
            },
            Kokkos::Max<double>(localMaxCenterError),
            Kokkos::Sum<double>(localMismatchCount));

        double globalMaxCenterError = 0.0;
        double globalMismatchCount = 0.0;
        ippl::Comm->allreduce(localMaxCenterError, globalMaxCenterError, 1,
                              std::greater<double>());
        ippl::Comm->allreduce(localMismatchCount, globalMismatchCount, 1,
                              std::plus<double>());

        const double spacingError =
            std::max(std::abs(dxp - hr[0]),
                     std::max(std::abs(dyp - hr[1]), std::abs(dzp - hr[2])));

        if (ippl::Comm->rank() == 0) {
            Inform m("vif3d_remesh_probe ");
            m << "step = " << (this->it_m + 1)
              << ", phase = particle_grid_alignment"
              << ", maxSpacingError = " << spacingError
              << ", maxCenterError = " << globalMaxCenterError
              << ", mismatchedCellCount = " << globalMismatchCount << endl;
        }
    }

    void computeSpectralParticleVelocity(bool diagnostics) {
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("spectralScatter");
        static IpplTimings::TimerRef grid2parTimer = IpplTimings::getTimer("spectralGather");

        IpplTimings::startTimer(par2gridTimer);
        refreshSpectralVorticityModes3D(true);
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

    void computeRK4ParticleRHS(bool diagnostics) {
        static IpplTimings::TimerRef PTimer = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("spectralScatter");
        static IpplTimings::TimerRef grid2parTimer = IpplTimings::getTimer("spectralGather");
        static IpplTimings::TimerRef gradientGatherTimer =
            IpplTimings::getTimer("spectralGradientGather");

        IpplTimings::startTimer(par2gridTimer);
        refreshSpectralVorticityModes3D(true);
        IpplTimings::stopTimer(par2gridTimer);

        IpplTimings::startTimer(SolveTimer);
        this->computeSpectralVelocityModes3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
            this->Hou_Li_filter(this->uz_hat_m);
        }
        this->computeSpectralVelocityGradientModes3D();

        if (this->viscosity_m > 0.0) {
            this->viscosity_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_z_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->computeSpectralViscosityModes3D();
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

        IpplTimings::startTimer(gradientGatherTimer);
        this->spectralGatherGradientModes3D();
        if (this->viscosity_m > 0.0) {
            this->spectralGatherViscosity3D();
        }
        IpplTimings::stopTimer(gradientGatherTimer);

        IpplTimings::startTimer(grid2parTimer);
        this->spectralGather3D();
        IpplTimings::stopTimer(grid2parTimer);
    }

    void storeRK4OmegaRHS(typename ParticleContainer_t::particle_position_type& target) {
        static IpplTimings::TimerRef stretchingTimer =
            IpplTimings::getTimer("vortexStretching");

        auto& pc = *this->pcontainer_m;
        auto omega = pc.omega.getView();
        auto duxdx = pc.duxdx.getView();
        auto duxdy = pc.duxdy.getView();
        auto duxdz = pc.duxdz.getView();
        auto duydx = pc.duydx.getView();
        auto duydy = pc.duydy.getView();
        auto duydz = pc.duydz.getView();
        auto duzdx = pc.duzdx.getView();
        auto duzdy = pc.duzdy.getView();
        auto duzdz = pc.duzdz.getView();
        auto viscosity = pc.viscosity.getView();
        auto rhs = target.getView();
        const auto n = pc.getLocalNum();

        const unsigned nxp = static_cast<unsigned>(
            std::round(std::cbrt(static_cast<double>(this->np_m))));
        const T dxp = T(this->rmax_m[0] - this->rmin_m[0]) / nxp;
        const T dyp = T(this->rmax_m[1] - this->rmin_m[1]) / nxp;
        const T dzp = T(this->rmax_m[2] - this->rmin_m[2]) / nxp;
        const T particleVolume = dxp * dyp * dzp;
        const bool useViscosity = this->viscosity_m > 0.0;

        IpplTimings::startTimer(stretchingTimer);
        Kokkos::parallel_for(
            "store_rk4_vorticity_rhs_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                const T omegaX = omega(p)[0];
                const T omegaY = omega(p)[1];
                const T omegaZ = omega(p)[2];

                rhs(p)[0] = omegaX * duxdx(p) + omegaY * duxdy(p) + omegaZ * duxdz(p);
                rhs(p)[1] = omegaX * duydx(p) + omegaY * duydy(p) + omegaZ * duydz(p);
                rhs(p)[2] = omegaX * duzdx(p) + omegaY * duzdy(p) + omegaZ * duzdz(p);

                if (useViscosity) {
                    rhs(p)[0] += viscosity(p)[0] * particleVolume;
                    rhs(p)[1] += viscosity(p)[1] * particleVolume;
                    rhs(p)[2] += viscosity(p)[2] * particleVolume;
                }
            });
        Kokkos::fence();
        IpplTimings::stopTimer(stretchingTimer);
    }

    void setRK4StageState(typename ParticleContainer_t::particle_position_type& baseR,
                          typename ParticleContainer_t::particle_position_type& baseOmega,
                          typename ParticleContainer_t::particle_position_type& kR,
                          typename ParticleContainer_t::particle_position_type& kOmega,
                          const T scale) {
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("rk4PushPosition");

        auto& pc = *this->pcontainer_m;
        auto R = pc.R.getView();
        auto omega = pc.omega.getView();
        auto omegaX = pc.omega_x.getView();
        auto omegaY = pc.omega_y.getView();
        auto omegaZ = pc.omega_z.getView();
        auto baseRView = baseR.getView();
        auto baseOmegaView = baseOmega.getView();
        auto kRView = kR.getView();
        auto kOmegaView = kOmega.getView();
        const auto n = pc.getLocalNum();
        const T dtScale = scale * T(this->dt_m);

        IpplTimings::startTimer(RTimer);
        Kokkos::parallel_for(
            "set_rk4_stage_state_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                for (unsigned d = 0; d < Dim; ++d) {
                    R(p)[d] = baseRView(p)[d] + dtScale * kRView(p)[d];
                    omega(p)[d] = baseOmegaView(p)[d] + dtScale * kOmegaView(p)[d];
                }
                omegaX(p) = omega(p)[0];
                omegaY(p) = omega(p)[1];
                omegaZ(p) = omega(p)[2];
            });
        Kokkos::fence();
        IpplTimings::stopTimer(RTimer);
    }

    void finalizeRK4State() {
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("rk4PushPosition");

        auto& pc = *this->pcontainer_m;
        auto R = pc.R.getView();
        auto Rold = pc.R_old.getView();
        auto omega = pc.omega.getView();
        auto omegaX = pc.omega_x.getView();
        auto omegaY = pc.omega_y.getView();
        auto omegaZ = pc.omega_z.getView();
        auto R0 = pc.rk4_R0.getView();
        auto omega0 = pc.rk4_omega0.getView();
        auto kR1 = pc.rk4_k1.getView();
        auto kR2 = pc.rk4_k2.getView();
        auto kR3 = pc.rk4_k3.getView();
        auto kR4 = pc.rk4_k4.getView();
        auto kOmega1 = pc.rk4_omega_k1.getView();
        auto kOmega2 = pc.rk4_omega_k2.getView();
        auto kOmega3 = pc.rk4_omega_k3.getView();
        auto kOmega4 = pc.rk4_omega_k4.getView();
        const auto n = pc.getLocalNum();
        const T sixthDt = T(this->dt_m) / T(6.0);

        IpplTimings::startTimer(RTimer);
        Kokkos::parallel_for(
            "finalize_rk4_state_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                Rold(p) = R0(p);
                for (unsigned d = 0; d < Dim; ++d) {
                    R(p)[d] = R0(p)[d] + sixthDt *
                        (kR1(p)[d] + T(2.0) * kR2(p)[d] +
                         T(2.0) * kR3(p)[d] + kR4(p)[d]);
                    omega(p)[d] = omega0(p)[d] + sixthDt *
                        (kOmega1(p)[d] + T(2.0) * kOmega2(p)[d] +
                         T(2.0) * kOmega3(p)[d] + kOmega4(p)[d]);
                }
                omegaX(p) = omega(p)[0];
                omegaY(p) = omega(p)[1];
                omegaZ(p) = omega(p)[2];
            });
        Kokkos::fence();
        IpplTimings::stopTimer(RTimer);
    }

    void RK4Step() {
        static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

        pc->rk4_R0 = pc->R;
        pc->rk4_omega0 = pc->omega;

        computeRK4ParticleRHS(true);
        pc->rk4_k1 = pc->P;
        storeRK4OmegaRHS(pc->rk4_omega_k1);

        setRK4StageState(pc->rk4_R0, pc->rk4_omega0, pc->rk4_k1, pc->rk4_omega_k1,
                         T(0.5));
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeRK4ParticleRHS(false);
        pc->rk4_k2 = pc->P;
        storeRK4OmegaRHS(pc->rk4_omega_k2);

        setRK4StageState(pc->rk4_R0, pc->rk4_omega0, pc->rk4_k2, pc->rk4_omega_k2,
                         T(0.5));
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeRK4ParticleRHS(false);
        pc->rk4_k3 = pc->P;
        storeRK4OmegaRHS(pc->rk4_omega_k3);

        setRK4StageState(pc->rk4_R0, pc->rk4_omega0, pc->rk4_k3, pc->rk4_omega_k3,
                         T(1.0));
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeRK4ParticleRHS(false);
        pc->rk4_k4 = pc->P;
        storeRK4OmegaRHS(pc->rk4_omega_k4);

        finalizeRK4State();

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        if (remesh_freq_m > 0 && (this->it_m + 1) % remesh_freq_m == 0) {
            remeshParticles3D();
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
        refreshSpectralVorticityModes3D(true);
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

        refreshSpectralVorticityModes3D(false);
        this->computeSpectralVelocityModes3D();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
            this->Hou_Li_filter(this->uz_hat_m);
        }
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());

        alvine::vtk::writeVectorField3D("data/VortexInFourier3D", "omega",
                                        this->fcontainer_m->getOmegaField(), this->rmin_m,
                                        this->hr_m, this->it_m);
        alvine::vtk::writeVectorField3D("data/VortexInFourier3D", "velocity",
                                        this->fcontainer_m->getUField(), this->rmin_m,
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
    void refreshSpectralVorticityModes3D(const bool consumeRemeshSkip) {
        const bool skipFilter =
            consumeRemeshSkip ? skip_next_rhs_filter_after_remesh_m
                              : (last_remesh_step_m == static_cast<int>(this->it_m));

        this->spectralScatter3D(!skipFilter);
        if (!skipFilter && this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_x_hat_m);
            this->Hou_Li_filter(this->omega_y_hat_m);
            this->Hou_Li_filter(this->omega_z_hat_m);
        }

        if (consumeRemeshSkip) {
            skip_next_rhs_filter_after_remesh_m = false;
        }
    }

    static double relativeLoss(const double before, const double after) {
        return (before - after) / std::max(std::abs(before), 1e-300);
    }

    unsigned particlesPerDirection3D() const {
        const double n = std::round(std::cbrt(static_cast<double>(this->np_m)));
        return static_cast<unsigned>(n);
    }
};

#endif
