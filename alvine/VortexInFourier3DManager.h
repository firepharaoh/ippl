#ifndef IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H
#define IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
    using ComplexField_t      = typename AlvineManager<T, Dim>::ComplexField_t;

    struct RemeshSpectrumShellMetrics {
        double before = 0.0;
        double after = 0.0;
        std::uint64_t count = 0;
    };

private:
    int remesh_freq_m = 0;
    int diagnostics_freq_m = 1;
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
                             int remesh_freq_ = 0,
                             int diagnostics_freq_ = 1)
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_, dt_, method_,
                                spectral_filter_, viscosity_, time_integrator_)
        , remesh_freq_m(remesh_freq_)
        , diagnostics_freq_m(diagnostics_freq_) {
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

    void dumpRemeshSpectrumDiagnostic3D() {
        reconstructFieldsForRemeshing3D();

        auto omegaXBefore = this->omega_x_hat_m.deepCopy();
        auto omegaYBefore = this->omega_y_hat_m.deepCopy();
        auto omegaZBefore = this->omega_z_hat_m.deepCopy();

        remeshParticlesFromGrid3D();

        writeRemeshSpectrumModeDump3D(omegaXBefore, omegaYBefore, omegaZBefore);
        writeRemeshSpectrumShellDump3D(omegaXBefore, omegaYBefore, omegaZBefore);

        if (ippl::Comm->rank() == 0) {
            Inform m("vif3d_remesh_spectrum ");
            m << "wrote remesh spectrum diagnostics at step = " << this->it_m
              << ", time = " << this->time_m << endl;
        }
    }

    void writeRemeshSpectrumModeDump3D(ComplexField_t& omegaXBefore,
                                       ComplexField_t& omegaYBefore,
                                       ComplexField_t& omegaZBefore) {
        auto oxBefore = omegaXBefore.getHostMirror();
        auto oyBefore = omegaYBefore.getHostMirror();
        auto ozBefore = omegaZBefore.getHostMirror();
        auto oxAfter = this->omega_x_hat_m.getHostMirror();
        auto oyAfter = this->omega_y_hat_m.getHostMirror();
        auto ozAfter = this->omega_z_hat_m.getHostMirror();

        Kokkos::deep_copy(oxBefore, omegaXBefore.getView());
        Kokkos::deep_copy(oyBefore, omegaYBefore.getView());
        Kokkos::deep_copy(ozBefore, omegaZBefore.getView());
        Kokkos::deep_copy(oxAfter, this->omega_x_hat_m.getView());
        Kokkos::deep_copy(oyAfter, this->omega_y_hat_m.getView());
        Kokkos::deep_copy(ozAfter, this->omega_z_hat_m.getView());

        const auto& layout = this->omega_x_hat_m.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = this->omega_x_hat_m.getNghost();

        const int Nx = this->nr_m[0];
        const int Ny = this->nr_m[1];
        const int Nz = this->nr_m[2];

        const T Lx = this->rmax_m[0] - this->rmin_m[0];
        const T Ly = this->rmax_m[1] - this->rmin_m[1];
        const T Lz = this->rmax_m[2] - this->rmin_m[2];
        const T twoPi = T(2.0 * std::acos(-1.0));
        const double amplitudeFloor = 1e-300;

        const std::string filename =
            this->diagnosticFileName("remesh_spectrum_modes_3d_rank_")
            + std::to_string(ippl::Comm->rank()) + ".csv";
        std::ofstream out(filename, std::ios::out);
        out.precision(16);
        out.setf(std::ios::scientific, std::ios::floatfield);
        out << "method,dt,step,time,viscosity,filter,kx,ky,kz,k_mag,"
            << "omega_amp_before,omega_amp_after,amp_ratio\n";

        for (int i = nghost; i < static_cast<int>(oxAfter.extent(0)) - nghost; ++i) {
            for (int j = nghost; j < static_cast<int>(oxAfter.extent(1)) - nghost; ++j) {
                for (int k = nghost; k < static_cast<int>(oxAfter.extent(2)) - nghost; ++k) {
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

                    const auto bx = k2 * oxBefore(i, j, k);
                    const auto by = k2 * oyBefore(i, j, k);
                    const auto bz = k2 * ozBefore(i, j, k);
                    const auto ax = k2 * oxAfter(i, j, k);
                    const auto ay = k2 * oyAfter(i, j, k);
                    const auto az = k2 * ozAfter(i, j, k);

                    const double beforeAmp =
                        bx.real() * bx.real() + bx.imag() * bx.imag()
                        + by.real() * by.real() + by.imag() * by.imag()
                        + bz.real() * bz.real() + bz.imag() * bz.imag();
                    const double afterAmp =
                        ax.real() * ax.real() + ax.imag() * ax.imag()
                        + ay.real() * ay.real() + ay.imag() * ay.imag()
                        + az.real() * az.real() + az.imag() * az.imag();
                    const double ratio =
                        beforeAmp > amplitudeFloor ? afterAmp / beforeAmp : 0.0;
                    const double kmag = std::sqrt(static_cast<double>(mx) * mx
                                                  + static_cast<double>(my) * my
                                                  + static_cast<double>(mz) * mz);

                    out << this->method_m << "," << this->dt_m << ","
                        << this->it_m << "," << this->time_m << ","
                        << this->viscosity_m << "," << this->spectral_filter_m << ","
                        << mx << "," << my << "," << mz << "," << kmag << ","
                        << beforeAmp << "," << afterAmp << "," << ratio << "\n";
                }
            }
        }
    }

    void writeRemeshSpectrumShellDump3D(ComplexField_t& omegaXBefore,
                                        ComplexField_t& omegaYBefore,
                                        ComplexField_t& omegaZBefore) {
        auto localShells = computeRemeshSpectrumShellMetrics3D(
            omegaXBefore, omegaYBefore, omegaZBefore);

        const int shellCount = static_cast<int>(localShells.size());
        std::vector<double> localBefore(shellCount);
        std::vector<double> localAfter(shellCount);
        std::vector<std::uint64_t> localCounts(shellCount);
        std::vector<double> globalBefore(shellCount);
        std::vector<double> globalAfter(shellCount);
        std::vector<std::uint64_t> globalCounts(shellCount);

        for (int shell = 0; shell < shellCount; ++shell) {
            localBefore[shell] = localShells[shell].before;
            localAfter[shell] = localShells[shell].after;
            localCounts[shell] = localShells[shell].count;
        }

        ippl::Comm->allreduce(
            localBefore.data(), globalBefore.data(), shellCount, std::plus<double>());
        ippl::Comm->allreduce(
            localAfter.data(), globalAfter.data(), shellCount, std::plus<double>());
        ippl::Comm->allreduce(
            localCounts.data(), globalCounts.data(), shellCount, std::plus<std::uint64_t>());

        if (ippl::Comm->rank() == 0) {
            std::ofstream out(this->diagnosticFileName("remesh_spectrum_shells_3d.csv"),
                              std::ios::out);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << "method,dt,step,time,viscosity,filter,k_shell,"
                << "enstrophy_before,enstrophy_after,ratio,mode_count\n";

            for (int shell = 0; shell < shellCount; ++shell) {
                const double ratio =
                    globalBefore[shell] > 1e-300
                        ? globalAfter[shell] / globalBefore[shell]
                        : 0.0;
                out << this->method_m << "," << this->dt_m << ","
                    << this->it_m << "," << this->time_m << ","
                    << this->viscosity_m << "," << this->spectral_filter_m << ","
                    << shell << "," << globalBefore[shell] << ","
                    << globalAfter[shell] << "," << ratio << ","
                    << globalCounts[shell] << "\n";
            }
        }
    }

    std::vector<RemeshSpectrumShellMetrics> computeRemeshSpectrumShellMetrics3D(
        ComplexField_t& omegaXBefore,
        ComplexField_t& omegaYBefore,
        ComplexField_t& omegaZBefore) {
        auto oxBefore = omegaXBefore.getView();
        auto oyBefore = omegaYBefore.getView();
        auto ozBefore = omegaZBefore.getView();
        auto oxAfter = this->omega_x_hat_m.getView();
        auto oyAfter = this->omega_y_hat_m.getView();
        auto ozAfter = this->omega_z_hat_m.getView();

        auto& layout = this->omega_x_hat_m.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = this->omega_x_hat_m.getNghost();

        const int Nx = this->nr_m[0];
        const int Ny = this->nr_m[1];
        const int Nz = this->nr_m[2];

        const T Lx = this->rmax_m[0] - this->rmin_m[0];
        const T Ly = this->rmax_m[1] - this->rmin_m[1];
        const T Lz = this->rmax_m[2] - this->rmin_m[2];
        const T volume = Lx * Ly * Lz;
        const T twoPi = T(2.0 * std::acos(-1.0));
        const int maxShell = static_cast<int>(std::floor(std::sqrt(
            static_cast<double>(Nx / 2) * static_cast<double>(Nx / 2)
            + static_cast<double>(Ny / 2) * static_cast<double>(Ny / 2)
            + static_cast<double>(Nz / 2) * static_cast<double>(Nz / 2))));
        const int shellCount = maxShell + 1;

        Kokkos::View<double*> shellBefore("vif3d_remesh_shell_before", shellCount);
        Kokkos::View<double*> shellAfter("vif3d_remesh_shell_after", shellCount);
        Kokkos::View<std::uint64_t*> shellCounts("vif3d_remesh_shell_counts", shellCount);
        Kokkos::deep_copy(shellBefore, 0.0);
        Kokkos::deep_copy(shellAfter, 0.0);
        Kokkos::deep_copy(shellCounts, std::uint64_t(0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "vif3d_remesh_spectrum_shell_metrics",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(oxAfter.extent(0)) - nghost,
                         static_cast<int>(oxAfter.extent(1)) - nghost,
                         static_cast<int>(oxAfter.extent(2)) - nghost}),
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

                const auto bx = k2 * oxBefore(i, j, k);
                const auto by = k2 * oyBefore(i, j, k);
                const auto bz = k2 * ozBefore(i, j, k);
                const auto ax = k2 * oxAfter(i, j, k);
                const auto ay = k2 * oyAfter(i, j, k);
                const auto az = k2 * ozAfter(i, j, k);

                const double beforeAmp =
                    bx.real() * bx.real() + bx.imag() * bx.imag()
                    + by.real() * by.real() + by.imag() * by.imag()
                    + bz.real() * bz.real() + bz.imag() * bz.imag();
                const double afterAmp =
                    ax.real() * ax.real() + ax.imag() * ax.imag()
                    + ay.real() * ay.real() + ay.imag() * ay.imag()
                    + az.real() * az.real() + az.imag() * az.imag();
                const T radius2 = T(mx) * T(mx) + T(my) * T(my) + T(mz) * T(mz);
                const int shell = static_cast<int>(Kokkos::floor(Kokkos::sqrt(radius2)));

                Kokkos::atomic_add(&shellBefore(shell), 0.5 * volume * beforeAmp);
                Kokkos::atomic_add(&shellAfter(shell), 0.5 * volume * afterAmp);
                Kokkos::atomic_add(&shellCounts(shell), std::uint64_t(1));
            });
        Kokkos::fence();

        auto hostBefore = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shellBefore);
        auto hostAfter = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shellAfter);
        auto hostCounts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shellCounts);

        std::vector<RemeshSpectrumShellMetrics> shells(shellCount);
        for (int shell = 0; shell < shellCount; ++shell) {
            shells[shell].before = hostBefore(shell);
            shells[shell].after = hostAfter(shell);
            shells[shell].count = hostCounts(shell);
        }
        return shells;
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

        this->spectralScatter3D(false);
        // The reconstructed grid field was already produced from filtered modes.
        // Applying a spectral filter again immediately after assigning remeshed
        // particles compounds attenuation at every remesh event.
        this->computeSpectralVelocityModes3D();
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
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
        IpplTimings::stopTimer(SolveTimer);

        IpplTimings::startTimer(PTimer);
        if (diagnostics) {
            this->logSpectralDiagnostics3D();
            // this->logSpectralVorticitySpectrum3D();
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
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
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
            // this->logSpectralVorticitySpectrum3D();
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

        computeRK4ParticleRHS(shouldLogDiagnostics3D());
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
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
        this->computeSpectralVelocityGradientModes3D();
        IpplTimings::stopTimer(SolveTimer);

        IpplTimings::startTimer(PTimer);
        if (shouldLogDiagnostics3D()) {
            this->logSpectralDiagnostics3D();
            // this->logSpectralVorticitySpectrum3D();
            this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
            this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
            this->logTaylorGreenDiagnostics3D();
        }
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
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
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
    bool shouldLogDiagnostics3D() const {
        return diagnostics_freq_m > 0
               && static_cast<int>(this->it_m) % diagnostics_freq_m == 0;
    }

    void refreshSpectralVorticityModes3D(const bool consumeRemeshSkip) {
        const bool skipFilter =
            consumeRemeshSkip ? skip_next_rhs_filter_after_remesh_m
                              : (last_remesh_step_m == static_cast<int>(this->it_m));

        this->spectralScatter3D(!skipFilter);
        if (!skipFilter) {
            this->applyConfiguredSpectralFilter3D(this->omega_x_hat_m);
            this->applyConfiguredSpectralFilter3D(this->omega_y_hat_m);
            this->applyConfiguredSpectralFilter3D(this->omega_z_hat_m);
        }

        if (consumeRemeshSkip) {
            skip_next_rhs_filter_after_remesh_m = false;
        }
    }

    unsigned particlesPerDirection3D() const {
        const double n = std::round(std::cbrt(static_cast<double>(this->np_m)));
        return static_cast<unsigned>(n);
    }
};

#endif
