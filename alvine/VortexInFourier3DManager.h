#ifndef IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H
#define IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    using VectorField_t       = ::VField_t<T, Dim>;

    struct RHSConsistencyMetrics {
        double particleL2 = 0.0;
        double referenceL2 = 0.0;
        double errorL2 = 0.0;
        double relativeL2 = 0.0;
        double projectionScale = 0.0;
    };

    struct SpectrumShellMetrics {
        double enstrophy = 0.0;
        std::uint64_t count = 0;
    };

private:
    int remesh_freq_m = 0;
    int diagnostics_freq_m = 1;
    bool bootstrap_next_push_m = false;
    bool skip_next_rhs_filter_after_remesh_m = false;
    bool rhs_consistency_done_m = false;
    double rhs_consistency_time_m = -1.0;
    bool spectrum_dump_m = false;
    bool spectrum_shells_initialized_m = false;
    std::vector<double> spectrum_previous_shells_m;
    int last_remesh_step_m = -1;
    bool pipeline_trace_m = false;
    int pipeline_trace_freq_m = 1;

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
        if (spectrum_dump_m) {
            writeStepSpectrumDump3D();
        }
    }

    void pre_step() override {
        maybeRunRHSConsistencyDiagnostic3D();
    }

    void post_step() override {
        AlvineManager<T, Dim>::post_step();
        maybeRunRHSConsistencyDiagnostic3D();
        if (spectrum_dump_m) {
            writeStepSpectrumDump3D();
        }
    }

    void maybeRunRHSConsistencyDiagnostic3D() {
        if (!rhs_consistency_done_m
            && rhs_consistency_time_m >= 0.0
            && this->time_m + 1e-12 >= rhs_consistency_time_m) {
            runRHSConsistencyDiagnostic3D();
            rhs_consistency_done_m = true;
        }
    }

    void advance() override {
        if (this->useRK4()) {
            RK4Step();
        } else {
            LeapFrogStep();
        }
    }

    void setRHSConsistencyTime(const double time) {
        rhs_consistency_time_m = time;
        rhs_consistency_done_m = false;
    }

    void setSpectrumDump(const bool enabled) {
        spectrum_dump_m = enabled;
    }

    void setPipelineTrace(const bool enabled, const int frequency) {
        pipeline_trace_m = enabled;
        pipeline_trace_freq_m = std::max(1, frequency);
    }

    void writeStepSpectrumDump3D() {
        this->spectralScatter3D(false);
        this->computeSpectralVelocityModes3D();
        writeStepSpectrumShellDump3D();
    }

    void writeStepSpectrumShellDump3D() {
        auto localShells = computeCurrentVorticitySpectrumShellMetrics3D();

        const int shellCount = static_cast<int>(localShells.size());
        std::vector<double> localEnstrophy(shellCount);
        std::vector<std::uint64_t> localCounts(shellCount);
        std::vector<double> globalEnstrophy(shellCount);
        std::vector<std::uint64_t> globalCounts(shellCount);

        for (int shell = 0; shell < shellCount; ++shell) {
            localEnstrophy[shell] = localShells[shell].enstrophy;
            localCounts[shell] = localShells[shell].count;
        }

        ippl::Comm->allreduce(
            localEnstrophy.data(), globalEnstrophy.data(), shellCount, std::plus<double>());
        ippl::Comm->allreduce(
            localCounts.data(), globalCounts.data(), shellCount, std::plus<std::uint64_t>());

        if (ippl::Comm->rank() == 0) {
            std::ofstream out(this->diagnosticFileName("remesh_spectrum_shells_3d.csv"),
                              spectrum_shells_initialized_m ? std::ios::app : std::ios::out);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            if (!spectrum_shells_initialized_m) {
                out << "method,dt,step,time,viscosity,filter,is_remesh_step,k_shell,"
                    << "enstrophy,ratio_to_previous_step,mode_count\n";
            }

            const bool isRemeshStep =
                this->it_m > 0
                && remesh_freq_m > 0
                && static_cast<int>(this->it_m) % remesh_freq_m == 0;
            for (int shell = 0; shell < shellCount; ++shell) {
                double ratio = 0.0;
                if (spectrum_shells_initialized_m
                    && shell < static_cast<int>(spectrum_previous_shells_m.size())
                    && spectrum_previous_shells_m[shell] > 1e-300) {
                    ratio = globalEnstrophy[shell] / spectrum_previous_shells_m[shell];
                }
                out << this->method_m << "," << this->dt_m << ","
                    << this->it_m << "," << this->time_m << ","
                    << this->viscosity_m << "," << this->spectral_filter_m << ","
                    << (isRemeshStep ? 1 : 0) << ","
                    << shell << "," << globalEnstrophy[shell] << "," << ratio << ","
                    << globalCounts[shell] << "\n";
            }

            spectrum_previous_shells_m = globalEnstrophy;
        }
        spectrum_shells_initialized_m = true;
    }

    std::vector<SpectrumShellMetrics> computeCurrentVorticitySpectrumShellMetrics3D() {
        auto ox = this->omega_x_hat_m.getView();
        auto oy = this->omega_y_hat_m.getView();
        auto oz = this->omega_z_hat_m.getView();

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

        Kokkos::View<double*> shellEnstrophy("vif3d_step_shell_enstrophy", shellCount);
        Kokkos::View<std::uint64_t*> shellCounts("vif3d_step_shell_counts", shellCount);
        Kokkos::deep_copy(shellEnstrophy, 0.0);
        Kokkos::deep_copy(shellCounts, std::uint64_t(0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "vif3d_spectrum_shell_metrics",
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

                const auto wx = k2 * ox(i, j, k);
                const auto wy = k2 * oy(i, j, k);
                const auto wz = k2 * oz(i, j, k);

                const double omegaAmp =
                    wx.real() * wx.real() + wx.imag() * wx.imag()
                    + wy.real() * wy.real() + wy.imag() * wy.imag()
                    + wz.real() * wz.real() + wz.imag() * wz.imag();
                const T radius2 = T(mx) * T(mx) + T(my) * T(my) + T(mz) * T(mz);
                const int shell = static_cast<int>(Kokkos::floor(Kokkos::sqrt(radius2)));

                Kokkos::atomic_add(&shellEnstrophy(shell), 0.5 * volume * omegaAmp);
                Kokkos::atomic_add(&shellCounts(shell), std::uint64_t(1));
            });
        Kokkos::fence();

        auto hostEnstrophy =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shellEnstrophy);
        auto hostCounts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shellCounts);

        std::vector<SpectrumShellMetrics> shells(shellCount);
        for (int shell = 0; shell < shellCount; ++shell) {
            shells[shell].enstrophy = hostEnstrophy(shell);
            shells[shell].count = hostCounts(shell);
        }
        return shells;
    }

    void runRHSConsistencyDiagnostic3D(
        const std::string& filename = "rhs_consistency_3d.csv") {
        refreshSpectralVorticityModes3D(false);
        this->computeSpectralVelocityModes3D();
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
        this->computeSpectralVelocityGradientModes3D();

        this->viscosity_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
        this->viscosity_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
        this->viscosity_z_hat_m = Kokkos::complex<T>(0.0, 0.0);
        if (this->viscosity_m > 0.0) {
            this->computeSpectralViscosityModes3D();
        }

        auto omegaXState = this->omega_x_hat_m.deepCopy();
        auto omegaYState = this->omega_y_hat_m.deepCopy();
        auto omegaZState = this->omega_z_hat_m.deepCopy();
        auto uxState = this->ux_hat_m.deepCopy();
        auto uyState = this->uy_hat_m.deepCopy();
        auto uzState = this->uz_hat_m.deepCopy();
        auto viscXState = this->viscosity_x_hat_m.deepCopy();
        auto viscYState = this->viscosity_y_hat_m.deepCopy();
        auto viscZState = this->viscosity_z_hat_m.deepCopy();

        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());

        auto sReference = this->fcontainer_m->getOmegaField().deepCopy();
        auto vReference = this->fcontainer_m->getOmegaField().deepCopy();
        computeGridRHSReference3D(sReference, vReference);

        this->spectralGatherGradientModes3D();
        if (this->viscosity_m > 0.0) {
            this->spectralGatherViscosity3D();
        }

        auto sParticle = this->fcontainer_m->getOmegaField().deepCopy();
        auto vParticle = this->fcontainer_m->getOmegaField().deepCopy();
        computeParticleRHSField3D(sParticle, true, false);
        computeParticleRHSField3D(vParticle, false, true);

        auto totalReference = sReference.deepCopy();
        auto totalParticle = sParticle.deepCopy();
        addRHSFields3D(totalReference, sReference, vReference);
        addRHSFields3D(totalParticle, sParticle, vParticle);

        const RHSConsistencyMetrics sMetrics =
            computeRHSFieldMetrics3D(sParticle, sReference);
        const RHSConsistencyMetrics vMetrics =
            computeRHSFieldMetrics3D(vParticle, vReference);
        const RHSConsistencyMetrics totalMetrics =
            computeRHSFieldMetrics3D(totalParticle, totalReference);

        Kokkos::deep_copy(this->omega_x_hat_m.getView(), omegaXState.getView());
        Kokkos::deep_copy(this->omega_y_hat_m.getView(), omegaYState.getView());
        Kokkos::deep_copy(this->omega_z_hat_m.getView(), omegaZState.getView());
        Kokkos::deep_copy(this->ux_hat_m.getView(), uxState.getView());
        Kokkos::deep_copy(this->uy_hat_m.getView(), uyState.getView());
        Kokkos::deep_copy(this->uz_hat_m.getView(), uzState.getView());
        Kokkos::deep_copy(this->viscosity_x_hat_m.getView(), viscXState.getView());
        Kokkos::deep_copy(this->viscosity_y_hat_m.getView(), viscYState.getView());
        Kokkos::deep_copy(this->viscosity_z_hat_m.getView(), viscZState.getView());

        if (ippl::Comm->rank() == 0) {
            std::ofstream out(this->diagnosticFileName(filename), std::ios::out);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << "method,dt,step,time,target_time,viscosity,filter,term,"
                << "particle_l2,reference_l2,error_l2,relative_l2,projection_scale\n";
            writeRHSConsistencyRow3D(out, "stretching", sMetrics);
            writeRHSConsistencyRow3D(out, "viscosity", vMetrics);
            writeRHSConsistencyRow3D(out, "total", totalMetrics);

            Inform m("rhs_consistency_3d ");
            m << "step = " << this->it_m
              << ", time = " << this->time_m
              << ", stretchingRelL2 = " << sMetrics.relativeL2
              << ", viscosityRelL2 = " << vMetrics.relativeL2
              << ", totalRelL2 = " << totalMetrics.relativeL2 << endl;
        }
    }

    void writeRHSConsistencyRow3D(std::ofstream& out,
                                  const std::string& term,
                                  const RHSConsistencyMetrics& metrics) {
        out << this->method_m << "," << this->dt_m << ","
            << this->it_m << "," << this->time_m << ","
            << rhs_consistency_time_m << "," << this->viscosity_m << ","
            << this->spectral_filter_m << "," << term << ","
            << metrics.particleL2 << "," << metrics.referenceL2 << ","
            << metrics.errorL2 << "," << metrics.relativeL2 << ","
            << metrics.projectionScale << "\n";
    }

    void inverseCellCenteredModes3D(ComplexField_t& modes) {
        this->applyCellCenteredIfftPhase3D(modes);
        this->spectralFft_mp->transform(ippl::BACKWARD, modes);
    }

    void computeGridRHSReference3D(VectorField_t& sReference,
                                   VectorField_t& vReference) {
        auto duxdxModes = this->duxdx_hat_m.deepCopy();
        auto duxdyModes = this->duxdy_hat_m.deepCopy();
        auto duxdzModes = this->duxdz_hat_m.deepCopy();
        auto duydxModes = this->duydx_hat_m.deepCopy();
        auto duydyModes = this->duydy_hat_m.deepCopy();
        auto duydzModes = this->duydz_hat_m.deepCopy();
        auto duzdxModes = this->duzdx_hat_m.deepCopy();
        auto duzdyModes = this->duzdy_hat_m.deepCopy();
        auto duzdzModes = this->duzdz_hat_m.deepCopy();
        auto viscXModes = this->viscosity_x_hat_m.deepCopy();
        auto viscYModes = this->viscosity_y_hat_m.deepCopy();
        auto viscZModes = this->viscosity_z_hat_m.deepCopy();

        inverseCellCenteredModes3D(duxdxModes);
        inverseCellCenteredModes3D(duxdyModes);
        inverseCellCenteredModes3D(duxdzModes);
        inverseCellCenteredModes3D(duydxModes);
        inverseCellCenteredModes3D(duydyModes);
        inverseCellCenteredModes3D(duydzModes);
        inverseCellCenteredModes3D(duzdxModes);
        inverseCellCenteredModes3D(duzdyModes);
        inverseCellCenteredModes3D(duzdzModes);
        inverseCellCenteredModes3D(viscXModes);
        inverseCellCenteredModes3D(viscYModes);
        inverseCellCenteredModes3D(viscZModes);

        auto omega = this->fcontainer_m->getOmegaField().getView();
        auto sRef = sReference.getView();
        auto vRef = vReference.getView();
        auto duxdx = duxdxModes.getView();
        auto duxdy = duxdyModes.getView();
        auto duxdz = duxdzModes.getView();
        auto duydx = duydxModes.getView();
        auto duydy = duydyModes.getView();
        auto duydz = duydzModes.getView();
        auto duzdx = duzdxModes.getView();
        auto duzdy = duzdyModes.getView();
        auto duzdz = duzdzModes.getView();
        auto viscX = viscXModes.getView();
        auto viscY = viscYModes.getView();
        auto viscZ = viscZModes.getView();
        const int nghost = sReference.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "compute_grid_rhs_reference_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(sRef.extent(0)) - nghost,
                         static_cast<int>(sRef.extent(1)) - nghost,
                         static_cast<int>(sRef.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const T omegaX = omega(i, j, k)[0];
                const T omegaY = omega(i, j, k)[1];
                const T omegaZ = omega(i, j, k)[2];

                sRef(i, j, k)[0] =
                    omegaX * duxdx(i, j, k).real()
                    + omegaY * duxdy(i, j, k).real()
                    + omegaZ * duxdz(i, j, k).real();
                sRef(i, j, k)[1] =
                    omegaX * duydx(i, j, k).real()
                    + omegaY * duydy(i, j, k).real()
                    + omegaZ * duydz(i, j, k).real();
                sRef(i, j, k)[2] =
                    omegaX * duzdx(i, j, k).real()
                    + omegaY * duzdy(i, j, k).real()
                    + omegaZ * duzdz(i, j, k).real();

                vRef(i, j, k)[0] = viscX(i, j, k).real();
                vRef(i, j, k)[1] = viscY(i, j, k).real();
                vRef(i, j, k)[2] = viscZ(i, j, k).real();
            });
        Kokkos::fence();
    }

    void computeParticleRHSField3D(VectorField_t& rhsField,
                                   const bool includeStretching,
                                   const bool includeViscosity) {
        auto& pc = *this->pcontainer_m;
        const auto n = pc.getLocalNum();
        auto omega = pc.omega.getView();
        auto omegaX = pc.omega_x.getView();
        auto omegaY = pc.omega_y.getView();
        auto omegaZ = pc.omega_z.getView();
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

        Kokkos::View<Vector_t<T, Dim>*> omegaBackup("rhs_consistency_omega_backup", n);
        const bool useViscosity = includeViscosity && this->viscosity_m > 0.0;
        const unsigned nxp = particlesPerDirection3D();
        const T particleVolume =
            T(this->rmax_m[0] - this->rmin_m[0])
            * T(this->rmax_m[1] - this->rmin_m[1])
            * T(this->rmax_m[2] - this->rmin_m[2])
            / T(nxp * nxp * nxp);

        Kokkos::parallel_for(
            "pack_particle_rhs_for_scatter_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                omegaBackup(p) = omega(p);

                Vector_t<T, Dim> rhs(0.0);
                if (includeStretching) {
                    const T omegaPX = omega(p)[0];
                    const T omegaPY = omega(p)[1];
                    const T omegaPZ = omega(p)[2];
                    rhs[0] += omegaPX * duxdx(p) + omegaPY * duxdy(p) + omegaPZ * duxdz(p);
                    rhs[1] += omegaPX * duydx(p) + omegaPY * duydy(p) + omegaPZ * duydz(p);
                    rhs[2] += omegaPX * duzdx(p) + omegaPY * duzdy(p) + omegaPZ * duzdz(p);
                }
                if (useViscosity) {
                    rhs[0] += viscosity(p)[0] * particleVolume;
                    rhs[1] += viscosity(p)[1] * particleVolume;
                    rhs[2] += viscosity(p)[2] * particleVolume;
                }

                omega(p) = rhs;
                omegaX(p) = rhs[0];
                omegaY(p) = rhs[1];
                omegaZ(p) = rhs[2];
            });
        Kokkos::fence();

        this->spectralScatter3D(false);
        this->reconstructSpectralVorticity(rhsField);

        Kokkos::parallel_for(
            "restore_particle_omega_after_rhs_consistency_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                omega(p) = omegaBackup(p);
                omegaX(p) = omegaBackup(p)[0];
                omegaY(p) = omegaBackup(p)[1];
                omegaZ(p) = omegaBackup(p)[2];
            });
        Kokkos::fence();
    }

    void addRHSFields3D(VectorField_t& outField,
                        VectorField_t& lhsField,
                        VectorField_t& rhsField) {
        auto out = outField.getView();
        auto lhs = lhsField.getView();
        auto rhs = rhsField.getView();
        const int nghost = outField.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "add_rhs_fields_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(out.extent(0)) - nghost,
                         static_cast<int>(out.extent(1)) - nghost,
                         static_cast<int>(out.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                for (unsigned d = 0; d < Dim; ++d) {
                    out(i, j, k)[d] = lhs(i, j, k)[d] + rhs(i, j, k)[d];
                }
            });
        Kokkos::fence();
    }

    RHSConsistencyMetrics computeRHSFieldMetrics3D(VectorField_t& particleField,
                                                   VectorField_t& referenceField) {
        auto particle = particleField.getView();
        auto reference = referenceField.getView();
        const int nghost = particleField.getNghost();
        const T cellVolume = this->hr_m[0] * this->hr_m[1] * this->hr_m[2];

        double localParticle2 = 0.0;
        double localReference2 = 0.0;
        double localError2 = 0.0;
        double localDot = 0.0;

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "compute_rhs_consistency_metrics_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(particle.extent(0)) - nghost,
                         static_cast<int>(particle.extent(1)) - nghost,
                         static_cast<int>(particle.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& particle2,
                          double& reference2,
                          double& error2,
                          double& dot) {
                for (unsigned d = 0; d < Dim; ++d) {
                    const double p = particle(i, j, k)[d];
                    const double r = reference(i, j, k)[d];
                    const double e = p - r;
                    particle2 += cellVolume * p * p;
                    reference2 += cellVolume * r * r;
                    error2 += cellVolume * e * e;
                    dot += cellVolume * p * r;
                }
            },
            Kokkos::Sum<double>(localParticle2),
            Kokkos::Sum<double>(localReference2),
            Kokkos::Sum<double>(localError2),
            Kokkos::Sum<double>(localDot));

        double globalParticle2 = 0.0;
        double globalReference2 = 0.0;
        double globalError2 = 0.0;
        double globalDot = 0.0;
        ippl::Comm->allreduce(localParticle2, globalParticle2, 1, std::plus<double>());
        ippl::Comm->allreduce(localReference2, globalReference2, 1, std::plus<double>());
        ippl::Comm->allreduce(localError2, globalError2, 1, std::plus<double>());
        ippl::Comm->allreduce(localDot, globalDot, 1, std::plus<double>());

        RHSConsistencyMetrics metrics;
        metrics.particleL2 = std::sqrt(std::max(globalParticle2, 0.0));
        metrics.referenceL2 = std::sqrt(std::max(globalReference2, 0.0));
        metrics.errorL2 = std::sqrt(std::max(globalError2, 0.0));
        metrics.relativeL2 =
            metrics.errorL2 / std::max(metrics.referenceL2, 1e-300);
        metrics.projectionScale =
            globalDot / std::max(globalReference2, 1e-300);
        return metrics;
    }

    void remeshParticles3D() {
        tracePipelineState3D("BEFORE REMESH");
        reconstructFieldsForRemeshing3D();
        remeshParticlesFromGrid3D();
        bootstrap_next_push_m = true;
        skip_next_rhs_filter_after_remesh_m = true;
        last_remesh_step_m = static_cast<int>(this->it_m + 1);
        tracePipelineState3D("AFTER REMESH");
    }

    void reconstructFieldsForRemeshing3D() {
        tracePipelineState3D("REMESH RECONSTRUCT BEFORE SCATTER");
        tracedSpectralScatter3D(false);
        tracePipelineState3D("REMESH RECONSTRUCT AFTER SCATTER");

        if (this->useShapeFunctionFilter()) {
            this->applyShapeFunctionToSpectralVorticityModes3D();
            tracePipelineState3D("REMESH RECONSTRUCT AFTER SHAPE");
        }

        this->applyConfiguredSpectralFilter3D(this->omega_x_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_y_hat_m);
        this->applyConfiguredSpectralFilter3D(this->omega_z_hat_m);
        tracePipelineState3D("REMESH RECONSTRUCT AFTER VORTICITY FILTER");

        this->computeSpectralVelocityModes3D();
        tracePipelineState3D("REMESH RECONSTRUCT AFTER BIOT SAVART");
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
        tracePipelineState3D("REMESH RECONSTRUCT AFTER VELOCITY FILTER");

        // Remeshing samples these spectral modes directly with type-2 NUFFT.
        // Keep the IFFT reconstruction path out of this diagnostic/remesh path.
        // this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        // this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
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

        auto R_view       = pc->R.getView();
        auto R_old_view   = pc->R_old.getView();

        const double particle_volume = dxp * dyp * dzp;

        Kokkos::parallel_for(
            "place_vif_3d_remesh_lattice_particles",
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

                R_view(p)[0] = x;
                R_view(p)[1] = y;
                R_view(p)[2] = z;
                R_old_view(p) = R_view(p);
            });

        Kokkos::fence();

        sampleRemeshedParticlesFromSpectralModes3D(particle_volume);

        tracePipelineState3D("REMESH AFTER TYPE2 SAMPLING BEFORE SCATTER");
        tracedSpectralScatter3D(false);
        tracePipelineState3D("REMESH AFTER TYPE2 SAMPLING AFTER SCATTER");
        // The remeshed particles were already sampled from filtered modes.
        // Applying a spectral filter again immediately after assigning them
        // compounds attenuation at every remesh event.
        this->computeSpectralVelocityModes3D();
        tracePipelineState3D("REMESH AFTER TYPE2 SAMPLING AFTER BIOT SAVART");
    }

    void sampleRemeshedParticlesFromSpectralModes3D(const double particle_volume) {
        if (!this->nufftType2_mp) {
            throw std::runtime_error(
                "VIF 3D remeshing requires type-2 NUFFT to sample spectral modes.");
        }

        auto pc = this->pcontainer_m;

        pc->omega_x = 0.0;
        pc->omega_y = 0.0;
        pc->omega_z = 0.0;
        pc->ux = 0.0;
        pc->uy = 0.0;
        pc->uz = 0.0;

        auto oxModes = this->omega_x_hat_m.deepCopy();
        auto oyModes = this->omega_y_hat_m.deepCopy();
        auto ozModes = this->omega_z_hat_m.deepCopy();

        recoverFourierSeriesVorticityModesForSampling3D(oxModes, oyModes, ozModes);

        this->nufftType2_mp->transform(pc->R, pc->omega_x, oxModes);
        this->nufftType2_mp->transform(pc->R, pc->omega_y, oyModes);
        this->nufftType2_mp->transform(pc->R, pc->omega_z, ozModes);

        auto uxModes = this->ux_hat_m.deepCopy();
        auto uyModes = this->uy_hat_m.deepCopy();
        auto uzModes = this->uz_hat_m.deepCopy();

        this->nufftType2_mp->transform(pc->R, pc->ux, uxModes);
        this->nufftType2_mp->transform(pc->R, pc->uy, uyModes);
        this->nufftType2_mp->transform(pc->R, pc->uz, uzModes);

        auto omega = pc->omega.getView();
        auto omegaX = pc->omega_x.getView();
        auto omegaY = pc->omega_y.getView();
        auto omegaZ = pc->omega_z.getView();
        auto P = pc->P.getView();
        auto u = pc->u.getView();
        auto ux = pc->ux.getView();
        auto uy = pc->uy.getView();
        auto uz = pc->uz.getView();
        const auto nlocal = pc->getLocalNum();
        const T particleVolume = T(particle_volume);

        Kokkos::parallel_for(
            "pack_remeshed_vif_3d_type2_samples",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                omega(p)[0] = omegaX(p) * particleVolume;
                omega(p)[1] = omegaY(p) * particleVolume;
                omega(p)[2] = omegaZ(p) * particleVolume;
                omegaX(p) = omega(p)[0];
                omegaY(p) = omega(p)[1];
                omegaZ(p) = omega(p)[2];

                P(p)[0] = ux(p);
                P(p)[1] = uy(p);
                P(p)[2] = uz(p);
                u(p) = P(p);
            });
        Kokkos::fence();
    }

    void recoverFourierSeriesVorticityModesForSampling3D(ComplexField_t& oxModes,
                                                         ComplexField_t& oyModes,
                                                         ComplexField_t& ozModes) {
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
            "recover_vif_3d_type2_vorticity_modes",
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

    bool shouldTracePipeline3D() const {
        return pipeline_trace_m
               && pipeline_trace_freq_m > 0
               && static_cast<int>(this->it_m) % pipeline_trace_freq_m == 0;
    }

    std::array<std::array<int, 3>, 11> traceModes3D() const {
        return {{{1, 1, 1}, {2, 1, 1}, {4, 4, 4}, {8, 8, 8},
                 {16, 16, 16}, {32, 32, 32}, {64, 64, 64},
                 {127, 1, 1}, {128, 1, 1}, {128, 128, 1}, {128, 128, 128}}};
    }

    void tracedSpectralScatter3D(const bool applyShapeFilter) {
        if (!shouldTracePipeline3D()) {
            this->spectralScatter3D(applyShapeFilter);
            return;
        }

        if (!this->nufftType1_mp) {
            throw std::runtime_error("VIF 3D traced scatter called before initNUFFT3D");
        }

        auto& pc = *this->pcontainer_m;
        auto omega = pc.omega.getView();
        auto ox = pc.omega_x.getView();
        auto oy = pc.omega_y.getView();
        auto oz = pc.omega_z.getView();
        auto nlocal = pc.getLocalNum();

        Kokkos::parallel_for(
            "trace_split_omega_components_3d",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                ox(p) = omega(p)[0];
                oy(p) = omega(p)[1];
                oz(p) = omega(p)[2];
            });
        Kokkos::fence();

        this->omega_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
        this->omega_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
        this->omega_z_hat_m = Kokkos::complex<T>(0.0, 0.0);

        this->nufftType1_mp->transform(pc.R, pc.omega_x, this->omega_x_hat_m);
        this->nufftType1_mp->transform(pc.R, pc.omega_y, this->omega_y_hat_m);
        this->nufftType1_mp->transform(pc.R, pc.omega_z, this->omega_z_hat_m);
        traceRawSpectralVorticityModes3D("RAW TYPE1 AFTER SCATTER");

        if (applyShapeFilter && this->useShapeFunctionFilter()) {
            this->applyShapeFunctionToSpectralVorticityModes3D();
            traceRawSpectralVorticityModes3D("RAW TYPE1 AFTER SHAPE");
        }

        auto oxModes = this->omega_x_hat_m.getView();
        auto oyModes = this->omega_y_hat_m.getView();
        auto ozModes = this->omega_z_hat_m.getView();
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

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "trace_precondition_spectral_vorticity_modes_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(oxModes.extent(0)) - nghost,
                         static_cast<int>(oxModes.extent(1)) - nghost,
                         static_cast<int>(oxModes.extent(2)) - nghost}),
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

                if (k2 == T(0)) {
                    oxModes(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    oyModes(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    ozModes(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    return;
                }

                const T invVolumeK2 = T(1.0) / (volume * k2);
                oxModes(i, j, k) *= invVolumeK2;
                oyModes(i, j, k) *= invVolumeK2;
                ozModes(i, j, k) *= invVolumeK2;
            });
        Kokkos::fence();
        traceModes3D("STORED AFTER 1/(VOLUME*K2)");
    }

    void tracePipelineHeader3D(const std::string& stage) const {
        if (!shouldTracePipeline3D() || ippl::Comm->rank() != 0) {
            return;
        }
        std::cout << "\n=============================\n"
                  << "STEP " << this->it_m << " TIME " << std::setprecision(16)
                  << this->time_m << " : " << stage << "\n"
                  << "=============================\n";
    }

    void traceGlobalState3D(const std::string& stage) {
        if (!shouldTracePipeline3D()) {
            return;
        }

        const double energy = this->computeSpectralEnergy3D();
        const double enstrophy = this->computeSpectralEnstrophy3D();
        const double divU = this->computeSpectralDivergenceL23D();
        const double divOmega = this->computeSpectralVorticityDivergenceL23D();
        const double projectionU = this->computeTGVSpectralVelocityProjectionScale3D();
        const double projectionOmega = this->computeTGVSpectralVorticityProjectionScale3D();

        if (ippl::Comm->rank() == 0) {
            std::cout << "GLOBAL " << stage << "\n"
                      << "E = " << std::setprecision(16) << energy << "\n"
                      << "Z = " << enstrophy << "\n"
                      << "projection_u = " << projectionU << "\n"
                      << "projection_omega = " << projectionOmega << "\n"
                      << "div_u = " << divU << "\n"
                      << "div_omega = " << divOmega << "\n";
        }
    }

    void traceParticles3D(const std::string& stage) {
        if (!shouldTracePipeline3D() || ippl::Comm->rank() != 0) {
            return;
        }

        auto pc = this->pcontainer_m;
        const size_type nlocal = pc->getLocalNum();
        std::cout << "PARTICLES " << stage
                  << " rank=0 local_count=" << nlocal
                  << " sampled as rank-local indices\n";
        if (nlocal == 0) {
            return;
        }

        auto R = pc->R.getHostMirror();
        auto Rold = pc->R_old.getHostMirror();
        auto P = pc->P.getHostMirror();
        auto u = pc->u.getHostMirror();
        auto omega = pc->omega.getHostMirror();
        auto omegaX = pc->omega_x.getHostMirror();
        auto omegaY = pc->omega_y.getHostMirror();
        auto omegaZ = pc->omega_z.getHostMirror();
        auto ux = pc->ux.getHostMirror();
        auto uy = pc->uy.getHostMirror();
        auto uz = pc->uz.getHostMirror();
        auto duxdx = pc->duxdx.getHostMirror();
        auto duxdy = pc->duxdy.getHostMirror();
        auto duxdz = pc->duxdz.getHostMirror();
        auto duydx = pc->duydx.getHostMirror();
        auto duydy = pc->duydy.getHostMirror();
        auto duydz = pc->duydz.getHostMirror();
        auto duzdx = pc->duzdx.getHostMirror();
        auto duzdy = pc->duzdy.getHostMirror();
        auto duzdz = pc->duzdz.getHostMirror();
        auto viscosity = pc->viscosity.getHostMirror();
        auto viscosityX = pc->viscosity_x.getHostMirror();
        auto viscosityY = pc->viscosity_y.getHostMirror();
        auto viscosityZ = pc->viscosity_z.getHostMirror();

        Kokkos::deep_copy(R, pc->R.getView());
        Kokkos::deep_copy(Rold, pc->R_old.getView());
        Kokkos::deep_copy(P, pc->P.getView());
        Kokkos::deep_copy(u, pc->u.getView());
        Kokkos::deep_copy(omega, pc->omega.getView());
        Kokkos::deep_copy(omegaX, pc->omega_x.getView());
        Kokkos::deep_copy(omegaY, pc->omega_y.getView());
        Kokkos::deep_copy(omegaZ, pc->omega_z.getView());
        Kokkos::deep_copy(ux, pc->ux.getView());
        Kokkos::deep_copy(uy, pc->uy.getView());
        Kokkos::deep_copy(uz, pc->uz.getView());
        Kokkos::deep_copy(duxdx, pc->duxdx.getView());
        Kokkos::deep_copy(duxdy, pc->duxdy.getView());
        Kokkos::deep_copy(duxdz, pc->duxdz.getView());
        Kokkos::deep_copy(duydx, pc->duydx.getView());
        Kokkos::deep_copy(duydy, pc->duydy.getView());
        Kokkos::deep_copy(duydz, pc->duydz.getView());
        Kokkos::deep_copy(duzdx, pc->duzdx.getView());
        Kokkos::deep_copy(duzdy, pc->duzdy.getView());
        Kokkos::deep_copy(duzdz, pc->duzdz.getView());
        Kokkos::deep_copy(viscosity, pc->viscosity.getView());
        Kokkos::deep_copy(viscosityX, pc->viscosity_x.getView());
        Kokkos::deep_copy(viscosityY, pc->viscosity_y.getView());
        Kokkos::deep_copy(viscosityZ, pc->viscosity_z.getView());

        const std::array<size_type, 5> samples = {
            size_type(0), nlocal / 4, nlocal / 2, (3 * nlocal) / 4, nlocal - 1};
        const unsigned nxp = particlesPerDirection3D();
        const T particleVolume =
            T(this->rmax_m[0] - this->rmin_m[0])
            * T(this->rmax_m[1] - this->rmin_m[1])
            * T(this->rmax_m[2] - this->rmin_m[2])
            / T(nxp * nxp * nxp);
        size_type previous = nlocal;
        for (const auto p : samples) {
            if (p == previous || p >= nlocal) {
                continue;
            }
            previous = p;
            const T sx = omega(p)[0] * duxdx(p) + omega(p)[1] * duxdy(p)
                         + omega(p)[2] * duxdz(p);
            const T sy = omega(p)[0] * duydx(p) + omega(p)[1] * duydy(p)
                         + omega(p)[2] * duydz(p);
            const T sz = omega(p)[0] * duzdx(p) + omega(p)[1] * duzdy(p)
                         + omega(p)[2] * duzdz(p);
            const T dtStretchX = T(this->dt_m) * sx;
            const T dtStretchY = T(this->dt_m) * sy;
            const T dtStretchZ = T(this->dt_m) * sz;
            const T dtViscX = T(this->dt_m) * viscosity(p)[0] * particleVolume;
            const T dtViscY = T(this->dt_m) * viscosity(p)[1] * particleVolume;
            const T dtViscZ = T(this->dt_m) * viscosity(p)[2] * particleVolume;
            const T dispX = P(p)[0] * T(this->dt_m);
            const T dispY = P(p)[1] * T(this->dt_m);
            const T dispZ = P(p)[2] * T(this->dt_m);
            std::cout << "PARTICLE_LOCAL " << p << "\n"
                      << "R = " << R(p) << "\n"
                      << "R_old = " << Rold(p) << "\n"
                      << "P = " << P(p) << "\n"
                      << "u = " << u(p) << "\n"
                      << "u_components = (" << ux(p) << "," << uy(p) << "," << uz(p) << ")\n"
                      << "omega = " << omega(p) << "\n"
                      << "omega_components = (" << omegaX(p) << "," << omegaY(p)
                      << "," << omegaZ(p) << ")\n"
                      << "GRAD_U = [[" << duxdx(p) << "," << duxdy(p) << ","
                      << duxdz(p) << "],[" << duydx(p) << "," << duydy(p)
                      << "," << duydz(p) << "],[" << duzdx(p) << "," << duzdy(p)
                      << "," << duzdz(p) << "]]\n"
                      << "STRETCHING = (" << sx << "," << sy << "," << sz << ")\n"
                      << "VISCOSITY = " << viscosity(p) << "\n"
                      << "viscosity_components = (" << viscosityX(p) << ","
                      << viscosityY(p) << "," << viscosityZ(p) << ")\n"
                      << "RHS_OMEGA = (" << sx + viscosity(p)[0] << ","
                      << sy + viscosity(p)[1] << "," << sz + viscosity(p)[2] << ")\n"
                      << "dt_stretching_update = (" << dtStretchX << ","
                      << dtStretchY << "," << dtStretchZ << ")\n"
                      << "dt_viscosity_update = (" << dtViscX << ","
                      << dtViscY << "," << dtViscZ << ")\n"
                      << "dt_total_omega_update = (" << dtStretchX + dtViscX << ","
                      << dtStretchY + dtViscY << "," << dtStretchZ + dtViscZ << ")\n"
                      << "forward_dt_displacement = (" << dispX << ","
                      << dispY << "," << dispZ << ")\n";
        }
    }

    void traceRawSpectralVorticityModes3D(const std::string& stage) {
        if (!shouldTracePipeline3D()) {
            return;
        }

        auto oxHost = this->omega_x_hat_m.getHostMirror();
        auto oyHost = this->omega_y_hat_m.getHostMirror();
        auto ozHost = this->omega_z_hat_m.getHostMirror();
        Kokkos::deep_copy(oxHost, this->omega_x_hat_m.getView());
        Kokkos::deep_copy(oyHost, this->omega_y_hat_m.getView());
        Kokkos::deep_copy(ozHost, this->omega_z_hat_m.getView());

        const auto& lDom = this->omega_x_hat_m.getLayout().getLocalNDIndex();
        const int nghost = this->omega_x_hat_m.getNghost();
        const int Nx = this->nr_m[0];
        const int Ny = this->nr_m[1];
        const int Nz = this->nr_m[2];
        const T Lx = this->rmax_m[0] - this->rmin_m[0];
        const T Ly = this->rmax_m[1] - this->rmin_m[1];
        const T Lz = this->rmax_m[2] - this->rmin_m[2];
        const T volume = Lx * Ly * Lz;
        const T twoPi = T(2.0 * std::acos(-1.0));

        for (const auto& mode : traceModes3D()) {
            const int mx = mode[0];
            const int my = mode[1];
            const int mz = mode[2];
            if (std::abs(mx) > Nx / 2 || std::abs(my) > Ny / 2 || std::abs(mz) > Nz / 2) {
                continue;
            }

            const int gx = mx >= 0 ? mx : Nx + mx;
            const int gy = my >= 0 ? my : Ny + my;
            const int gz = mz >= 0 ? mz : Nz + mz;
            if (gx < lDom[0].first() || gx > lDom[0].last()
                || gy < lDom[1].first() || gy > lDom[1].last()
                || gz < lDom[2].first() || gz > lDom[2].last()) {
                continue;
            }

            const int i = gx - lDom[0].first() + nghost;
            const int j = gy - lDom[1].first() + nghost;
            const int k = gz - lDom[2].first() + nghost;
            const bool notMidX = (gx != Nx / 2);
            const bool notMidY = (gy != Ny / 2);
            const bool notMidZ = (gz != Nz / 2);
            const T kx = notMidX * twoPi * mx / Lx;
            const T ky = notMidY * twoPi * my / Ly;
            const T kz = notMidZ * twoPi * mz / Lz;
            const T k2 = kx * kx + ky * ky + kz * kz;
            const T invVolumeK2 = k2 == T(0) ? T(0) : T(1.0) / (volume * k2);

            std::cout << "MODE " << stage << " rank=" << ippl::Comm->rank()
                      << " (" << mx << "," << my << "," << mz << ")\n"
                      << "kx,ky,kz = (" << kx << "," << ky << "," << kz << ")\n"
                      << "k2 = " << k2 << " volume = " << volume
                      << " inv_volume_k2 = " << invVolumeK2 << "\n"
                      << "omega_raw = (" << oxHost(i, j, k) << ","
                      << oyHost(i, j, k) << "," << ozHost(i, j, k) << ")\n"
                      << "omega_raw_over_volume = (" << oxHost(i, j, k) / volume
                      << "," << oyHost(i, j, k) / volume << ","
                      << ozHost(i, j, k) / volume << ")\n"
                      << "expected_stored_raw_over_volume_k2 = ("
                      << invVolumeK2 * oxHost(i, j, k) << ","
                      << invVolumeK2 * oyHost(i, j, k) << ","
                      << invVolumeK2 * ozHost(i, j, k) << ")\n";
        }
    }

    void traceModes3D(const std::string& stage) {
        if (!shouldTracePipeline3D()) {
            return;
        }

        auto oxHost = this->omega_x_hat_m.getHostMirror();
        auto oyHost = this->omega_y_hat_m.getHostMirror();
        auto ozHost = this->omega_z_hat_m.getHostMirror();
        auto uxHost = this->ux_hat_m.getHostMirror();
        auto uyHost = this->uy_hat_m.getHostMirror();
        auto uzHost = this->uz_hat_m.getHostMirror();
        auto duxdxHost = this->duxdx_hat_m.getHostMirror();
        auto duxdyHost = this->duxdy_hat_m.getHostMirror();
        auto duxdzHost = this->duxdz_hat_m.getHostMirror();
        auto duydxHost = this->duydx_hat_m.getHostMirror();
        auto duydyHost = this->duydy_hat_m.getHostMirror();
        auto duydzHost = this->duydz_hat_m.getHostMirror();
        auto duzdxHost = this->duzdx_hat_m.getHostMirror();
        auto duzdyHost = this->duzdy_hat_m.getHostMirror();
        auto duzdzHost = this->duzdz_hat_m.getHostMirror();

        Kokkos::deep_copy(oxHost, this->omega_x_hat_m.getView());
        Kokkos::deep_copy(oyHost, this->omega_y_hat_m.getView());
        Kokkos::deep_copy(ozHost, this->omega_z_hat_m.getView());
        Kokkos::deep_copy(uxHost, this->ux_hat_m.getView());
        Kokkos::deep_copy(uyHost, this->uy_hat_m.getView());
        Kokkos::deep_copy(uzHost, this->uz_hat_m.getView());
        Kokkos::deep_copy(duxdxHost, this->duxdx_hat_m.getView());
        Kokkos::deep_copy(duxdyHost, this->duxdy_hat_m.getView());
        Kokkos::deep_copy(duxdzHost, this->duxdz_hat_m.getView());
        Kokkos::deep_copy(duydxHost, this->duydx_hat_m.getView());
        Kokkos::deep_copy(duydyHost, this->duydy_hat_m.getView());
        Kokkos::deep_copy(duydzHost, this->duydz_hat_m.getView());
        Kokkos::deep_copy(duzdxHost, this->duzdx_hat_m.getView());
        Kokkos::deep_copy(duzdyHost, this->duzdy_hat_m.getView());
        Kokkos::deep_copy(duzdzHost, this->duzdz_hat_m.getView());

        const auto& lDom = this->omega_x_hat_m.getLayout().getLocalNDIndex();
        const int nghost = this->omega_x_hat_m.getNghost();
        const int Nx = this->nr_m[0];
        const int Ny = this->nr_m[1];
        const int Nz = this->nr_m[2];
        const T Lx = this->rmax_m[0] - this->rmin_m[0];
        const T Ly = this->rmax_m[1] - this->rmin_m[1];
        const T Lz = this->rmax_m[2] - this->rmin_m[2];
        const T volume = Lx * Ly * Lz;
        const T twoPi = T(2.0 * std::acos(-1.0));

        for (const auto& mode : traceModes3D()) {
            const int mx = mode[0];
            const int my = mode[1];
            const int mz = mode[2];
            if (std::abs(mx) > Nx / 2 || std::abs(my) > Ny / 2 || std::abs(mz) > Nz / 2) {
                continue;
            }
            const int gx = mx >= 0 ? mx : Nx + mx;
            const int gy = my >= 0 ? my : Ny + my;
            const int gz = mz >= 0 ? mz : Nz + mz;
            if (gx < lDom[0].first() || gx > lDom[0].last()
                || gy < lDom[1].first() || gy > lDom[1].last()
                || gz < lDom[2].first() || gz > lDom[2].last()) {
                continue;
            }

            const int i = gx - lDom[0].first() + nghost;
            const int j = gy - lDom[1].first() + nghost;
            const int k = gz - lDom[2].first() + nghost;
            const bool notMidX = (gx != Nx / 2);
            const bool notMidY = (gy != Ny / 2);
            const bool notMidZ = (gz != Nz / 2);
            const T kx = notMidX * twoPi * mx / Lx;
            const T ky = notMidY * twoPi * my / Ly;
            const T kz = notMidZ * twoPi * mz / Lz;
            const T k2 = kx * kx + ky * ky + kz * kz;
            const T invVolumeK2 = k2 == T(0) ? T(0) : T(1.0) / (volume * k2);
            std::cout << "MODE " << stage << " rank=" << ippl::Comm->rank()
                      << " (" << mx << "," << my << "," << mz << ")\n"
                      << "kx,ky,kz = (" << kx << "," << ky << "," << kz << ")\n"
                      << "k2 = " << k2 << " volume = " << volume
                      << " inv_volume_k2 = " << invVolumeK2 << "\n"
                      << "omega_stored = (" << oxHost(i, j, k) << ","
                      << oyHost(i, j, k) << "," << ozHost(i, j, k) << ")\n"
                      << "omega_physical_coeff = (" << k2 * oxHost(i, j, k) << ","
                      << k2 * oyHost(i, j, k) << "," << k2 * ozHost(i, j, k) << ")\n"
                      << "u_hat = (" << uxHost(i, j, k) << "," << uyHost(i, j, k)
                      << "," << uzHost(i, j, k) << ")\n"
                      << "grad_u_hat = [[" << duxdxHost(i, j, k) << ","
                      << duxdyHost(i, j, k) << "," << duxdzHost(i, j, k)
                      << "],[" << duydxHost(i, j, k) << "," << duydyHost(i, j, k)
                      << "," << duydzHost(i, j, k) << "],[" << duzdxHost(i, j, k)
                      << "," << duzdyHost(i, j, k) << "," << duzdzHost(i, j, k)
                      << "]]\n";
        }
    }

    void tracePipelineState3D(const std::string& stage) {
        if (!shouldTracePipeline3D()) {
            return;
        }
        tracePipelineHeader3D(stage);
        traceGlobalState3D(stage);
        traceParticles3D(stage);
        ippl::Comm->barrier();
    }

    void zeroParticleViscosityForTrace3D() {
        auto pc = this->pcontainer_m;
        auto viscosity = pc->viscosity.getView();
        auto viscosityX = pc->viscosity_x.getView();
        auto viscosityY = pc->viscosity_y.getView();
        auto viscosityZ = pc->viscosity_z.getView();
        const auto n = pc->getLocalNum();

        Kokkos::parallel_for(
            "zero_particle_viscosity_for_trace_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                viscosityX(p) = T(0);
                viscosityY(p) = T(0);
                viscosityZ(p) = T(0);
                viscosity(p)[0] = T(0);
                viscosity(p)[1] = T(0);
                viscosity(p)[2] = T(0);
            });
        Kokkos::fence();
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

        tracePipelineState3D("BEFORE SCATTER");
        IpplTimings::startTimer(par2gridTimer);
        refreshSpectralVorticityModes3D(true);
        IpplTimings::stopTimer(par2gridTimer);
        tracePipelineState3D("AFTER SCATTER");

        IpplTimings::startTimer(SolveTimer);
        this->computeSpectralVelocityModes3D();
        tracePipelineState3D("AFTER BIOT SAVART");
        traceModes3D("AFTER BIOT SAVART");
        this->applyConfiguredSpectralFilter3D(this->ux_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uy_hat_m);
        this->applyConfiguredSpectralFilter3D(this->uz_hat_m);
        tracePipelineState3D("AFTER VELOCITY FILTER");
        this->computeSpectralVelocityGradientModes3D();
        IpplTimings::stopTimer(SolveTimer);
        tracePipelineState3D("AFTER GRADIENT");
        traceModes3D("AFTER GRADIENT");

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
        tracePipelineState3D("AFTER GRADIENT GATHER");

        IpplTimings::startTimer(stretchingTimer);
        tracePipelineState3D("BEFORE STRETCHING");
        this->applyParticleVortexStretching3D();
        IpplTimings::stopTimer(stretchingTimer);
        tracePipelineState3D("AFTER STRETCHING");

        IpplTimings::startTimer(grid2parTimer);
        this->spectralGather3D();
        IpplTimings::stopTimer(grid2parTimer);
        tracePipelineState3D("AFTER VELOCITY GATHER");

        if (this->viscosity_m > 0.0) {
            this->viscosity_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->viscosity_z_hat_m = Kokkos::complex<T>(0.0, 0.0);
            this->computeSpectralViscosityModes3D();
            tracePipelineState3D("AFTER VISCOSITY MODES");
            this->spectralGatherViscosity3D();
            tracePipelineState3D("AFTER VISCOSITY GATHER");
            this->applyParticleViscosity3D();
            tracePipelineState3D("AFTER VISCOSITY UPDATE");
        } else if (shouldTracePipeline3D()) {
            zeroParticleViscosityForTrace3D();
            tracePipelineState3D("VISCOSITY ZERO");
        }

        IpplTimings::startTimer(RTimer);
        tracePipelineState3D("BEFORE PARTICLE PUSH");
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
        tracePipelineState3D("AFTER PARTICLE PUSH BEFORE UPDATE");

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);
        tracePipelineState3D("AFTER PC UPDATE");

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

        tracedSpectralScatter3D(!skipFilter);
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
