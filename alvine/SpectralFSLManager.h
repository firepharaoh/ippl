#ifndef IPPL_SPECTRAL_FSL_MANAGER_H
#define IPPL_SPECTRAL_FSL_MANAGER_H

#include <fstream>
#include <memory>

#include "AlvineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randu.h"
// #include "SinusoidalJitter.hpp"
#include "VortexDistributions.h"
#include "VtkDump.hpp"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
using host_type = typename ippl::ParticleAttrib<T>::host_mirror_type;
/*using host_type = typename ippl::ParticleAttrib<T>::HostMirror;*/

template <typename T, unsigned Dim, typename VortexDistribution>
class SpectralFSLManager : public AlvineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

    FieldLayout_t<Dim> FL_m;    // Store the field layout
    Mesh_t<Dim> mesh_m;          // Store the mesh

    // Constructor declaration
    SpectralFSLManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_,
                        std::string& solver_, int dump_freq_,
                        double dt_ = 0.05,
                        std::string method_ = "sfsl",
                        int spectral_filter_ = 0,
                        std::string time_integrator_ = "leapfrog",
                        Vector_t<double, Dim> rmin_ = 0.0,
                        Vector_t<double, Dim> rmax_ = 10.0,
                        Vector_t<double, Dim> origin_ = 0.0,
                        FieldLayout_t<Dim>& FL_ = nullptr,
                        Mesh_t<Dim>& mesh_ = nullptr)
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_, dt_, method_,
                                spectral_filter_, 0.0, time_integrator_) {
        this->rmin_m   = rmin_;
        this->rmax_m   = rmax_;
        this->origin_m = origin_;
        this->FL_m     = FL_;       // Store the layout
        this->mesh_m   = mesh_;     // Store the mesh
    }

    ~SpectralFSLManager() {}

    void pre_run() override {
        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }

        Vector_t<double, Dim> dr = this->rmax_m - this->rmin_m;

        this->hr_m = dr / this->nr_m;

        // dt_m is set from the command line by the executable constructor.

        this->it_m   = 0;
        this->time_m = 0.0;

        // this->np_m = 10000; //this->nr_m[0] * this->nr_m[0];

        this->decomp_m.fill(true);
        this->isAllPeriodic_m = true;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m,
            this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields();

        this->setFieldSolver(std::make_shared<FieldSolver_t>(
            this->solver_m, &this->fcontainer_m->getOmegaField()));

        this->fsolver_m->initSolver();

        // this->setLoadBalancer( std::make_shared<LoadBalancer_t>( this->lbt_m,
        // this->fcontainer_m, this->pcontainer_m, this->fsolver_m) );

        // initializeParticles();

        // this->par2grid();
        this->initNUFFT();
        initializeGridVorticity();
        //normalizeInitialGridCirculation();

        auto omega0 = this->fcontainer_m->getOmegaField().deepCopy();
        this->fsolver_m->runSolver();
        this->computeVelocityField();
        Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega0.getView());
        double omega_init = computeOmegaL2();

        initializeVirtualParticles();
        this->spectralScatter();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_hat_m);
        }
        this->computeSpectralVelocityModes();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
        }

        logEnergyDiagnostics();
        logEnstrophyDiagnostics();
        this->logCirculationDiagnostics(this->computeParticleCirculation());
        logVorticitySpectrum();
        logDivergenceDiagnostics();

        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        this->logTgvVelocityDiagnostics("sfsl_tgv_velocity_error.csv");

        if (ippl::Comm->rank() == 0) {
            Inform m("debug ");
            m << "omega L2 after initialization = " << omega_init << endl;
        }
    }

    double computeOmegaL2() {
        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega       = omegaField.getView();
        double local     = 0.0;
        const int nghost = omegaField.getNghost();

        Kokkos::parallel_reduce(
            "omega_l2", ippl::getRangePolicy(omega, nghost),
            KOKKOS_LAMBDA(const int i, const int j, double& sum) {
                sum += omega(i, j) * omega(i, j);
            },
            local);

        double global = 0.0;
        ippl::Comm->reduce(local, global, 1, std::plus<double>());

        return std::sqrt(global);
    }

void logPushDebug() {
    auto pc = this->pcontainer_m;

    auto P_view = pc->P.getView();

    double dt = this->dt_m;
    double dx = this->hr_m[0];

    double localMaxVel = 0.0;
    double localMaxDx  = 0.0;
    double localMaxDy  = 0.0;

    Kokkos::parallel_reduce(
        "push_debug",
        pc->getLocalNum(),
        KOKKOS_LAMBDA(const int p,
                      double& maxVel,
                      double& maxDx,
                      double& maxDy) {

            double ux = P_view(p)[0];
            double uy = P_view(p)[1];

            double vel = sqrt(ux * ux + uy * uy);

            double dispX = fabs(ux * dt);
            double dispY = fabs(uy * dt);

            if (vel   > maxVel) maxVel = vel;
            if (dispX > maxDx)  maxDx  = dispX;
            if (dispY > maxDy)  maxDy  = dispY;

        },
        Kokkos::Max<double>(localMaxVel),
        Kokkos::Max<double>(localMaxDx),
        Kokkos::Max<double>(localMaxDy)
    );

    double globalMaxVel = localMaxVel;
    double globalMaxDx  = localMaxDx;
    double globalMaxDy  = localMaxDy;

    if (ippl::Comm->rank() == 0) {
        Inform m("push_debug ");

        m << "step = " << this->it_m
          << ", max velocity = " << globalMaxVel
          << ", max |dx| = " << globalMaxDx
          << ", max |dy| = " << globalMaxDy
          << ", max displacement/dx = "
          << std::max(globalMaxDx, globalMaxDy) / dx
          << endl;
    }
}

void logVelocityRatioDebug() {
    auto pc = this->pcontainer_m;
    auto P_view = pc->P.getView();

    double localMaxRatio = 0.0;
    double localSumRatio = 0.0;
    size_type localN = pc->getLocalNum();

    Kokkos::parallel_reduce(
        "velocity_ratio_debug",
        localN,
        KOKKOS_LAMBDA(const int p, double& maxRatio, double& sumRatio) {
            double ux = fabs(P_view(p)[0]);
            double uy = fabs(P_view(p)[1]);

            double ratio = uy / (ux + 1.0e-30);

            if (ratio > maxRatio) maxRatio = ratio;
            sumRatio += ratio;
        },
        Kokkos::Max<double>(localMaxRatio),
        localSumRatio
    );

    double globalMaxRatio = 0.0;
    double globalSumRatio = 0.0;
    size_type globalN = 0;

    MPI_Allreduce(&localMaxRatio, &globalMaxRatio, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localSumRatio, &globalSumRatio, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localN, &globalN, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (ippl::Comm->rank() == 0) {
        Inform m("velocity_ratio ");
        m << "step = " << this->it_m
          << ", max |uy|/|ux| = " << globalMaxRatio
          << ", avg |uy|/|ux| = " << globalSumRatio / globalN
          << endl;
    }
}

void initializeVirtualParticles(){
    clearVirtualParticles();
    auto* FL = &this->fcontainer_m->getFL();
    auto local = FL->getLocalNDIndex();

    int i0 = local[0].first();
    int i1 = local[0].last();
    int j0 = local[1].first();
    int j1 = local[1].last();

    size_type nx = i1 - i0 + 1;
    size_type ny = j1 - j0 + 1;
    size_type nlocal = nx * ny;
    const int nghost = this->fcontainer_m->getOmegaField().getNghost();

    auto pc = this->pcontainer_m;
    pc->create(nlocal);
    

    auto R_view = pc->R.getView();
    auto omega_p = pc->omega.getView();
    auto omega_g = this->fcontainer_m->getOmegaField().getView();
    auto P_view = pc->P.getView();
    auto u_g = this->fcontainer_m->getUField().getView();



    Vector_t<double, Dim> rmin = this->rmin_m;
    Vector_t<double, Dim> hr   = this->hr_m;
    double dA = hr[0] * hr[1];
    Kokkos::parallel_for(
        "init_virtual_particles_from_grid",
        nlocal,
        KOKKOS_LAMBDA(const int p) {
            int ii = p % nx;
            int jj = p / nx;

            int i = i0 + ii;
            int j = j0 + jj;
            int li = ii + nghost;
            int lj = jj + nghost;

            R_view(p)[0] = rmin[0] + (i+0.5) * hr[0];
            R_view(p)[1] = rmin[1] + (j+0.5) * hr[1];

            omega_p(p) = omega_g(li, lj) * dA;
            P_view(p) = u_g(li, lj);
        }
    );

    Kokkos::fence();
}


void initializeGridVorticity() {
    auto& omegaField = this->fcontainer_m->getOmegaField();
    auto omega_view = omegaField.getView();

    auto localND = this->fcontainer_m->getFL().getLocalNDIndex();

    int i0 = localND[0].first();
    int i1 = localND[0].last();
    int j0 = localND[1].first();
    int j1 = localND[1].last();
    const int nghost = omegaField.getNghost();

    Vector_t<double, Dim> rmin = this->rmin_m;
    Vector_t<double, Dim> rmax = this->rmax_m;
    Vector_t<double, Dim> hr   = this->hr_m;

    double y_mid = 0.5 * (rmin[1] + rmax[1]);
    double y_low = y_mid - 1.0;
    double y_high = y_mid + 1.0;

    Kokkos::parallel_for(
        "initialize_grid_vorticity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost},
                                               {nghost + i1 - i0 + 1,
                                                nghost + j1 - j0 + 1}),
        KOKKOS_LAMBDA(const int li, const int lj) {
            int i = i0 + li - nghost;
            int j = j0 + lj - nghost;

            double y = rmin[1] + (j + 0.5) * hr[1];
            //double perturb = alvine::sinusoidalVorticityPerturbation(i, j);

            //omega_view(li, lj) = (y >= y_low && y <= y_high) ? 1.0 + perturb : 0.0;
            omega_view(li, lj) = 2.0 * cos(rmin[0] + (i + 0.5) * hr[0]) * cos(rmin[1] + (j + 0.5) * hr[1]) ; //Taylor-Green vortex initial condition.
        }
    );

    Kokkos::fence();
}

void normalizeInitialGridCirculation() {
    double gamma = this->computeGridCirculation();
    double targetGamma = (this->rmax_m[0] - this->rmin_m[0]) * 2.0;

    if (std::fabs(gamma) < 1e-30) {
        return;
    }

    this->fcontainer_m->getOmegaField() =
        this->fcontainer_m->getOmegaField() * (targetGamma / gamma);
}

void pushVirtualParticlesForward() {
    auto pc = this->pcontainer_m;

    // 1. Gather velocity from grid to particles
    //this->grid2par();   // fills pc->P using uField

    // Debug before push: checks expected displacement = P * dt
    logPushDebug();
    logVelocityRatioDebug();
    // 2. Move particles forward
    auto R_view = pc->R.getView();
    auto P_view = pc->P.getView();

    double dt = this->dt_m;

    Kokkos::parallel_for(
        "push_virtual_particles_forward",
        pc->getLocalNum(),
        KOKKOS_LAMBDA(const int p) {
            R_view(p) = R_view(p) + P_view(p) * dt;
        }
    );

    Kokkos::fence();

    // 3. Apply particle boundary conditions / update ownership
    pc->update();
}

void clearVirtualParticles() {
    auto pc = this->pcontainer_m;
    size_type nlocal = pc->getLocalNum();

    if (nlocal == 0) {
        return;
    }

    Kokkos::View<bool*> invalid("invalid", nlocal);

    Kokkos::parallel_for(
        "mark_all_particles_invalid",
        nlocal,
        KOKKOS_LAMBDA(const int p) {
            invalid(p) = true;
        }
    );

    Kokkos::fence();

    pc->destroy(invalid, nlocal);
}

    void logEnergyDiagnostics() {
        double energy = this->computeSpectralEnergy();

        if (!this->energy_initialized_m) {
            this->energy0_m            = energy;
            this->energy_initialized_m = true;

            if (ippl::Comm->rank() == 0) {
                std::ofstream out(this->diagnosticFileName("spectral_energy.csv"), std::ios::out);
                out << "method,dt,step,time,energy,rel_error,normalized_energy\n";
            }
            ippl::Comm->barrier();
        }

        double relErr = this->relativeError(energy, this->energy0_m);
        double normalizedEnergy =
            energy / (std::fabs(this->energy0_m) > 1e-30 ? this->energy0_m : 1e-30);

        if (ippl::Comm->rank() == 0) {
            Inform m("spectral energy ");
            m << "kinetic energy = " << energy << ", relError = " << relErr
              << ", normalizedEnergy = " << normalizedEnergy << endl;

            std::ofstream out(this->diagnosticFileName("spectral_energy.csv"), std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << energy << "," << relErr << "," << normalizedEnergy
                << "\n";
        }
    }

    void logEnstrophyDiagnostics() {
        double enstrophy = this->computeSpectralEnstrophy();

        if (!this->enstrophy_initialized_m) {
            this->enstrophy0_m            = enstrophy;
            this->enstrophy_initialized_m = true;

            if (ippl::Comm->rank() == 0) {
                std::ofstream out(this->diagnosticFileName("spectral_enstrophy.csv"),
                                  std::ios::out);
                out << "method,dt,step,time,enstrophy,rel_error\n";
            }
            ippl::Comm->barrier();
        }

        double relErr = this->relativeError(enstrophy, this->enstrophy0_m);

        if (ippl::Comm->rank() == 0) {
            Inform m("spectral enstrophy ");
            m << "enstrophy = " << enstrophy << ", relError = " << relErr << endl;

            std::ofstream out(this->diagnosticFileName("spectral_enstrophy.csv"), std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << enstrophy << "," << relErr << "\n";
        }
    }

    void logVorticitySpectrum() {
        const auto spectrum = this->computeSpectralVorticitySpectrum();

        if (ippl::Comm->rank() != 0) {
            return;
        }

        std::ofstream out(
            this->diagnosticFileName("spectral_vorticity_spectrum.csv"),
            this->it_m == 0 ? std::ios::out : std::ios::app);
        out.precision(16);
        out.setf(std::ios::scientific, std::ios::floatfield);

        if (this->it_m == 0) {
            out << "method,dt,step,time,shell,normalized_radius,mode_count,shell_enstrophy,"
                   "mean_mode_enstrophy,cumulative_fraction,complete_shell\n";
        }

        double totalEnstrophy = 0.0;
        for (const auto& shell : spectrum) {
            totalEnstrophy += shell.enstrophy;
        }

        std::size_t completeShellLimit = spectrum.size();
        for (std::size_t shell = 0; shell < spectrum.size(); ++shell) {
            if (!spectrum[shell].complete) {
                completeShellLimit = shell;
                break;
            }
        }

        const std::size_t tailStart =
            static_cast<std::size_t>(std::ceil(0.8 * static_cast<double>(completeShellLimit)));
        double completeEnstrophy = 0.0;
        double tailEnstrophy     = 0.0;
        for (std::size_t shell = 0; shell < completeShellLimit; ++shell) {
            completeEnstrophy += spectrum[shell].enstrophy;
            if (shell >= tailStart) {
                tailEnstrophy += spectrum[shell].enstrophy;
            }
        }

        double cumulativeEnstrophy = 0.0;
        for (std::size_t shell = 0; shell < spectrum.size(); ++shell) {
            const auto& bin = spectrum[shell];
            cumulativeEnstrophy += bin.enstrophy;

            const double normalizedRadius =
                completeShellLimit > 0
                    ? static_cast<double>(shell) / static_cast<double>(completeShellLimit)
                    : 0.0;
            const double meanModeEnstrophy =
                bin.modeCount > 0 ? bin.enstrophy / static_cast<double>(bin.modeCount) : 0.0;
            const double cumulativeFraction =
                totalEnstrophy > 0.0 ? cumulativeEnstrophy / totalEnstrophy : 0.0;

            out << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << shell << ","
                << normalizedRadius << "," << bin.modeCount << "," << bin.enstrophy << ","
                << meanModeEnstrophy << "," << cumulativeFraction << ","
                << (bin.complete ? 1 : 0) << "\n";
        }

        std::ofstream tailOut(
            this->diagnosticFileName("spectral_vorticity_tail.csv"),
            this->it_m == 0 ? std::ios::out : std::ios::app);
        tailOut.precision(16);
        tailOut.setf(std::ios::scientific, std::ios::floatfield);

        if (this->it_m == 0) {
            tailOut << "method,dt,step,time,tail_start_shell,complete_shell_limit,"
                       "tail_enstrophy,complete_enstrophy,tail_fraction\n";
        }

        const double tailFraction =
            completeEnstrophy > 0.0 ? tailEnstrophy / completeEnstrophy : 0.0;
        tailOut << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << tailStart << ","
                << completeShellLimit << "," << tailEnstrophy << "," << completeEnstrophy << ","
                << tailFraction << "\n";
    }

    void logDivergenceDiagnostics() {
        double divL2 = this->computeSpectralDivergenceL2();

        if (ippl::Comm->rank() == 0) {
            Inform m("spectral divergence ");
            m << "L2 = " << divL2 << endl;

            std::ofstream out(this->diagnosticFileName("spectral_divergence.csv"), std::ios::app);

            if (this->it_m == 0) {
                out << "method,dt,step,time,div_l2\n";
            }

            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            out << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << divL2 << "\n";
        }
    }

    void dump() override {
        static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("vtkDump");
        IpplTimings::startTimer(dumpTimer);

        alvine::vtk::writeScalarField2D("data/SpectralFSL", "omega", this->fcontainer_m->getOmegaField(),
                                        this->rmin_m, this->hr_m, this->it_m);

        IpplTimings::stopTimer(dumpTimer);
    }

    void logOmegaField() {
        alvine::vtk::writeScalarFieldCsv2D("data/SpectralFSL/omega_csv", "omega",
                                           this->fcontainer_m->getOmegaField(), this->rmin_m,
                                           this->hr_m, this->it_m);
    }

    void post_step() override {
        Inform m("Step: ");
        this->time_m += this->dt_m;
        this->it_m++;

        if (this->dump_freq_m > 0 && this->it_m % this->dump_freq_m == 0) {
            // this->logOmegaField();
            this->dump();
        }

        m << this->it_m << " Done" << endl;
    }

    void advance() override {
        if (this->useRK4()) {
            advectForwardRK4();
        } else {
            advectForward();
        }
    }

    void pushVirtualParticlesForwardRK4() {
        auto pc = this->pcontainer_m;
        const T dt = this->dt_m;

        pc->rk4_R0 = pc->R;
        pc->rk4_k1 = pc->P;

        pc->R = pc->rk4_R0 + (0.5 * dt) * pc->rk4_k1;
        pc->update();
        this->spectralGather();
        pc->rk4_k2 = pc->P;

        pc->R = pc->rk4_R0 + (0.5 * dt) * pc->rk4_k2;
        pc->update();
        this->spectralGather();
        pc->rk4_k3 = pc->P;

        pc->R = pc->rk4_R0 + dt * pc->rk4_k3;
        pc->update();
        this->spectralGather();
        pc->rk4_k4 = pc->P;

        pc->R = pc->rk4_R0 + (dt / 6.0) *
                              (pc->rk4_k1 + 2.0 * pc->rk4_k2 + 2.0 * pc->rk4_k3 + pc->rk4_k4);
        pc->update();
    }

    void advectForwardRK4() {
        static IpplTimings::TimerRef RTimer        = IpplTimings::getTimer("rk4PushPosition");
        static IpplTimings::TimerRef SolveTimer    = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("par2grid");

        initializeVirtualParticles();

        IpplTimings::startTimer(RTimer);
        pushVirtualParticlesForwardRK4();
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(par2gridTimer);
        this->spectralScatter();
        IpplTimings::stopTimer(par2gridTimer);

        IpplTimings::startTimer(SolveTimer);
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_hat_m);
        }
        this->computeSpectralVelocityModes();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
        }
        logEnergyDiagnostics();
        logEnstrophyDiagnostics();
        logVorticitySpectrum();
        logDivergenceDiagnostics();

        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->logCirculationDiagnostics(this->computeGridCirculation());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        this->logTgvVelocityDiagnostics("sfsl_tgv_velocity_error.csv");
        IpplTimings::stopTimer(SolveTimer);
    }

    void advectForward() {
        static IpplTimings::TimerRef PTimer        = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer        = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef SolveTimer    = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("par2grid");

        //auto omega_n = this->fcontainer_m->getOmegaField().deepCopy();

        // 1. Compute velocity u^n from omega^n
        // The FFT solver writes the Poisson solution into omegaField. Restore the
        // saved vorticity before creating/remapping virtual particles.
                
        // 2. Create virtual particles from omega^n
        initializeVirtualParticles();

        // 3. Push particles using u^n
        IpplTimings::startTimer(RTimer);
        pushVirtualParticlesForward();
        IpplTimings::stopTimer(RTimer);

        // 4. Scatter particles to form omega^{n+1}
        IpplTimings::startTimer(par2gridTimer);
        this->spectralScatter();
        IpplTimings::stopTimer(par2gridTimer);

        //clearVirtualParticles(); Instead of deleting temporary particles, we can reuse them in the next step to avoid unnecessary memory allocation and deallocation overhead.
        // 5. Compute the new vorticity field omega^{n+1} from the scattered particles
        IpplTimings::startTimer(SolveTimer);
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->omega_hat_m);
        }
        this->computeSpectralVelocityModes();
        if (this->useHouLiFilter()) {
            this->Hou_Li_filter(this->ux_hat_m);
            this->Hou_Li_filter(this->uy_hat_m);
        }
        logEnergyDiagnostics();
        logEnstrophyDiagnostics();
        logVorticitySpectrum();
        logDivergenceDiagnostics();

        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->logCirculationDiagnostics(this->computeGridCirculation());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
        this->logTgvVelocityDiagnostics("sfsl_tgv_velocity_error.csv");
        IpplTimings::stopTimer(SolveTimer);
    }
};
#endif
