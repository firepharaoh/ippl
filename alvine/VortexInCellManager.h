#ifndef IPPL_VORTEX_IN_CELL_MANAGER_H
#define IPPL_VORTEX_IN_CELL_MANAGER_H

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
class VortexInCellManager : public AlvineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

    FieldLayout_t<Dim> FL_m;    // Store the field layout
    Mesh_t<Dim> mesh_m;          // Store the mesh
    bool bootstrap_next_push_m = false;
    int remesh_freq_m = 1;

    // Constructor declaration
    VortexInCellManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_,
                        std::string& solver_, int dump_freq_, int remesh_freq_ = 1,
                        double dt_ = 0.05,
                        std::string method_ = "vic",
                        std::string time_integrator_ = "leapfrog",
                        Vector_t<double, Dim> rmin_ = 0.0,
                        Vector_t<double, Dim> rmax_ = 10.0,
                        Vector_t<double, Dim> origin_ = 0.0,
                        FieldLayout_t<Dim>& FL_ = nullptr,
                        Mesh_t<Dim>& mesh_ = nullptr)
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_, dt_, method_, 0, 0.0,
                                time_integrator_) {
        this->rmin_m   = rmin_;
        this->rmax_m   = rmax_;
        this->origin_m = origin_;
        this->FL_m     = FL_;       // Store the layout
        this->mesh_m   = mesh_;     // Store the mesh
        remesh_freq_m  = remesh_freq_;
    }

    ~VortexInCellManager() {}

    void pre_run() override {
        const std::string particlesFile = this->diagnosticFileName("particles.csv");
        Inform csvout(NULL, particlesFile.c_str(), Inform::OVERWRITE);
        csvout.precision(16);
        csvout.setf(std::ios::scientific, std::ios::floatfield);

        if constexpr (Dim == 2) {
            csvout << "time,index,pos_x,pos_y,vorticity" << endl;
        } else {
            csvout << "time,index,pos_x,pos_y,pos_z" << endl;
        }

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

        initializeParticles();
        this->pcontainer_m->R_old = this->pcontainer_m->R;
        this->par2grid();
        this->logTgvVorticityDiagnostics();
    }

    void initializeParticles() {
        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL   = &this->fcontainer_m->getFL();
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout;
        const bool isFEM = (this->solver_m == "FEM") || (this->solver_m == "FEM_PRECON");
        rlayout = ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>(*FL, *mesh, isFEM);
        const ippl::NDIndex<Dim>& local = FL->getLocalNDIndex();

        // 1. Global lattice dimensions based on user-supplied particle count
        unsigned nxp_global     = static_cast<unsigned>(std::sqrt(this->np_m));
        unsigned nyp_global     = this->np_m / nxp_global;
        size_type totalP_global = nxp_global * nyp_global;

        // 2. Physical bounds and spacing
        double xmin_global = this->rmin_m[0];
        double xmax_global = this->rmax_m[0];
        double ymin_global = this->rmin_m[1];
        double ymax_global = this->rmax_m[1];

        double dxp = (xmax_global - xmin_global) / nxp_global;
        double dyp = (ymax_global - ymin_global) / nyp_global;

        // 3. Local domain from grid decomposition
        int local_start_x = local[0].first();
        int local_end_x   = local[0].last();
        int local_start_y = local[1].first();
        int local_end_y   = local[1].last();

        double xmin_local = xmin_global + local_start_x * this->hr_m[0];
        double xmax_local = xmin_global + (local_end_x + 1) * this->hr_m[0];
        double ymin_local = ymin_global + local_start_y * this->hr_m[1];
        double ymax_local = ymin_global + (local_end_y + 1) * this->hr_m[1];

        // 4. Intersect rank's physical rectangle with the full TGV domain
        double y_low  = std::max(ymin_local, ymin_global);
        double y_high = std::min(ymax_local, ymax_global);
        if (y_low >= y_high) {
            pc->create(0);
            return;
        }

        // 5. Find the range of lattice indices that fall inside the rank's rectangle
        int ix_start =
            static_cast<int>(std::ceil((xmin_local - xmin_global - 0.5 * dxp) / dxp));
        int ix_end =
            static_cast<int>(std::floor((xmax_local - xmin_global - 0.5 * dxp) / dxp));
        ix_start = std::max(0, ix_start);
        ix_end   = std::min(static_cast<int>(nxp_global - 1), ix_end);

        int iy_start = static_cast<int>(std::ceil((y_low - ymin_global - 0.5 * dyp) / dyp));
        int iy_end   = static_cast<int>(std::floor((y_high - ymin_global - 0.5 * dyp) / dyp));
        iy_start = std::max(0, iy_start);
        iy_end   = std::min(static_cast<int>(nyp_global - 1), iy_end);

        unsigned nxp_local = ix_end - ix_start + 1;
        unsigned nyp_local = iy_end - iy_start + 1;
        size_type nlocal   = nxp_local * nyp_local;

        // 6. Create particles
        pc->create(nlocal);

        // 7. Fill positions on the lattice (device side)
        auto R_view = pc->R.getView();

        Kokkos::parallel_for(
            "init_particle_positions", nlocal,
            KOKKOS_LAMBDA(const int i) {
                unsigned ix_local  = i % nxp_local;
                unsigned iy_local  = i / nxp_local;
                unsigned ix_global = ix_start + ix_local;
                unsigned iy_global = iy_start + iy_local;

                // double jitter_x = alvine::sinusoidalPositionJitter(dxp, ix_global,
                // iy_global, 0);
                // double jitter_y = alvine::sinusoidalPositionJitter(dyp, ix_global,
                // iy_global, 1);
                double jitter_x = 0.0;
                double jitter_y = 0.0;

                double x = xmin_global + (ix_global + 0.5) * dxp + jitter_x;
                double y = ymin_global + (iy_global + 0.5) * dyp + jitter_y;

                R_view(i)[0] = x;
                R_view(i)[1] = y;
            });

        // 9. Particle circulation strength (2D VIC)
        auto omega_view = pc->omega.getView();
        double omega0   = 1.0;       // physical vorticity amplitude
        double Ap       = dxp * dyp; // particle area

        Kokkos::parallel_for(
            "init_particle_vorticity", nlocal,
            KOKKOS_LAMBDA(const int i) {
                unsigned ix_local  = i % nxp_local;
                unsigned iy_local  = i / nxp_local;
                unsigned ix_global = ix_start + ix_local;
                unsigned iy_global = iy_start + iy_local;

                double x = xmin_global + (ix_global + 0.5) * dxp;
                double y = ymin_global + (iy_global + 0.5) * dyp;

                omega_view(i) =
                    omega0 * 2.0 * cos(x) * cos(y) * Ap; // Taylor-Green circulation.
            });

        Kokkos::fence();
    }

    void clearParticles() {
        auto pc = this->pcontainer_m;
        size_type nlocal = pc->getLocalNum();

        if (nlocal == 0) {
            return;
        }

        Kokkos::View<bool*> invalid("invalid_particles", nlocal);

        Kokkos::parallel_for(
            "mark_all_particles_invalid", nlocal,
            KOKKOS_LAMBDA(const int p) {
                invalid(p) = true;
            });
        Kokkos::fence();

        pc->destroy(invalid, nlocal);
    }

    void remeshParticlesFromGrid() {
        clearParticles();

        auto* mesh = &this->fcontainer_m->getMesh();
        auto* FL = &this->fcontainer_m->getFL();
        const bool isFEM = (this->solver_m == "FEM") || (this->solver_m == "FEM_PRECON");
        ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>> rlayout(*FL, *mesh, isFEM);
        auto local = FL->getLocalNDIndex();

        unsigned nxp_global = static_cast<unsigned>(std::sqrt(this->np_m));
        unsigned nyp_global = this->np_m / nxp_global;

        double xmin_global = this->rmin_m[0];
        double xmax_global = this->rmax_m[0];
        double ymin_global = this->rmin_m[1];
        double ymax_global = this->rmax_m[1];

        double dxp = (xmax_global - xmin_global) / nxp_global;
        double dyp = (ymax_global - ymin_global) / nyp_global;

        int local_start_x = local[0].first();
        int local_end_x   = local[0].last();
        int local_start_y = local[1].first();
        int local_end_y   = local[1].last();

        double xmin_local = xmin_global + local_start_x * this->hr_m[0];
        double xmax_local = xmin_global + (local_end_x + 1) * this->hr_m[0];
        double ymin_local = ymin_global + local_start_y * this->hr_m[1];
        double ymax_local = ymin_global + (local_end_y + 1) * this->hr_m[1];

        double y_low  = std::max(ymin_local, ymin_global);
        double y_high = std::min(ymax_local, ymax_global);
        if (y_low >= y_high) {
            return;
        }

        int ix_start =
            static_cast<int>(std::ceil((xmin_local - xmin_global - 0.5 * dxp) / dxp));
        int ix_end =
            static_cast<int>(std::floor((xmax_local - xmin_global - 0.5 * dxp) / dxp));
        ix_start = std::max(0, ix_start);
        ix_end   = std::min(static_cast<int>(nxp_global - 1), ix_end);

        int iy_start = static_cast<int>(std::ceil((y_low - ymin_global - 0.5 * dyp) / dyp));
        int iy_end   = static_cast<int>(std::floor((y_high - ymin_global - 0.5 * dyp) / dyp));
        iy_start = std::max(0, iy_start);
        iy_end   = std::min(static_cast<int>(nyp_global - 1), iy_end);

        unsigned nxp_local = ix_end - ix_start + 1;
        unsigned nyp_local = iy_end - iy_start + 1;
        size_type nlocal   = nxp_local * nyp_local;
        const int nghost = this->fcontainer_m->getOmegaField().getNghost();

        auto pc = this->pcontainer_m;
        pc->create(nlocal);

        auto R_view     = pc->R.getView();
        auto R_old_view = pc->R_old.getView();
        auto omega_p    = pc->omega.getView();
        auto omega_g    = this->fcontainer_m->getOmegaField().getView();
        auto P_view     = pc->P.getView();
        auto u_g        = this->fcontainer_m->getUField().getView();

        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> hr   = this->hr_m;
        double Ap = dxp * dyp;

        Kokkos::parallel_for(
            "remesh_vic_particles_from_grid",
            nlocal,
            KOKKOS_LAMBDA(const int p) {
                unsigned ix_local  = p % nxp_local;
                unsigned iy_local  = p / nxp_local;
                unsigned ix_global = ix_start + ix_local;
                unsigned iy_global = iy_start + iy_local;

                double x = xmin_global + (ix_global + 0.5) * dxp;
                double y = ymin_global + (iy_global + 0.5) * dyp;

                int grid_i = static_cast<int>(Kokkos::floor((x - rmin[0]) / hr[0]));
                int grid_j = static_cast<int>(Kokkos::floor((y - rmin[1]) / hr[1]));
                grid_i = grid_i < local_start_x ? local_start_x : grid_i;
                grid_i = grid_i > local_end_x ? local_end_x : grid_i;
                grid_j = grid_j < local_start_y ? local_start_y : grid_j;
                grid_j = grid_j > local_end_y ? local_end_y : grid_j;

                int li = grid_i - local_start_x + nghost;
                int lj = grid_j - local_start_y + nghost;

                R_view(p)[0] = x;
                R_view(p)[1] = y;
                R_old_view(p) = R_view(p);

                omega_p(p) = omega_g(li, lj) * Ap;
                P_view(p) = u_g(li, lj);
            });
        Kokkos::fence();

        bootstrap_next_push_m = true;
    }

    void logEnergyDiagnostics() {
        double energy = this->computeKineticEnergy();

        if (!this->energy_initialized_m) {
            this->energy0_m            = energy;
            this->energy_initialized_m = true;

            if (ippl::Comm->rank() == 0) {
                std::ofstream out(this->diagnosticFileName("energy.csv"), std::ios::out);
                out << "method,dt,step,time,energy,rel_error,normalized_energy\n";
            }
            ippl::Comm->barrier();
        }

        double relErr = this->relativeError(energy, this->energy0_m);
        double normalizedEnergy =
            energy / (std::fabs(this->energy0_m) > 1e-30 ? this->energy0_m : 1e-30);

        if (ippl::Comm->rank() == 0) {
            Inform m("energy ");
            m << "kinetic energy = " << energy << ", relError = " << relErr
              << ", normalizedEnergy = " << normalizedEnergy << endl;

            std::ofstream out(this->diagnosticFileName("energy.csv"), std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << energy << "," << relErr << "," << normalizedEnergy
                << "\n";
        }
    }

    void logEnstrophyDiagnostics() {
        double enstrophy = this->computeEnstrophy();

        if (!this->enstrophy_initialized_m) {
            this->enstrophy0_m            = enstrophy;
            this->enstrophy_initialized_m = true;

            if (ippl::Comm->rank() == 0) {
                std::ofstream out(this->diagnosticFileName("enstrophy.csv"), std::ios::out);
                out << "method,dt,step,time,enstrophy,rel_error\n";
            }
            ippl::Comm->barrier();
        }

        double relErr = this->relativeError(enstrophy, this->enstrophy0_m);

        if (ippl::Comm->rank() == 0) {
            Inform m("enstrophy ");
            m << "enstrophy = " << enstrophy << ", relError = " << relErr << endl;

            std::ofstream out(this->diagnosticFileName("enstrophy.csv"), std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << enstrophy << "," << relErr << "\n";
        }
    }

    void logDivergenceDiagnostics() {
        double divL2 = this->computeDivergenceL2();

        if (ippl::Comm->rank() == 0) {
            Inform m("divergence ");
            m << "L2 = " << divL2 << endl;

            std::ofstream out(this->diagnosticFileName("divergence.csv"), std::ios::app);

            if (this->it_m == 0) {
                out << "method,dt,step,time,div_l2\n";
            }

            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            out << this->method_m << "," << this->dt_m << "," << this->it_m << ","
                << this->time_m << "," << divL2 << "\n";
        }
    }
    void advance() override {
        if (this->useRK4()) {
            RK4Step();
        } else {
            LeapFrogStep();
        }
    }

    void computeParticleVelocityFromGridVorticity(bool diagnostics) {
        static IpplTimings::TimerRef PTimer     = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("par2grid");
        static IpplTimings::TimerRef grid2parTimer = IpplTimings::getTimer("grid2par");

        IpplTimings::startTimer(par2gridTimer);
        this->par2grid();
        IpplTimings::stopTimer(par2gridTimer);

        if (diagnostics) {
            this->logTgvVorticityDiagnostics();
        }

        auto omega_n = this->fcontainer_m->getOmegaField().deepCopy();

        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        IpplTimings::startTimer(PTimer);
        this->computeVelocityField();
        if (diagnostics) {
            logEnergyDiagnostics();
            Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega_n.getView());
            logEnstrophyDiagnostics();
            this->logCirculationDiagnostics(this->computeParticleCirculation());
            logDivergenceDiagnostics();
        }
        IpplTimings::stopTimer(PTimer);

        IpplTimings::startTimer(grid2parTimer);
        this->grid2par();
        IpplTimings::stopTimer(grid2parTimer);
    }

    void RK4Step() {
        static IpplTimings::TimerRef RTimer = IpplTimings::getTimer("rk4PushPosition");
        static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("par2grid");

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        const T dt = this->dt_m;

        pc->rk4_R0 = pc->R;

        computeParticleVelocityFromGridVorticity(true);
        pc->rk4_k1 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R = pc->rk4_R0 + (0.5 * dt) * pc->rk4_k1;
        IpplTimings::stopTimer(RTimer);
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeParticleVelocityFromGridVorticity(false);
        pc->rk4_k2 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R = pc->rk4_R0 + (0.5 * dt) * pc->rk4_k2;
        IpplTimings::stopTimer(RTimer);
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeParticleVelocityFromGridVorticity(false);
        pc->rk4_k3 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R = pc->rk4_R0 + dt * pc->rk4_k3;
        IpplTimings::stopTimer(RTimer);
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        computeParticleVelocityFromGridVorticity(false);
        pc->rk4_k4 = pc->P;

        IpplTimings::startTimer(RTimer);
        pc->R_old = pc->rk4_R0;
        pc->R = pc->rk4_R0 + (dt / 6.0) *
                              (pc->rk4_k1 + 2.0 * pc->rk4_k2 + 2.0 * pc->rk4_k3 + pc->rk4_k4);
        bootstrap_next_push_m = true;
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        if (remesh_freq_m > 0 && (this->it_m + 1) % remesh_freq_m == 0) {
            IpplTimings::startTimer(par2gridTimer);
            this->par2grid();
            remeshParticlesFromGrid();
            IpplTimings::stopTimer(par2gridTimer);
            bootstrap_next_push_m = true;
        }
    }

    void LeapFrogStep() {
        static IpplTimings::TimerRef PTimer      = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer      = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef SolveTimer  = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef par2gridTimer = IpplTimings::getTimer("par2grid");
        static IpplTimings::TimerRef grid2parTimer = IpplTimings::getTimer("grid2par");

        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;

        // scatter the vorticity to the underlying grid
        IpplTimings::startTimer(par2gridTimer);
        this->par2grid();
        IpplTimings::stopTimer(par2gridTimer);
        this->logTgvVorticityDiagnostics();
        auto omega_n = this->fcontainer_m->getOmegaField().deepCopy();

        // claculate stream function
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // calculate velocity from stream function
        IpplTimings::startTimer(PTimer);
        this->computeVelocityField();
        logEnergyDiagnostics();
        Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega_n.getView());
        logEnstrophyDiagnostics();
        this->logCirculationDiagnostics(this->computeParticleCirculation());
        logDivergenceDiagnostics();
        IpplTimings::stopTimer(PTimer);

        // gather velocity field
        IpplTimings::startTimer(grid2parTimer);
        this->grid2par();
        IpplTimings::stopTimer(grid2parTimer);

        // drift
        IpplTimings::startTimer(RTimer);
        if (this->it_m == 0 || bootstrap_next_push_m) {
            pc->R_old = pc->R;
            pc->R     = pc->R + pc->P * this->dt_m;
            bootstrap_next_push_m = false;
        } else {
            typename ippl::ParticleBase<
                ippl::ParticleSpatialLayout<T, Dim>>::particle_position_type R_old_temp =
                pc->R_old;

            pc->R_old = pc->R;
            pc->R     = R_old_temp + 2 * pc->P * this->dt_m;
        }
        IpplTimings::stopTimer(RTimer);

        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);

        if (remesh_freq_m > 0 && (this->it_m + 1) % remesh_freq_m == 0) {
            IpplTimings::startTimer(par2gridTimer);
            this->par2grid();
            remeshParticlesFromGrid();
            IpplTimings::stopTimer(par2gridTimer);
        }
    }
#include <memory>
#include <fstream>
#include <sstream>

    void dumpParticleDataPerRank() {
        auto pc = this->pcontainer_m;

        auto R_host     = pc->R.getHostMirror();
        auto omega_host = pc->omega.getHostMirror();

        Kokkos::deep_copy(R_host, pc->R.getView());
        Kokkos::deep_copy(omega_host, pc->omega.getView());

        std::stringstream fname;
        fname << this->diagnosticFileName("particles_rank_") << ippl::Comm->rank() << ".csv";

        bool write_header = (this->it_m == 1);

        std::ofstream csvout;
        if (write_header) {
            csvout.open(fname.str(), std::ios::out);
        } else {
            csvout.open(fname.str(), std::ios::app);
        }

        if constexpr (Dim == 2) {
            if (write_header) {
                csvout << "time,index,pos_x,pos_y,vorticity\n";
            }

            for (size_type i = 0; i < pc->getLocalNum(); i++) {
                csvout << this->it_m << "," << i << "," << R_host(i)[0] << ","
                       << R_host(i)[1] << "," << omega_host(i) << "\n";
            }
        } else {
            if (write_header) {
                csvout << "time,index,pos_x,pos_y,pos_z\n";
            }

            for (size_type i = 0; i < pc->getLocalNum(); i++) {
                csvout << this->it_m << "," << i << "," << R_host(i)[0] << ","
                       << R_host(i)[1] << "," << R_host(i)[2] << "\n";
            }
        }

        csvout.close();
        ippl::Comm->barrier();
    }

    void dump() override {
        static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("vtkDump");
        IpplTimings::startTimer(dumpTimer);

        this->par2grid();
        auto omega_current = this->fcontainer_m->getOmegaField().deepCopy();
        this->fsolver_m->runSolver();
        this->computeVelocityField();
        Kokkos::deep_copy(this->fcontainer_m->getOmegaField().getView(), omega_current.getView());

        alvine::vtk::writeScalarField2D("data/VortexInCell", "omega",
                                        this->fcontainer_m->getOmegaField(), this->rmin_m,
                                        this->hr_m, this->it_m);

        IpplTimings::stopTimer(dumpTimer);
    }

    void logOmegaField() {
        this->par2grid();
        alvine::vtk::writeScalarFieldCsv2D("data/VortexInCell/omega_csv", "omega",
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

    /*  void dump() override {
        static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("dump");
        IpplTimings::startTimer(dumpTimer);
        dumpParticleDataPerRank();
        IpplTimings::stopTimer(dumpTimer);

      }*/

};
#endif
