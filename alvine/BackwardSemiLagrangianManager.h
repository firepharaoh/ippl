#ifndef IPPL_BACKWARD_SEMI_LAGRANGIAN_MANAGER_H
#define IPPL_BACKWARD_SEMI_LAGRANGIAN_MANAGER_H

#include <memory>
#include <fstream>

#include "AlvineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"

template <typename T, unsigned Dim>
class BackwardSemiLagrangianManager : public AlvineManager<T, Dim> {
public:
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t    = FieldSolver<T, Dim>;

    BackwardSemiLagrangianManager(
        unsigned nt_,
        Vector_t<int, Dim>& nr_,
        unsigned np_,
        std::string& solver_,
        int dump_freq_,
        Vector_t<double, Dim> rmin_ = 0.0,
        Vector_t<double, Dim> rmax_ = 10.0,
        Vector_t<double, Dim> origin_ = 0.0
    )
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_) {
        this->rmin_m  = rmin_;
        this->rmax_m  = rmax_;
        this->origin_m = origin_;
    }

    ~BackwardSemiLagrangianManager() {}

    void pre_run() override {
        // 1. Define domain
        for (unsigned d = 0; d < Dim; d++) {
            this->domain_m[d] = ippl::Index(this->nr_m[d]);
        }

        Vector_t<double, Dim> dr = this->rmax_m - this->rmin_m;
        this->hr_m = dr / this->nr_m;

        this->dt_m   = 0.05;
        this->it_m   = 0;
        this->time_m = 0.0;

        this->decomp_m.fill(true);
        this->isAllPeriodic_m = true;

        // 2. Create field container
        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m,
            this->rmin_m,
            this->rmax_m,
            this->decomp_m,
            this->domain_m,
            this->origin_m,
            this->isAllPeriodic_m
        ));

        this->fcontainer_m->initializeFields();

        // 3. Initialize grid vorticity directly
        initializeGridVorticity();

        // 4. Create Poisson solver
        this->setFieldSolver(std::make_shared<FieldSolver_t>(
            this->solver_m,
            &this->fcontainer_m->getOmegaField()
        ));

        this->fsolver_m->initSolver();

        // 5. Initial solve + velocity
        this->fsolver_m->runSolver();
        this->computeVelocityField();

        // 6. Initial diagnostics
        logDiagnostics();
    }

    void advance() override {
        BackwardSemiLagrangianStep();
    }

private:

    void initializeGridVorticity() {
        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega_view  = omegaField.getView();

        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> hr   = this->hr_m;

        auto localND = this->fcontainer_m->getFL().getLocalNDIndex();

        int i0 = localND[0].first();
        int i1 = localND[0].last();
        int j0 = localND[1].first();
        int j1 = localND[1].last();

        Kokkos::parallel_for(
            "initialize_grid_vorticity",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i0, j0}, {i1 + 1, j1 + 1}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double x = rmin[0] + (i + 0.5) * hr[0];
                double y = rmin[1] + (j + 0.5) * hr[1];

                // TODO: replace with your initial vortex/band condition
                omega_view(i, j) = initialOmega(x, y);
            }
        );
    }

    KOKKOS_INLINE_FUNCTION
    static T initialOmega(const T x, const T y) {
        return 1.0;
    }

    void BackwardSemiLagrangianStep() {
        /*
            BSL algorithm:

            1. omega^n is already on the grid

            2. Solve Poisson equation:
                   omega -> stream function

            3. Compute velocity field:
                   u = d psi / dy
                   v = - d psi / dx

            4. For each grid point (x_i, y_j):
                   x* = x_i - u(x_i,y_j) dt
                   y* = y_j - v(x_i,y_j) dt

            5. Interpolate old omega at (x*, y*) using CIC/bilinear interpolation

            6. Store result:
                   omega^{n+1}(x_i,y_j) = omega^n(x*, y*)
        */

        // A. Solve Poisson
        this->fsolver_m->runSolver();

        // B. Compute velocity
        this->computeVelocityField();

        // C. Advect vorticity backward
        advectVorticityBSL();

        // D. Diagnostics
        logDiagnostics();
    }

    void advectVorticityBSL() {
        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto& uField     = this->fcontainer_m->getUField();

        omegaField.fillHalo();
        uField.fillHalo();

        /*
            IMPORTANT:
            You need two omega fields:

            omegaOld = omega^n
            omegaNew = omega^{n+1}

            Do NOT overwrite omega while still interpolating from it.

            For now this is only a skeleton.
            Best long-term solution:
            add omegaOld / omegaTmp field to FieldContainer.
        */

        // TODO:
        // 1. Copy omegaField into temporary old field
        // 2. For each grid node/cell center:
        //      compute x_i, y_j
        //      read u, v
        //      compute departure point x*, y*
        //      apply periodic wrapping
        //      interpolate omegaOld at x*, y*
        //      store in omegaField(i,j)
auto omega_old = omegaOldField.getView();   // read only
auto omega_new = omegaNewField.getView();   // write only
auto u_view    = uField.getView();

double dt = this->dt_m;
auto rmin = this->rmin_m;
auto rmax = this->rmax_m;
auto hr   = this->hr_m;
auto nr   = this->nr_m;

Kokkos::parallel_for(
    "backward_tracing_step",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i0, j0}, {i1 + 1, j1 + 1}),
    KOKKOS_LAMBDA(const int i, const int j) {
        double x = rmin[0] + (i + 0.5) * hr[0];
        double y = rmin[1] + (j + 0.5) * hr[1];

        double u = u_view(i, j)[0];
        double v = u_view(i, j)[1];

        double x_dep = x - u * dt;
        double y_dep = y - v * dt;

        // periodic wrapping
        double Lx = rmax[0] - rmin[0];
        double Ly = rmax[1] - rmin[1];

        while (x_dep < rmin[0]) x_dep += Lx;
        while (x_dep >= rmax[0]) x_dep -= Lx;
        while (y_dep < rmin[1]) y_dep += Ly;
        while (y_dep >= rmax[1]) y_dep -= Ly;

        // convert physical departure point to cell index coordinate
        double gx = (x_dep - rmin[0]) / hr[0] - 0.5;
        double gy = (y_dep - rmin[1]) / hr[1] - 0.5;

        int i0_dep = floor(gx);
        int j0_dep = floor(gy);

        double alpha = gx - i0_dep;
        double beta  = gy - j0_dep;

        int i1_dep = i0_dep + 1;
        int j1_dep = j0_dep + 1;

        // periodic index wrapping
        if (i0_dep < 0) i0_dep += nr[0];
        if (j0_dep < 0) j0_dep += nr[1];
        if (i1_dep >= nr[0]) i1_dep -= nr[0];
        if (j1_dep >= nr[1]) j1_dep -= nr[1];

        omega_new(i, j) =
            (1.0 - alpha) * (1.0 - beta) * omega_old(i0_dep, j0_dep)
          + alpha         * (1.0 - beta) * omega_old(i1_dep, j0_dep)
          + (1.0 - alpha) * beta         * omega_old(i0_dep, j1_dep)
          + alpha         * beta         * omega_old(i1_dep, j1_dep);
    }
);
        
    }

    void applyPeriodic(T& x, T& y) {
        const T Lx = this->rmax_m[0] - this->rmin_m[0];
        const T Ly = this->rmax_m[1] - this->rmin_m[1];

        while (x < this->rmin_m[0]) x += Lx;
        while (x >= this->rmax_m[0]) x -= Lx;

        while (y < this->rmin_m[1]) y += Ly;
        while (y >= this->rmax_m[1]) y -= Ly;
    }

    void logDiagnostics() {
        // Reuse diagnostics from AlvineManager
        double gamma = this->computeGridCirculation();
        double E     = this->computeKineticEnergy();
        double Z     = this->computeEnstrophy();
        double divL2 = this->computeDivergenceL2();

        if (ippl::Comm->rank() == 0) {
            Inform m("BSL diagnostics ");
            m << "circulation = " << gamma
              << ", energy = " << E
              << ", enstrophy = " << Z
              << ", divL2 = " << divL2
              << endl;
        }
    }
};

#endif