#ifndef IPPL_PERFECT_INVISCID_2D_MANAGER_H
#define IPPL_PERFECT_INVISCID_2D_MANAGER_H

#include <memory>
#include <string>

#include "AlvineManager.h"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "ParticleContainer.hpp"
#include "VtkDump.hpp"

template <typename T, unsigned Dim>
class PerfectInviscid2DManager : public AlvineManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;

    PerfectInviscid2DManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_,
                             std::string& solver_, int dump_freq_,
                             double dt_ = 0.05,
                             std::string method_ = "perfect_2d_inviscid",
                             Vector_t<double, Dim> rmin_ = 0.0,
                             Vector_t<double, Dim> rmax_ = 10.0,
                             Vector_t<double, Dim> origin_ = 0.0)
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_, dt_, method_) {
        this->rmin_m   = rmin_;
        this->rmax_m   = rmax_;
        this->origin_m = origin_;
    }

    ~PerfectInviscid2DManager() override {}

    void pre_run() override {
        for (unsigned d = 0; d < Dim; ++d) {
            this->domain_m[d] = ippl::Index(this->nr_m[d]);
        }

        const Vector_t<double, Dim> dr = this->rmax_m - this->rmin_m;
        this->hr_m                     = dr / this->nr_m;
        this->it_m                     = 0;
        this->time_m                   = 0.0;
        this->decomp_m.fill(true);
        this->isAllPeriodic_m = true;

        this->setFieldContainer(std::make_shared<FieldContainer_t>(
            this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m,
            this->origin_m, this->isAllPeriodic_m));

        this->setParticleContainer(std::make_shared<ParticleContainer_t>(
            this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()));

        this->fcontainer_m->initializeFields();

        this->setFieldSolver(std::make_shared<FieldSolver_t>(
            this->solver_m, &this->fcontainer_m->getOmegaField()));
        this->fsolver_m->initSolver();

        exact_vorticity(this->fcontainer_m->getOmegaField());
        exact_velocity(this->fcontainer_m->getUField());
    }

    void exact_vorticity(Field<T, Dim>& omegaField) {
        auto omega_view  = omegaField.getView();

        auto localND = this->fcontainer_m->getFL().getLocalNDIndex();

        int i0           = localND[0].first();
        int i1           = localND[0].last();
        int j0           = localND[1].first();
        int j1           = localND[1].last();
        const int nghost = omegaField.getNghost();

        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> hr   = this->hr_m;

        Kokkos::parallel_for(
            "perfect_inviscid_2d_exact_vorticity",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost},
                                                   {nghost + i1 - i0 + 1,
                                                    nghost + j1 - j0 + 1}),
            KOKKOS_LAMBDA(const int li, const int lj) {
                const int i = i0 + li - nghost;
                const int j = j0 + lj - nghost;
                const double x = rmin[0] + (i + 0.5) * hr[0];
                const double y = rmin[1] + (j + 0.5) * hr[1];

                omega_view(li, lj) = TaylorGreen2D<T>::vorticity(x, y, T(0.0), T(0.0));
            });

        Kokkos::fence();
    }

    void exact_velocity(VField_t<T, Dim>& uField) {
        uField       = 0.0;

        if constexpr (Dim == 2) {
            auto u_view = uField.getView();

            auto localND = this->fcontainer_m->getFL().getLocalNDIndex();

            int i0           = localND[0].first();
            int i1           = localND[0].last();
            int j0           = localND[1].first();
            int j1           = localND[1].last();
            const int nghost = uField.getNghost();

            Vector_t<double, Dim> rmin = this->rmin_m;
            Vector_t<double, Dim> hr   = this->hr_m;

            Kokkos::parallel_for(
                "perfect_inviscid_2d_exact_velocity",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost},
                                                       {nghost + i1 - i0 + 1,
                                                        nghost + j1 - j0 + 1}),
                KOKKOS_LAMBDA(const int li, const int lj) {
                    const int i = i0 + li - nghost;
                    const int j = j0 + lj - nghost;
                    const double x = rmin[0] + (i + 0.5) * hr[0];
                    const double y = rmin[1] + (j + 0.5) * hr[1];

                    u_view(li, lj) = {TaylorGreen2D<T>::velocityX(x, y, T(0.0), T(0.0)),
                                      TaylorGreen2D<T>::velocityY(x, y, T(0.0), T(0.0))};
                });

            Kokkos::fence();
        }
    }

    void advance() override {}

    void par2grid() override {}

    void grid2par() override {}

  void dump() override {
      static IpplTimings::TimerRef dumpTimer = IpplTimings::getTimer("vtkDump");
      IpplTimings::startTimer(dumpTimer);

      alvine::vtk::writeScalarField2D(
          "data/PerfectInviscid2D",
          "omega",
          this->fcontainer_m->getOmegaField(),
          this->rmin_m,
          this->hr_m,
          this->it_m);

      IpplTimings::stopTimer(dumpTimer);
  }

};

#endif
