#ifndef IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H
#define IPPL_VORTEX_IN_FOURIER_3D_MANAGER_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "core/AlvineManager.h"
#include "FieldSolver.hpp"
#include "fields/FieldContainer.hpp"
#include "particles/ParticleContainer.hpp"
#include "test_cases/TaylorGreen3D.hpp"

template <typename T>
class VortexInFourier3DManager : public AlvineManager<T, 3> {
public:
    static constexpr unsigned Dim = 3;

    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;

    VortexInFourier3DManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_,
                             std::string& solver_, int dump_freq_,
                             double dt_ = 0.05,
                             std::string method_ = "vif3d",
                             int spectral_filter_ = 0,
                             double viscosity_ = 0.0,
                             std::string time_integrator_ = "leapfrog",
                             Vector_t<double, Dim> rmin_ = 0.0,
                             Vector_t<double, Dim> rmax_ = 1.0,
                             Vector_t<double, Dim> origin_ = 0.0)
        : AlvineManager<T, Dim>(nt_, nr_, np_, solver_, dump_freq_, dt_, method_,
                                spectral_filter_, viscosity_, time_integrator_) {
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
        this->spectralSolveParticles();
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        this->reconstructSpectralVelocity(this->fcontainer_m->getUField());
    }

    void par2grid() override {}

    void grid2par() override {}

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
};

#endif
