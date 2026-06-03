#ifndef IPPL_ALVINE_MANAGER_H
#define IPPL_ALVINE_MANAGER_H

#include <cmath>
#include <memory>
#include <stdexcept>

#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
#include "LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"
#include "FFT/Transform/Transform.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

template <typename T, unsigned Dim>
class AlvineManager
    : public ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                              LoadBalancer<T, Dim>> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;
    using Base= ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using RealField_t = Field<T, Dim>;
    using Nufft_t = ippl::FFT<ippl::NUFFTransform, RealField_t>;
    using ComplexField_t = typename Nufft_t::ComplexField;

    std::shared_ptr<Nufft_t> nufftType1_mp;
    std::shared_ptr<Nufft_t> nufftType2_mp;

    ComplexField_t omega_hat_m;
    ComplexField_t ux_hat_m;
    ComplexField_t uy_hat_m;


protected:
    unsigned nt_m;
    unsigned it_m;
    Vector_t<int, Dim> nr_m;
    unsigned np_m;
    std::array<bool, Dim> decomp_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;
    std::string solver_m;
    int dump_freq_m;

public:
    AlvineManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, std::string& solver_, int dump_freq_)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>, LoadBalancer<T, Dim>>() 
        , nt_m(nt_)
        , nr_m(nr_)
        , np_m(np_)
        , solver_m(solver_)
	, dump_freq_m(dump_freq_) {}

    ~AlvineManager(){}

protected:
    double time_m;
    double dt_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> origin_m;
    Vector_t<double, Dim> hr_m;
    double energy0_m = 0.0;
    bool energy_initialized_m = false;
    double enstrophy0_m = 0.0;
    bool enstrophy_initialized_m = false;
public:

    double getTime() { return time_m; }

    void setTime(double time_) { time_m = time_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    virtual void dump() { /* default does nothing */ };

    void pre_step() override {
    }

    void post_step() override {
      Inform m("Step: ");
      this->time_m += this->dt_m;
      this->it_m++;

      if(this->it_m % dump_freq_m == 0) {
      	this->dump();
      }
      m << this->it_m << " Done" << endl;
    }
    void initNUFFT(double tol = 1e-10) {
      ippl::ParameterList p1, p2;

      p1.add("tolerance", tol);
      p2.add("tolerance", tol);

      // 2D currently uses native NUFFT path. FINUFFT path is 3D-only here.
      p1.add("use_finufft", false);
      p2.add("use_finufft", false);
      p1.add("use_upsampled_inputs", false);
      p2.add("use_upsampled_inputs", false);
      p1.add("spread_method", "tiled");
      p2.add("gather_method", "atomic_sort");

      auto& FL = this->fcontainer_m->getFL();
      auto& mesh = this->fcontainer_m->getMesh();

      omega_hat_m.initialize(mesh, FL);
      ux_hat_m.initialize(mesh, FL);
      uy_hat_m.initialize(mesh, FL);

      nufftType1_mp = std::make_shared<Nufft_t>(
          FL, this->pcontainer_m->getLocalNum(), 1, p1);

      nufftType2_mp = std::make_shared<Nufft_t>(
          FL, this->pcontainer_m->getLocalNum(), 2, p2);
    }

    void grid2par() override { 
	gatherCIC(); 
    }

    void gatherCIC() {
      this->pcontainer_m->P = 0.0;
      gather(this->pcontainer_m->P, this->fcontainer_m->getUField(), this->pcontainer_m->R);
    }

    void spectralScatter() {
      if constexpr (Dim == 2) {
        if (!nufftType1_mp) {
          throw std::runtime_error("AlvineManager::spectralScatter called before initNUFFT");
        }

        omega_hat_m = Kokkos::complex<T>(0.0, 0.0);
        nufftType1_mp->transform(
            this->pcontainer_m->R,
            this->pcontainer_m->omega,
            omega_hat_m);
      } else {
        throw std::runtime_error("AlvineManager::spectralScatter is implemented for 2D VIC only");
      }
    }

    void computeSpectralVelocityModes() {
      if constexpr (Dim == 2) {
        auto omega = omega_hat_m.getView();
        auto ux    = ux_hat_m.getView();
        auto uy    = uy_hat_m.getView();

        auto& layout = omega_hat_m.getLayout();
        auto& mesh   = omega_hat_m.get_mesh();

        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = omega_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const T Lx   = dx[0] * Nx;
        const T Ly   = dx[1] * Ny;
        const T area = Lx * Ly;

        const T twoPi = T(2.0 * std::acos(-1.0));
        const Kokkos::complex<T> imag(0.0, 1.0);

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "compute_spectral_velocity_modes",
            policy_type({nghost, nghost},
                        {static_cast<int>(omega.extent(0)) - nghost,
                         static_cast<int>(omega.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              const int gx = i - nghost + lDom[0].first();
              const int gy = j - nghost + lDom[1].first();

              const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
              const int my = (gy <= Ny / 2) ? gy : gy - Ny;

              const bool notMidX = (gx != Nx / 2);
              const bool notMidY = (gy != Ny / 2);

              const T kx = notMidX * twoPi * mx / Lx;
              const T ky = notMidY * twoPi * my / Ly;
              const T k2 = kx * kx + ky * ky;

              if (k2 == T(0)) {
                ux(i, j) = Kokkos::complex<T>(0.0, 0.0);
                uy(i, j) = Kokkos::complex<T>(0.0, 0.0);
              } else {
                const auto psi = omega(i, j) / (area * k2);
                ux(i, j) = -imag * ky * psi;
                uy(i, j) =  imag * kx * psi;
              }
            });
      } else {
        throw std::runtime_error(
            "AlvineManager::computeSpectralVelocityModes is implemented for 2D VIC only");
      }
    }

    void spectralGather() {
      if constexpr (Dim == 2) {
          if (!nufftType2_mp) {
              throw std::runtime_error("spectralGather called before initNUFFT");
          }

          auto& pc = *this->pcontainer_m;

          pc.ux = 0.0;
          pc.uy = 0.0;

          nufftType2_mp->transform(pc.R, pc.ux, ux_hat_m);
          nufftType2_mp->transform(pc.R, pc.uy, uy_hat_m);

          auto P = pc.P.getView();
          auto ux = pc.ux.getView();
          auto uy = pc.uy.getView();
          const auto n = pc.getLocalNum();

          Kokkos::parallel_for(
              "pack_spectral_velocity",
              n,
              KOKKOS_LAMBDA(const size_t i) {
                  P(i)[0] = ux(i);
                  P(i)[1] = uy(i);
              });
          Kokkos::fence();
      } else {
          throw std::runtime_error("AlvineManager::spectralGather is implemented for 2D VIC only");
      }
    }

    void spectralSolveParticles() {
      spectralScatter();
      computeSpectralVelocityModes();
      spectralGather();
    }


    void par2grid() override {
	scatterCIC(); 
    }

    void computeVelocityField() {

      VField_t<T, Dim> u_field = this->fcontainer_m->getUField();
      u_field = 0.0;

      if constexpr (Dim == 2) {
        const int nghost = u_field.getNghost();
        auto view = u_field.getView();

        auto omega_view = this->fcontainer_m->getOmegaField().getView();
        this->fcontainer_m->getOmegaField().fillHalo();

        Vector_t<double, Dim> hr = hr_m;
        Kokkos::parallel_for(
            "Assign rhs", ippl::getRangePolicy(view, nghost),
            KOKKOS_LAMBDA(const int i, const int j) {
                view(i, j) = {
                    (omega_view(i, j + 1) - omega_view(i, j - 1)) / (2 * hr(1)), 
                    -(omega_view(i + 1, j) - omega_view(i - 1, j)) / (2 * hr(0))
                };

            });
      } else if constexpr (Dim == 3) {
        //TODO compute velocity field in 3D, this should be a simple curl operation (one line)
      }
    }

double relativeError(double value, double reference) const {
    return std::fabs((value - reference) / std::max(std::fabs(reference), 1e-30));
}

double computeParticleCirculation() {
    double gamma_local = 0.0;

    auto omega_view = this->pcontainer_m->omega.getView();
    auto nlocal = this->pcontainer_m->getLocalNum();

    Kokkos::parallel_reduce(
        "particle_circulation",
        nlocal,
        KOKKOS_LAMBDA(const int i, double& lsum) {
            lsum += omega_view(i);
        },
        gamma_local
    );

    double gamma_global = 0.0;
    ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

    return gamma_global;
}


double computeGridCirculation() {
    double gamma_local = 0.0;

    auto& omegaField = this->fcontainer_m->getOmegaField();
    auto omega_view = omegaField.getView();

    const double dA = hr_m[0] * hr_m[1];
    const int nghost = omegaField.getNghost();

    Kokkos::parallel_reduce(
        "grid_circulation",
        ippl::getRangePolicy(omega_view, nghost),
        KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
            lsum += omega_view(i, j);
        },
        gamma_local
    );

    gamma_local *= dA;

    double gamma_global = 0.0;
    ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

    return gamma_global;
}

void checkCirculationConservation(double relError, Inform& m) {
    size_type TotalParticles = 0;
    size_type localParticles = this->pcontainer_m->getLocalNum();

    ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

    if (ippl::Comm->rank() == 0) {
        if (TotalParticles != np_m || relError > 1e-12) {
            m << "Time step: " << it_m << endl;
            m << "Total particles expected: " << np_m
              << " after update: " << TotalParticles << endl;
            m << "Rel. error in circulation conservation: " << relError << endl;
            ippl::Comm->abort();
        }
    }
}


double computeKineticEnergy() {
    double energy_local = 0.0;

    auto& uField = this->fcontainer_m->getUField();
    auto u_view = uField.getView();

    const double dA = hr_m[0] * hr_m[1];
    const int nghost = uField.getNghost();

    Kokkos::parallel_reduce(
        "kinetic_energy",
        ippl::getRangePolicy(u_view, nghost),
        KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
            const double ux = u_view(i, j)[0];
            const double uy = u_view(i, j)[1];
            lsum += 0.5 * (ux * ux + uy * uy);
        },
        energy_local
    );

    energy_local *= dA;

    double energy_global = 0.0;
    ippl::Comm->reduce(energy_local, energy_global, 1, std::plus<double>());

    return energy_global;
}

void checkEnergyConservation(double energy, double relError, Inform& m) {
    if (ippl::Comm->rank() == 0) {
        m << "kinetic energy = " << energy
          << ", relError = " << relError << endl;
    }
}

double computeEnstrophy() {
    double enstrophy_local = 0.0;

    auto& omegaField = this->fcontainer_m->getOmegaField();
    auto omega_view = omegaField.getView();
    const double dA = hr_m[0] * hr_m[1];

    const int nghost = omegaField.getNghost();

    Kokkos::parallel_reduce(
        "enstrophy",
        ippl::getRangePolicy(omega_view, nghost),
        KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
            const double omega = omega_view(i, j);
            lsum += 0.5 * omega * omega;
        },
        enstrophy_local
    );

    enstrophy_local *= dA;

    double enstrophy_global = 0.0;
    ippl::Comm->reduce(enstrophy_local, enstrophy_global, 1, std::plus<double>());

    return enstrophy_global;
}



double computeDivergenceL2() {
    auto& uField = this->fcontainer_m->getUField();
//    uField.fillHalo();

    auto divField = this->fcontainer_m->getOmegaField().deepCopy();

    divField = div(uField);
    double N = this->nr_m[0]*this->nr_m[1];
    double div_l2 = norm(divField, 2)/std::sqrt(N);

    // restore omega by recomputing par2grid later if needed
    return div_l2;
}


void scatterCIC() {
    Inform m("scatter ");

    this->fcontainer_m->getOmegaField() = 0.0;

    if constexpr (Dim == 2) {
        // Scatter particle strengths to grid
        scatter(this->pcontainer_m->omega,
                this->fcontainer_m->getOmegaField(),
                this->pcontainer_m->R);

        // Convert deposited circulation to vorticity density
        this->fcontainer_m->getOmegaField() =
            this->fcontainer_m->getOmegaField() / (hr_m[0] * hr_m[1]);

        // Conservation check
        double gammaParticles = computeParticleCirculation();
        double gammaGrid      = computeGridCirculation();

        double relError = std::fabs((gammaParticles - gammaGrid) /
                                    std::max(std::fabs(gammaParticles), 1e-30));

        m << "particle circulation = " << gammaParticles
          << ", grid circulation = " << gammaGrid
          << ", relError = " << relError << endl;

        checkCirculationConservation(relError, m);

    } else if constexpr (Dim == 3) {
        // TODO 3D version
    }
}
};
#endif
