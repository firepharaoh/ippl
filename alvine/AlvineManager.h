#ifndef IPPL_ALVINE_MANAGER_H
#define IPPL_ALVINE_MANAGER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    using Base                = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using RealField_t         = Field<T, Dim>;
    using Nufft_t             = ippl::FFT<ippl::NUFFTransform, RealField_t>;
    using ComplexField_t = typename Nufft_t::ComplexField;
    using SpectralFft_t  = ippl::FFT<ippl::CCTransform, ComplexField_t>;

    struct VorticitySpectrumShell {
        double enstrophy = 0.0;
        std::uint64_t modeCount = 0;
        bool complete = false;
    };

    std::shared_ptr<Nufft_t> nufftType1_mp;
    std::shared_ptr<Nufft_t> nufftType2_mp;
    std::shared_ptr<SpectralFft_t> spectralFft_mp;

    ComplexField_t omega_hat_m; // Vorticity field in the Fourier domain
    ComplexField_t ux_hat_m; //Velocity field in the Fourier domain x
    ComplexField_t uy_hat_m; //Velocity field in the Fourier domain y
    ComplexField_t viscosity_hat_m; // Viscosity tendency in the Fourier domain
    RealField_t Sk_m; //Shape function field 

protected:
    unsigned nt_m;
    unsigned it_m;
    Vector_t<int, Dim> nr_m;
    unsigned np_m;
    std::array<bool, Dim> decomp_m;
    bool isAllPeriodic_m;
    ippl::NDIndex<Dim> domain_m;
    std::string solver_m;
    std::string method_m;
    int dump_freq_m;
    int spectral_filter_m;
    int shapedegree_m = 1;// degree of the shape function for VIF, default is 1 (linear, CIC like)
    double viscosity_m; //Viscosity for the viscous filter, default is 0.0 (no viscosity)

public:
    AlvineManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, std::string& solver_,
                  int dump_freq_, double dt_ = 0.05, std::string method_ = "alvine",
                  int spectral_filter_ = 0 , double viscosity= 0.0)
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                            LoadBalancer<T, Dim>>()
        , nt_m(nt_)
        , nr_m(nr_)
        , np_m(np_)
        , solver_m(solver_)
        , method_m(method_)
        , dump_freq_m(dump_freq_)
        , spectral_filter_m(spectral_filter_)
        , viscosity_m(viscosity)
        , dt_m(dt_) {}

    ~AlvineManager() {}

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
    double circulation0_m = 0.0;
    bool circulation_initialized_m = false;
    bool tgv_velocity_diagnostics_initialized_m = false;

public:
    double getTime() { return time_m; }

    void setTime(double time_) { time_m = time_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    bool useShapeFunctionFilter() const { return spectral_filter_m == 1; }

    bool useHouLiFilter() const { return spectral_filter_m == 2; }

    virtual void dump() { /* default does nothing */ };

    std::string sanitizeFileToken(std::string token) const {
        for (char& c : token) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        return token;
    }

    std::string dtFileToken() const {
        std::ostringstream os;
        os << std::setprecision(12) << dt_m;
        std::string token = os.str();

        if (token.find('.') != std::string::npos) {
            while (!token.empty() && token.back() == '0') {
                token.pop_back();
            }
            if (!token.empty() && token.back() == '.') {
                token.pop_back();
            }
        }

        return sanitizeFileToken(token);
    }

    std::string diagnosticFileName(const std::string& baseName) const {
        return sanitizeFileToken(method_m) + "_dt_" + dtFileToken() + "_" + baseName;
    }

    void pre_step() override {}

    void post_step() override {
        Inform m("Step: ");
        this->time_m += this->dt_m;
        this->it_m++;

        if (dump_freq_m > 0 && this->it_m % dump_freq_m == 0) {
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

        auto& FL   = this->fcontainer_m->getFL();
        auto& mesh = this->fcontainer_m->getMesh();

        omega_hat_m.initialize(mesh, FL);
        ux_hat_m.initialize(mesh, FL);
        uy_hat_m.initialize(mesh, FL);
        viscosity_hat_m.initialize(mesh, FL);
        Sk_m.initialize(mesh, FL);
        if (useShapeFunctionFilter()) {
            initializeShapeFunctionVIF();
        }

        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", true);
        spectralFft_mp = std::make_shared<SpectralFft_t>(FL, fftParams);

        nufftType1_mp =
            std::make_shared<Nufft_t>(FL, this->pcontainer_m->getLocalNum(), 1, p1);

        nufftType2_mp =
            std::make_shared<Nufft_t>(FL, this->pcontainer_m->getLocalNum(), 2, p2);
    }

    void grid2par() override {
        gatherCIC();
    }

    void gatherCIC() {
        this->pcontainer_m->P = 0.0;
        gather(this->pcontainer_m->P, this->fcontainer_m->getUField(), this->pcontainer_m->R);
    }

    void par2grid() override {
        scatterCIC();
    }

    void computeVelocityField() {
        VField_t<T, Dim> u_field = this->fcontainer_m->getUField();
        u_field = 0.0;

        if constexpr (Dim == 2) {
            const int nghost = u_field.getNghost();
            auto view        = u_field.getView();

            auto omega_view = this->fcontainer_m->getOmegaField().getView();
            this->fcontainer_m->getOmegaField().fillHalo();

            Vector_t<double, Dim> hr = hr_m;
            Kokkos::parallel_for(
                "Assign rhs", ippl::getRangePolicy(view, nghost),
                KOKKOS_LAMBDA(const int i, const int j) {
                    view(i, j) = {
                        (omega_view(i, j + 1) - omega_view(i, j - 1)) / (2 * hr(1)),
                        -(omega_view(i + 1, j) - omega_view(i - 1, j)) / (2 * hr(0))};
                });
        } else if constexpr (Dim == 3) {
            // TODO compute velocity field in 3D, this should be a simple curl operation (one line)
        }
    }

    double relativeError(double value, double reference) const {
        return std::fabs((value - reference) / std::max(std::fabs(reference), 1e-30));
    }

    double computeParticleCirculation() {
        double gamma_local = 0.0;

        auto omega_view = this->pcontainer_m->omega.getView();
        auto nlocal     = this->pcontainer_m->getLocalNum();

        Kokkos::parallel_reduce(
            "particle_circulation", nlocal,
            KOKKOS_LAMBDA(const int i, double& lsum) { lsum += omega_view(i); },
            gamma_local);

        double gamma_global = 0.0;
        ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

        return gamma_global;
    }

    double computeGridCirculation() {
        double gamma_local = 0.0;

        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega_view  = omegaField.getView();

        const double dA  = hr_m[0] * hr_m[1];
        const int nghost = omegaField.getNghost();

        Kokkos::parallel_reduce(
            "grid_circulation", ippl::getRangePolicy(omega_view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                lsum += omega_view(i, j);
            },
            gamma_local);

        gamma_local *= dA;

        double gamma_global = 0.0;
        ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

        return gamma_global;
    }

    void logCirculationDiagnostics(double circulation) {
        if (!circulation_initialized_m) {
            circulation0_m             = circulation;
            circulation_initialized_m  = true;

            if (ippl::Comm->rank() == 0) {
                std::ofstream out(diagnosticFileName("circulation.csv"), std::ios::out);
                out << "method,dt,step,time,circulation,rel_error,normalized_circulation\n";
            }
            ippl::Comm->barrier();
        }

        const double relError = relativeError(circulation, circulation0_m);
        const double normalizedCirculation =
            circulation / (std::fabs(circulation0_m) > 1e-30 ? circulation0_m : 1e-30);

        if (ippl::Comm->rank() == 0) {
            Inform m("circulation ");
            m << "circulation = " << circulation << ", relError = " << relError
              << ", normalizedCirculation = " << normalizedCirculation << endl;

            std::ofstream out(diagnosticFileName("circulation.csv"), std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << circulation << "," << relError << "," << normalizedCirculation << "\n";
        }
    }

    void checkCirculationConservation(double absError, double relError, Inform& m) {
        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            const double circulationTol = 1e-10;
            if (TotalParticles != np_m || relError > circulationTol) {
                m << "Time step: " << it_m << endl;
                m << "Total particles expected: " << np_m
                  << " after update: " << TotalParticles << endl;
                m << "Abs. error in circulation conservation: " << absError << endl;
                m << "Rel. error in circulation conservation: " << relError << endl;
                ippl::Comm->abort();
            }
        }
    }

    double computeKineticEnergy() {
        double energy_local = 0.0;

        auto& uField = this->fcontainer_m->getUField();
        auto u_view  = uField.getView();

        const double dA  = hr_m[0] * hr_m[1];
        const int nghost = uField.getNghost();

        Kokkos::parallel_reduce(
            "kinetic_energy", ippl::getRangePolicy(u_view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                const double ux = u_view(i, j)[0];
                const double uy = u_view(i, j)[1];
                lsum += 0.5 * (ux * ux + uy * uy);
            },
            energy_local);

        energy_local *= dA;

        double energy_global = 0.0;
        ippl::Comm->reduce(energy_local, energy_global, 1, std::plus<double>());

        return energy_global;
    }

    void logTgvVelocityDiagnostics(const std::string& filename = "tgv_velocity_error.csv") {
        if constexpr (Dim == 2) {
            auto& uField = this->fcontainer_m->getUField();
            auto u_view  = uField.getView();

            const auto& localND = uField.getLayout().getLocalNDIndex();
            const double dA     = hr_m[0] * hr_m[1];
            const int nghost    = uField.getNghost();

            Vector_t<double, Dim> rmin = rmin_m;
            Vector_t<double, Dim> hr   = hr_m;

            double localErr2        = 0.0;
            double localOppositeErr2 = 0.0;
            double localRef2        = 0.0;

            Kokkos::parallel_reduce(
                "tgv_velocity_error_l2",
                ippl::getRangePolicy(u_view, nghost),
                KOKKOS_LAMBDA(const int i, const int j,
                              double& err2,
                              double& oppositeErr2,
                              double& ref2) {
                    const int gx = i - nghost + localND[0].first();
                    const int gy = j - nghost + localND[1].first();

                    const double x = rmin[0] + (gx + 0.5) * hr[0];
                    const double y = rmin[1] + (gy + 0.5) * hr[1];

                    const double uxExact = -Kokkos::cos(x) * Kokkos::sin(y);
                    const double uyExact = Kokkos::sin(x) * Kokkos::cos(y);

                    const double dux = u_view(i, j)[0] - uxExact;
                    const double duy = u_view(i, j)[1] - uyExact;
                    const double duxOpposite = u_view(i, j)[0] + uxExact;
                    const double duyOpposite = u_view(i, j)[1] + uyExact;

                    err2 += dux * dux + duy * duy;
                    oppositeErr2 += duxOpposite * duxOpposite + duyOpposite * duyOpposite;
                    ref2 += uxExact * uxExact + uyExact * uyExact;
                },
                Kokkos::Sum<double>(localErr2),
                Kokkos::Sum<double>(localOppositeErr2),
                Kokkos::Sum<double>(localRef2));

            double localLinf         = 0.0;
            double localOppositeLinf = 0.0;
            double localRefLinf      = 0.0;

            Kokkos::parallel_reduce(
                "tgv_velocity_error_linf",
                ippl::getRangePolicy(u_view, nghost),
                KOKKOS_LAMBDA(const int i, const int j,
                              double& maxErr,
                              double& maxOppositeErr,
                              double& maxRef) {
                    const int gx = i - nghost + localND[0].first();
                    const int gy = j - nghost + localND[1].first();

                    const double x = rmin[0] + (gx + 0.5) * hr[0];
                    const double y = rmin[1] + (gy + 0.5) * hr[1];

                    const double uxExact = -Kokkos::cos(x) * Kokkos::sin(y);
                    const double uyExact = Kokkos::sin(x) * Kokkos::cos(y);

                    const double dux = u_view(i, j)[0] - uxExact;
                    const double duy = u_view(i, j)[1] - uyExact;
                    const double duxOpposite = u_view(i, j)[0] + uxExact;
                    const double duyOpposite = u_view(i, j)[1] + uyExact;

                    const double err = Kokkos::sqrt(dux * dux + duy * duy);
                    const double oppositeErr =
                        Kokkos::sqrt(duxOpposite * duxOpposite + duyOpposite * duyOpposite);
                    const double ref = Kokkos::sqrt(uxExact * uxExact + uyExact * uyExact);

                    if (err > maxErr) {
                        maxErr = err;
                    }
                    if (oppositeErr > maxOppositeErr) {
                        maxOppositeErr = oppositeErr;
                    }
                    if (ref > maxRef) {
                        maxRef = ref;
                    }
                },
                Kokkos::Max<double>(localLinf),
                Kokkos::Max<double>(localOppositeLinf),
                Kokkos::Max<double>(localRefLinf));

            localErr2 *= dA;
            localOppositeErr2 *= dA;
            localRef2 *= dA;

            double globalErr2         = 0.0;
            double globalOppositeErr2 = 0.0;
            double globalRef2         = 0.0;
            double globalLinf         = 0.0;
            double globalOppositeLinf = 0.0;
            double globalRefLinf      = 0.0;

            ippl::Comm->reduce(localErr2, globalErr2, 1, std::plus<double>());
            ippl::Comm->reduce(localOppositeErr2, globalOppositeErr2, 1, std::plus<double>());
            ippl::Comm->reduce(localRef2, globalRef2, 1, std::plus<double>());
            ippl::Comm->reduce(localLinf, globalLinf, 1, std::greater<double>());
            ippl::Comm->reduce(localOppositeLinf, globalOppositeLinf, 1, std::greater<double>());
            ippl::Comm->reduce(localRefLinf, globalRefLinf, 1, std::greater<double>());

            const double l2Error = std::sqrt(globalErr2);
            const double l2OppositeError = std::sqrt(globalOppositeErr2);
            const double l2Reference = std::sqrt(std::max(globalRef2, 1e-30));
            const double linfReference = std::max(globalRefLinf, 1e-30);

            if (ippl::Comm->rank() == 0) {
                const bool firstWrite = !tgv_velocity_diagnostics_initialized_m;
                std::ofstream out(diagnosticFileName(filename),
                                  firstWrite ? std::ios::out : std::ios::app);
                out.precision(16);
                out.setf(std::ios::scientific, std::ios::floatfield);

                if (firstWrite) {
                    out << "method,dt,step,time,l2_error,l2_rel_error,linf_error,linf_rel_error,"
                        << "opposite_sign_l2_error,opposite_sign_l2_rel_error,"
                        << "opposite_sign_linf_error,opposite_sign_linf_rel_error\n";
                }

                out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                    << l2Error << "," << l2Error / l2Reference << ","
                    << globalLinf << "," << globalLinf / linfReference << ","
                    << l2OppositeError << "," << l2OppositeError / l2Reference << ","
                    << globalOppositeLinf << "," << globalOppositeLinf / linfReference << "\n";

                Inform m("tgv_velocity_error ");
                m << "l2Rel = " << l2Error / l2Reference
                  << ", linfRel = " << globalLinf / linfReference
                  << ", oppositeSignL2Rel = " << l2OppositeError / l2Reference
                  << ", oppositeSignLinfRel = " << globalOppositeLinf / linfReference << endl;
            }
            tgv_velocity_diagnostics_initialized_m = true;
        }
    }

    void checkEnergyConservation(double energy, double relError, Inform& m) {
        if (ippl::Comm->rank() == 0) {
            m << "kinetic energy = " << energy << ", relError = " << relError << endl;
        }
    }

    double computeEnstrophy() {
        double enstrophy_local = 0.0;

        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega_view  = omegaField.getView();
        const double dA  = hr_m[0] * hr_m[1];

        const int nghost = omegaField.getNghost();

        Kokkos::parallel_reduce(
            "enstrophy", ippl::getRangePolicy(omega_view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                const double omega = omega_view(i, j);
                lsum += 0.5 * omega * omega;
            },
            enstrophy_local);

        enstrophy_local *= dA;

        double enstrophy_global = 0.0;
        ippl::Comm->reduce(enstrophy_local, enstrophy_global, 1, std::plus<double>());

        return enstrophy_global;
    }

    double computeDivergenceL2() {
        auto& uField = this->fcontainer_m->getUField();
        // uField.fillHalo();

        auto divField = this->fcontainer_m->getOmegaField().deepCopy();

        divField     = div(uField);
        double N     = this->nr_m[0] * this->nr_m[1];
        double div_l2 = norm(divField, 2) / std::sqrt(N);

        // restore omega by recomputing par2grid later if needed
        return div_l2;
    }

    void scatterCIC() {
        Inform m("scatter ");

        this->fcontainer_m->getOmegaField() = 0.0;

        if constexpr (Dim == 2) {
            // Scatter particle strengths to grid
            scatter(this->pcontainer_m->omega, this->fcontainer_m->getOmegaField(),
                    this->pcontainer_m->R);

            // Convert deposited circulation to vorticity density
            this->fcontainer_m->getOmegaField() =
                this->fcontainer_m->getOmegaField() / (hr_m[0] * hr_m[1]);

            // Conservation check
            double gammaParticles = computeParticleCirculation();
            double gammaGrid      = computeGridCirculation();

            const double absError = std::fabs(gammaParticles - gammaGrid);
            const double circulationScale =
                std::max(std::max(std::fabs(gammaParticles), std::fabs(gammaGrid)), 1.0);
            const double relError = absError / circulationScale;

            m << "particle circulation = " << gammaParticles
              << ", grid circulation = " << gammaGrid << ", absError = " << absError
              << ", relError = " << relError << endl;

            checkCirculationConservation(absError, relError, m);

        } else if constexpr (Dim == 3) {
            // TODO 3D version
        }
    }

    // SPECTRAL IMPLEMENTATION
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

        if (useShapeFunctionFilter()) {
            auto omegaView = omega_hat_m.getView();
            auto shapeView = Sk_m.getView();
            const int nghost = omega_hat_m.getNghost();
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for(
                "Multiply with shape function in Fourier space",
                policy_type({nghost, nghost},
                            {static_cast<int>(omegaView.extent(0)) - nghost,
                             static_cast<int>(omegaView.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j) {
                  omegaView(i,j) *= shapeView(i,j);
                });
            Kokkos::fence();
        }
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

              const T laplaceKx = twoPi * mx / Lx;
              const T laplaceKy = twoPi * my / Ly;
              const T k2 = laplaceKx * laplaceKx + laplaceKy * laplaceKy;
              const T derivativeKx = notMidX * laplaceKx;
              const T derivativeKy = notMidY * laplaceKy;

              if (k2 == T(0)) {
                omega(i, j) = Kokkos::complex<T>(0.0, 0.0);
                ux(i, j) = Kokkos::complex<T>(0.0, 0.0);
                uy(i, j) = Kokkos::complex<T>(0.0, 0.0);
              } else {
                const auto psi = omega(i, j) / (area * k2);
                ux(i, j) = imag * derivativeKy * psi;
                uy(i, j) = -imag * derivativeKx * psi;
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

            if (useShapeFunctionFilter()) {
                auto uxModes = ux_hat_m.deepCopy();
                auto uyModes = uy_hat_m.deepCopy();
                auto uxModeView = uxModes.getView();
                auto uyModeView = uyModes.getView();
                auto shapeView = Sk_m.getView();
                const int nghost = uxModes.getNghost();

                using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
                Kokkos::parallel_for(
                    "shape_spectral_velocity_modes",
                    policy_type({nghost, nghost},
                                {static_cast<int>(uxModeView.extent(0)) - nghost,
                                 static_cast<int>(uxModeView.extent(1)) - nghost}),
                    KOKKOS_LAMBDA(const int i, const int j) {
                        uxModeView(i, j) *= shapeView(i, j);
                        uyModeView(i, j) *= shapeView(i, j);
                    });
                Kokkos::fence();

                nufftType2_mp->transform(pc.R, pc.ux, uxModes);
                nufftType2_mp->transform(pc.R, pc.uy, uyModes);
            } else {
                nufftType2_mp->transform(pc.R, pc.ux, ux_hat_m);
                nufftType2_mp->transform(pc.R, pc.uy, uy_hat_m);
            }

            auto P       = pc.P.getView();
            auto ux      = pc.ux.getView();
            auto uy      = pc.uy.getView();
            const auto n = pc.getLocalNum();

            Kokkos::parallel_for(
                "pack_spectral_velocity", n,
                KOKKOS_LAMBDA(const size_t i) {
                    P(i)[0] = ux(i);
                    P(i)[1] = uy(i);
                });
            Kokkos::fence();
        } else {
            throw std::runtime_error("AlvineManager::spectralGather is implemented for 2D VIC only");
        }
    }
    void spectralGatherViscosity(ComplexField_t& visc_hat) {
        if constexpr (Dim == 2) {
            if(!nufftType2_mp){
                throw std::runtime_error("spectralGatherViscosity called before initNUFFT");
            }
            auto & pc = *this->pcontainer_m;
            pc.viscosity = 0.0;
            nufftType2_mp->transform(pc.R, pc.viscosity, visc_hat);
            Kokkos::fence();
        } else {
            throw std::runtime_error("AlvineManager::spectralGatherViscosity is implemented for 2D VIC only");
        }
    }
    void spectralSolveParticles() {
      spectralScatter();
      computeSpectralVelocityModes();
      spectralGather();
    }

    void applyCellCenteredIfftPhase(ComplexField_t& modes) {
      if constexpr (Dim == 2) {
        auto view = modes.getView();

        auto& layout       = modes.getLayout();
        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost   = modes.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const T pi   = std::acos(T(-1.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "apply_cell_centered_ifft_phase",
            policy_type({nghost, nghost},
                        {static_cast<int>(view.extent(0)) - nghost,
                         static_cast<int>(view.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              const int gx = i - nghost + lDom[0].first();
              const int gy = j - nghost + lDom[1].first();

              const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
              const int my = (gy <= Ny / 2) ? gy : gy - Ny;

              const T phase = pi * (T(mx) / T(Nx) + T(my) / T(Ny));
              const Kokkos::complex<T> factor(Kokkos::cos(phase), Kokkos::sin(phase));

              view(i, j) *= factor;
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::applyCellCenteredIfftPhase is implemented for 2D VIC only");
      }
    }

    void Hou_Li_filter(ComplexField_t& modes, double alpha = 36.0, int exponent = 36) {
      if constexpr (Dim == 2) {
        auto view = modes.getView();

        auto& layout       = modes.getLayout();
        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost   = modes.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const T kxMax = T(Nx) / T(2.0);
        const T kyMax = T(Ny) / T(2.0);
        const T invSqrtDim = T(1.0) / Kokkos::sqrt(T(2.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "hou_li_filter",
            policy_type({nghost, nghost},
                        {static_cast<int>(view.extent(0)) - nghost,
                         static_cast<int>(view.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              const int gx = i - nghost + lDom[0].first();
              const int gy = j - nghost + lDom[1].first();

              const T kx = (gx <= Nx / 2) ? T(gx) : T(gx - Nx);
              const T ky = (gy <= Ny / 2) ? T(gy) : T(gy - Ny);

              const T etaX = Kokkos::abs(kx) / kxMax;
              const T etaY = Kokkos::abs(ky) / kyMax;
              const T eta  = Kokkos::sqrt(etaX * etaX + etaY * etaY) * invSqrtDim;
              const T filterFactor =
                  Kokkos::exp(-T(alpha) * Kokkos::pow(eta, exponent));

              view(i, j) *= filterFactor;
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::Hou_Li_filter is implemented for 2D VIC only");
      }
    }

    void initializeShapeFunctionVIF() { //Source is from InitializeShapeFunctionPIF in alpine/ChargedParticles.hpp
        if constexpr (Dim == 2) {
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            auto Skview = Sk_m.getView();
            auto N = nr_m;
            const int nghost = Sk_m.getNghost();
            const auto& mesh = Sk_m.get_mesh();
            const Vector_t<T, Dim> dx = mesh.getMeshSpacing();
            const Vector_t<T, Dim> Len = rmax_m - rmin_m;
            const T pi = T(3.141592653589793238462643383279502884);
            const int order = shapedegree_m + 1;
            const auto& layout = Sk_m.getLayout();
            const auto& lDom = layout.getLocalNDIndex();
            Kokkos::parallel_for(
                "B-spline shape function initialization",
                mdrange_type({nghost, nghost},
                            {static_cast<int>(Skview.extent(0)) - nghost,
                                static_cast<int>(Skview.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    Vector<int, Dim> iVec  = {i,j};
                    for (unsigned d=0;d < Dim; d++) {
                        iVec[d] = iVec[d] - nghost+lDom[d].first();
                    }
                    Vector<double, Dim> kVec;
                    double Sk = 1.0;
                    for (unsigned d=0;d < Dim; d++) {
                        bool shift = (iVec[d] > (N[d]/2));
                        kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                        double khbytwo = kVec[d] * dx[d] / 2.0;
                        bool isNotZero = (khbytwo != 0.0);
                        double factor = (1.0 / (khbytwo +((!isNotZero) *1.0))) ;
                        double arg = isNotZero * (Kokkos::sin(khbytwo) * factor) + (!isNotZero) * 1.0;
                        //Fourier Transform of B-spline of order n is (sin(kh/2)/(kh/2))^n, where h is the mesh spacing and k is the wavenumber
                        Sk *= Kokkos::pow(arg, order);
                    }
                    Skview(i,j) = Sk;
                }
            );
        } else {
            throw std::runtime_error(
                "AlvineManager::initializeShapeFunctionVIF is implemented for 2D VIC only");
        }

    }
    void computeSpectralViscosity(ComplexField_t& visc_hat){
      if constexpr (Dim == 2) {
        auto omega = omega_hat_m.getView();
        auto visc  = visc_hat.getView();
        
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

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "Compute spectral viscosity",
            policy_type({nghost, nghost},
                        {static_cast<int>(omega.extent(0)) - nghost,
                         static_cast<int>(omega.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const T laplaceKx = twoPi * mx / Lx;
                const T laplaceKy = twoPi * my / Ly;
                const T k2 = laplaceKx * laplaceKx + laplaceKy * laplaceKy;
                const bool isNotZero = (k2 != T(0));
                visc(i, j) = -T(isNotZero) * viscosity_m * k2 * omega(i, j) / area;
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::computeSpectralViscosity is implemented for 2D VIF only");
      }
    }

    void updateParticleVorticityViscosity(){
        if constexpr (Dim == 2) {
            auto& pc = *this->pcontainer_m;
            auto omega = pc.omega.getView();
            auto visc = pc.viscosity.getView();
            const auto n = pc.getLocalNum();
            const T dt = dt_m;

            const unsigned nxp_global = static_cast<unsigned>(std::sqrt(this->np_m));
            const unsigned nyp_global = this->np_m / nxp_global;
            const T dxp = (this->rmax_m[0] - this->rmin_m[0]) / nxp_global;
            const T dyp = (this->rmax_m[1] - this->rmin_m[1]) / nyp_global;
            const T particleArea = dxp * dyp;

            Kokkos::parallel_for(
                "update_particle_vorticity_viscosity",
                n,
                KOKKOS_LAMBDA(const size_t i){
                    omega(i) += dt * visc(i) * particleArea;
                });
            Kokkos::fence();
        } else {
            throw std::runtime_error("AlvineManager::updateParticleVorticityViscosity is implemented for 2D only");
        }
    }
    void reconstructSpectralVorticity(RealField_t& omegaField) {
      if constexpr (Dim == 2) {
        if (!spectralFft_mp) {
          throw std::runtime_error(
              "AlvineManager::reconstructSpectralVorticity called before initNUFFT");
        }

        auto omegaModes = omega_hat_m.deepCopy();

        const auto& domain = omega_hat_m.getLayout().getDomain();
        const auto& dx     = omega_hat_m.get_mesh().getMeshSpacing();
        const T area =
            (dx[0] * domain[0].length()) * (dx[1] * domain[1].length());

        omegaModes = omegaModes / area;
        applyCellCenteredIfftPhase(omegaModes);

        spectralFft_mp->transform(ippl::BACKWARD, omegaModes);

        auto omegaOut = omegaField.getView();
        auto omegaGrid = omegaModes.getView();
        const int nghost = omegaField.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "reconstruct_spectral_vorticity",
            policy_type({nghost, nghost},
                        {static_cast<int>(omegaOut.extent(0)) - nghost,
                         static_cast<int>(omegaOut.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              omegaOut(i, j) = omegaGrid(i, j).real();
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::reconstructSpectralVorticity is implemented for 2D VIC only");
      }
    }

    void reconstructSpectralVelocity(VField_t<T, Dim>& uField) {
      if constexpr (Dim == 2) {
        if (!spectralFft_mp) {
          throw std::runtime_error(
              "AlvineManager::reconstructSpectralVelocity called before initNUFFT");
        }

        auto uxModes = ux_hat_m.deepCopy();
        auto uyModes = uy_hat_m.deepCopy();

        applyCellCenteredIfftPhase(uxModes);
        applyCellCenteredIfftPhase(uyModes);

        spectralFft_mp->transform(ippl::BACKWARD, uxModes);
        spectralFft_mp->transform(ippl::BACKWARD, uyModes);

        auto uOut   = uField.getView();
        auto uxGrid = uxModes.getView();
        auto uyGrid = uyModes.getView();
        const int nghost = uField.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
	        Kokkos::parallel_for(
	            "reconstruct_spectral_velocity",
	            policy_type({nghost, nghost},
	                        {static_cast<int>(uOut.extent(0)) - nghost,
	                         static_cast<int>(uOut.extent(1)) - nghost}),
	            KOKKOS_LAMBDA(const int i, const int j) {
	              uOut(i, j)[0] = uxGrid(i, j).real();
	              uOut(i, j)[1] = uyGrid(i, j).real();
	            });
	        Kokkos::fence();

	        const auto uAverage = uField.getVolumeAverage();
	        Kokkos::parallel_for(
	            "remove_reconstructed_velocity_mean",
	            policy_type({nghost, nghost},
	                        {static_cast<int>(uOut.extent(0)) - nghost,
	                         static_cast<int>(uOut.extent(1)) - nghost}),
	            KOKKOS_LAMBDA(const int i, const int j) {
	              uOut(i, j)[0] -= uAverage[0];
	              uOut(i, j)[1] -= uAverage[1];
	            });
	        Kokkos::fence();
	      } else {
	        throw std::runtime_error(
	            "AlvineManager::reconstructSpectralVelocity is implemented for 2D VIC only");
	      }
	    }

    double computeSpectralDivergenceL2() {
        if constexpr (Dim == 2) {
            auto ux = ux_hat_m.getView();
            auto uy = uy_hat_m.getView();

            auto& layout = ux_hat_m.getLayout();
            auto& mesh   = ux_hat_m.get_mesh();

            const auto& lDom   = layout.getLocalNDIndex();
            const auto& domain = layout.getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = ux_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T Lx   = dx[0] * Nx;
            const T Ly   = dx[1] * Ny;

            const T twoPi = T(2.0 * std::acos(-1.0));
            const Kokkos::complex<T> imag(0.0, 1.0);

            double localDiv2 = 0.0;
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_reduce(
                "compute_spectral_divergence_l2",
                policy_type({nghost, nghost},
                            {static_cast<int>(ux.extent(0)) - nghost,
                             static_cast<int>(ux.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                    const int gx = i - nghost + lDom[0].first();
                    const int gy = j - nghost + lDom[1].first();

                    const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                    const int my = (gy <= Ny / 2) ? gy : gy - Ny;

                    const bool notMidX = (gx != Nx / 2);
                    const bool notMidY = (gy != Ny / 2);

                    const T kx = notMidX * twoPi * mx / Lx;
                    const T ky = notMidY * twoPi * my / Ly;

                    const auto divHat = imag * (kx * ux(i, j) + ky * uy(i, j));
                    lsum += divHat.real() * divHat.real() + divHat.imag() * divHat.imag();
                },
                localDiv2);

            double globalDiv2 = 0.0;
            ippl::Comm->reduce(localDiv2, globalDiv2, 1, std::plus<double>());

            const double N = static_cast<double>(Nx) * static_cast<double>(Ny);
            return std::sqrt(globalDiv2 / N);
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralDivergenceL2 is implemented for 2D VIC only");
        }
    }

    double computeSpectralEnergy() {
        if constexpr (Dim == 2) {
            auto ux = ux_hat_m.getView();
            auto uy = uy_hat_m.getView();

            auto& layout       = ux_hat_m.getLayout();
            auto& mesh         = ux_hat_m.get_mesh();
            const auto& domain = layout.getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = ux_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T Lx   = dx[0] * Nx;
            const T Ly   = dx[1] * Ny;
            const T area = Lx * Ly;

            double localEnergy = 0.0;
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_reduce(
                "compute_spectral_energy",
                policy_type({nghost, nghost},
                            {static_cast<int>(ux.extent(0)) - nghost,
                             static_cast<int>(ux.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                    const auto uxMode = ux(i, j);
                    const auto uyMode = uy(i, j);

                    const double ux2 =
                        uxMode.real() * uxMode.real() + uxMode.imag() * uxMode.imag();
                    const double uy2 =
                        uyMode.real() * uyMode.real() + uyMode.imag() * uyMode.imag();

                    lsum += 0.5 * (ux2 + uy2);
                },
                localEnergy);

            double globalEnergy = 0.0;
            ippl::Comm->allreduce(localEnergy, globalEnergy, 1, std::plus<double>());

            // ux_hat_m and uy_hat_m are Fourier-series coefficients because
            // computeSpectralVelocityModes divides the raw type-1 NUFFT modes by
            // the domain area. Parseval therefore contributes one factor of area.
            return area * globalEnergy;
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralEnergy is implemented for 2D VIC only");
        }
    }

    double computeSpectralEnstrophy() {
        if constexpr (Dim == 2) {
            auto omega = omega_hat_m.getView();

            auto& mesh         = omega_hat_m.get_mesh();
            const auto& domain = omega_hat_m.getLayout().getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = omega_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T Lx   = dx[0] * Nx;
            const T Ly   = dx[1] * Ny;
            const T area = Lx * Ly;

            double localEnstrophy = 0.0;
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_reduce(
                "compute_spectral_enstrophy",
                policy_type({nghost, nghost},
                            {static_cast<int>(omega.extent(0)) - nghost,
                             static_cast<int>(omega.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                    const auto omegaMode = omega(i, j);
                    const double omega2 =
                        omegaMode.real() * omegaMode.real() + omegaMode.imag() * omegaMode.imag();
                    lsum += 0.5 * omega2;
                },
                localEnstrophy);

            double globalEnstrophy = 0.0;
            ippl::Comm->allreduce(localEnstrophy, globalEnstrophy, 1, std::plus<double>());

            // Dividing raw type-1 modes by area gives Fourier-series coefficients,
            // so Parseval contributes one inverse factor of area.
            return globalEnstrophy / area;
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralEnstrophy is implemented for 2D VIC only");
        }
    }

    std::vector<VorticitySpectrumShell> computeSpectralVorticitySpectrum() {
        if constexpr (Dim == 2) {
            auto omega = omega_hat_m.getView();

            auto& layout       = omega_hat_m.getLayout();
            auto& mesh         = omega_hat_m.get_mesh();
            const auto& lDom   = layout.getLocalNDIndex();
            const auto& domain = layout.getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = omega_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T area = (dx[0] * Nx) * (dx[1] * Ny);
            const int maxShell =
                static_cast<int>(std::floor(std::hypot(Nx / 2.0, Ny / 2.0)));
            const int shellCount = maxShell + 1;

            Kokkos::View<double*> localSpectrum("local_vorticity_spectrum", shellCount);
            Kokkos::View<std::uint64_t*> localModeCounts(
                "local_vorticity_spectrum_mode_counts", shellCount);
            Kokkos::deep_copy(localSpectrum, 0.0);
            Kokkos::deep_copy(localModeCounts, std::uint64_t(0));

            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for(
                "compute_spectral_vorticity_spectrum",
                policy_type({nghost, nghost},
                            {static_cast<int>(omega.extent(0)) - nghost,
                             static_cast<int>(omega.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    const int gx = i - nghost + lDom[0].first();
                    const int gy = j - nghost + lDom[1].first();

                    const int mx    = (gx <= Nx / 2) ? gx : gx - Nx;
                    const int my    = (gy <= Ny / 2) ? gy : gy - Ny;
                    const int shell =
                        static_cast<int>(Kokkos::floor(Kokkos::sqrt(T(mx * mx + my * my))));

                    const auto omegaMode = omega(i, j);
                    const double omega2 =
                        omegaMode.real() * omegaMode.real() + omegaMode.imag() * omegaMode.imag();

                    Kokkos::atomic_add(&localSpectrum(shell), 0.5 * omega2 / area);
                    Kokkos::atomic_add(&localModeCounts(shell), std::uint64_t(1));
                });
            Kokkos::fence();

            auto hostSpectrum =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localSpectrum);
            auto hostModeCounts =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localModeCounts);

            std::vector<double> localValues(shellCount);
            std::vector<double> globalValues(shellCount);
            std::vector<std::uint64_t> localCounts(shellCount);
            std::vector<std::uint64_t> globalCounts(shellCount);
            for (int shell = 0; shell < shellCount; ++shell) {
                localValues[shell] = hostSpectrum(shell);
                localCounts[shell] = hostModeCounts(shell);
            }

            ippl::Comm->allreduce(
                localValues.data(), globalValues.data(), shellCount, std::plus<double>());
            ippl::Comm->allreduce(
                localCounts.data(), globalCounts.data(), shellCount, std::plus<std::uint64_t>());

            const int completeShellLimit = std::min(Nx, Ny) / 2;
            std::vector<VorticitySpectrumShell> spectrum(shellCount);
            for (int shell = 0; shell < shellCount; ++shell) {
                spectrum[shell].enstrophy = globalValues[shell];
                spectrum[shell].modeCount = globalCounts[shell];
                spectrum[shell].complete  = shell < completeShellLimit;
            }
            return spectrum;
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralVorticitySpectrum is implemented for 2D VIC only");
        }
    }
};
#endif
