#ifndef IPPL_ALVINE_CORE_ALVINE_MANAGER_H
#define IPPL_ALVINE_CORE_ALVINE_MANAGER_H

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

#include "../FieldContainer.hpp"
#include "../FieldSolver.hpp"
#include "../LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "../ParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"
#include "FFT/Transform/Transform.h"
#include "../test_cases/TaylorGreen2D.hpp"
#include "../test_cases/TestCaseSelection.hpp"

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
    std::string time_integrator_m;
    int dump_freq_m;
    int spectral_filter_m;
    int shapedegree_m = 1;// degree of the shape function for VIF, default is 1 (linear, CIC like)
    double viscosity_m; //Viscosity for the viscous filter, default is 0.0 (no viscosity)

public:
    AlvineManager(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, std::string& solver_,
                  int dump_freq_, double dt_ = 0.05, std::string method_ = "alvine",
                  int spectral_filter_ = 0 , double viscosity= 0.0,
                  std::string time_integrator_ = "leapfrog")
        : ippl::PicManager<T, Dim, ParticleContainer<T, Dim>, FieldContainer<T, Dim>,
                            LoadBalancer<T, Dim>>()
        , nt_m(nt_)
        , nr_m(nr_)
        , np_m(np_)
        , solver_m(solver_)
        , method_m(method_)
        , time_integrator_m(time_integrator_)
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
    bool tgv_vorticity_diagnostics_initialized_m = false;

public:
    double getTime() { return time_m; }

    void setTime(double time_) { time_m = time_; }

    int getNt() const { return nt_m; }

    void setNt(int nt_) { nt_m = nt_; }

    bool useShapeFunctionFilter() const { return spectral_filter_m == 1; }

    bool useHouLiFilter() const { return spectral_filter_m == 2; }

    bool useRK4() const { return time_integrator_m == "rk4"; }

    bool useLeapFrog() const { return time_integrator_m == "leapfrog"; }

    virtual void dump() { /* default does nothing */ };

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

#include "../io/AlvineFileNaming.hpp"
#include "../spectral/AlvineSpectralSetup.hpp"
#include "../particles/AlvineParticleGridTransfer.hpp"
#include "../spectral/AlvineGridVelocity.hpp"
#include "../test_cases/AlvineTaylorGreen2D.hpp"
#include "../diagnostics/AlvineDiagnostics.hpp"
#include "../particles/AlvineScatterCIC.hpp"
#include "../spectral/AlvineSpectralOps.hpp"
#include "../diagnostics/AlvineSpectralDiagnostics.hpp"
};

#endif
