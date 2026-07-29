#ifndef IPPL_ALVINE_CORE_ALVINE_MANAGER3D_H
#define IPPL_ALVINE_CORE_ALVINE_MANAGER3D_H

#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "../fields/FieldContainer.hpp"
#include "../FieldSolver.hpp"
#include "../LoadBalancer.hpp"
#include "Manager/BaseManager.h"
#include "Manager/PicManager.h"
#include "../particles/ParticleContainer.hpp"
#include "../test_cases/TaylorGreen3D.hpp"
#include "FFT/Transform/Transform.h"

template <typename T>
class AlvineManager3D
    : public ippl::PicManager<T, 3, ParticleContainer<T, 3>, FieldContainer<T, 3>,
                              LoadBalancer<T, 3>> {
public:
    static constexpr unsigned Dim = 3;

    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t    = FieldContainer<T, Dim>;
    using FieldSolver_t       = FieldSolver<T, Dim>;
    using LoadBalancer_t      = LoadBalancer<T, Dim>;
    using Base                = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using RealField_t         = Field<T, Dim>;
    using Nufft_t             = ippl::FFT<ippl::NUFFTransform, RealField_t>;
    using ComplexField_t      = typename Nufft_t::ComplexField;
    using SpectralFft_t       = ippl::FFT<ippl::CCTransform, ComplexField_t>;

    std::shared_ptr<Nufft_t> nufftType1_mp;
    std::shared_ptr<Nufft_t> nufftType2_mp;
    std::shared_ptr<SpectralFft_t> spectralFft_mp;

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
    int shapedegree_m = 1;
    double viscosity_m;

    double time_m;
    double dt_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;
    Vector_t<double, Dim> origin_m;
    Vector_t<double, Dim> hr_m;
    ComplexField_t omega_x_hat_m;
    ComplexField_t omega_y_hat_m;
    ComplexField_t omega_z_hat_m;

    ComplexField_t ux_hat_m;
    ComplexField_t uy_hat_m;
    ComplexField_t uz_hat_m;

    ComplexField_t viscosity_x_hat_m;
    ComplexField_t viscosity_y_hat_m;
    ComplexField_t viscosity_z_hat_m;
    RealField_t Sk_m;
    bool spectral_3d_diagnostics_initialized_m = false;
    bool tgv_single_mode_3d_initialized_m = false;
    bool tgv_3d_diagnostics_initialized_m = false;

public:
    AlvineManager3D(unsigned nt_, Vector_t<int, Dim>& nr_, unsigned np_, std::string& solver_,
                    int dump_freq_, double dt_ = 0.05, std::string method_ = "alvine_3d",
                    int spectral_filter_ = 0, double viscosity = 0.0,
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

    ~AlvineManager3D() {}

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

#include "../spectral/AlvineSpectralSetup3D.hpp"
#include "../spectral/AlvineSpectralOps3D.hpp"
#include "../io/AlvineFileNaming.hpp"
#include "../diagnostics/AlvineSpectralDiagnostics3D.hpp"
#include "../diagnostics/AlvineTaylorGreenDiagnostics3D.hpp"
};

#endif
