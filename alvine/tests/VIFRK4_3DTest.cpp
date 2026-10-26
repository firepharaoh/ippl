constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "VIFRK4_3DTest";
#include "Ippl.h"
#include "datatypes.h"
#include "VortexInFourier3DManager.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "VIF3DTestSupport.hpp"

// For transverse shear, stretching vanishes and the coupled equations are
// Gamma'=-nu*Gamma, x'=A*sin(y), y'=z'=0. This independently checks both
// RK combinations, including the influence of stage strengths on velocity.
class RK4Probe : public VIFProbe {
public:
    using VIFProbe::VIFProbe;

    void prepareRemeshedState(bool consumeSkip) {
        this->remeshParticles3D();
        if (consumeSkip) {
            auto pc = this->pcontainer_m;
            this->evaluateParticleTimeRHS3D(pc->rk4_k1, pc->rk4_omega_k1);
        }
    }

    void checkIndependentStorage() {
        auto pc = this->pcontainer_m;
        const Vector_t<T,3>* views[] = {pc->R.getView().data(), pc->rk4_R0.getView().data(),
            pc->R_old.getView().data(), pc->P.getView().data(),
            pc->rk4_k1.getView().data(), pc->rk4_k2.getView().data(),
            pc->rk4_k3.getView().data(), pc->rk4_k4.getView().data(),
            pc->omega.getView().data(), pc->rk4_omega0.getView().data()};
        for (unsigned i=0; i<10; ++i)
            for (unsigned j=0; j<i; ++j)
                if (views[i] == views[j])
                    throw std::runtime_error("RK4 buffers share storage");
    }

    T strengthNorm() {
        auto w = this->pcontainer_m->omega.getView();
        T local = 0;
        Kokkos::parallel_reduce("vif_rk4_strength_norm", this->pcontainer_m->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p, T& sum) {
                for (unsigned d=0; d<3; ++d) sum += w(p)[d]*w(p)[d];
            }, local);
        T global = 0;
        MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,ippl::Comm->getCommunicator());
        return global;
    }

    T positionError(T displacement) {
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto base = pc->rk4_R0.getView();
        auto old = pc->R_old.getView();
        const auto lower = this->rmin_m;
        const Vector_t<T,3> L = this->rmax_m-this->rmin_m;
        T local = 0;
        Kokkos::parallel_reduce("vif_rk4_shear_positions", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p, T& error) {
                for (unsigned d=0; d<3; ++d) {
                    T expected = base(p)[d];
                    if (d==0) expected += displacement*Kokkos::sin(base(p)[1]);
                    expected -= L[d]*Kokkos::floor((expected-lower[d])/L[d]);
                    T delta = Kokkos::abs(R(p)[d]-expected);
                    delta = Kokkos::min(delta,L[d]-delta);
                    error = Kokkos::max(error,delta);
                    error = Kokkos::max(error,Kokkos::abs(old(p)[d]-base(p)[d]));
                }
            }, Kokkos::Max<T>(local));
        T global = 0;
        MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,ippl::Comm->getCommunicator());
        return global;
    }
};

T stability(T z) {
    return 1+z+z*z/2+z*z*z/6+z*z*z*z/24;
}

int main(int argc, char** argv) {
    ippl::initialize(argc,argv);
    int status=0;
    try {
        Vector_t<int,3> nr(16);
        auto lo=TaylorGreen3D<T>::domainMin(), hi=TaylorGreen3D<T>::domainMax();
        std::string solver="FFT";
        for (T nu : {0.,0.2}) {
            for (unsigned np : {4096u,32768u}) {
                const T dt=0.3, z=-nu*dt;
                RK4Probe p(1,nr,np,solver,0,dt,"vif_rk4_shear",0,nu,"rk4",lo,hi,lo,0,0);
                p.pre_run(); p.setShear(); p.run(1);
                p.checkIndependentStorage();
                check("RK4 shear strength polynomial",p.stateError(true,stability(z)));
                check("RK4 coupled position polynomial",
                      p.positionError(dt*(1+z/2+z*z/6+z*z*z/24)));
            }
            for (int remesh : {0,1}) {
                RK4Probe p(3,nr,4096,solver,0,0.1,"vif_rk4_remesh",0,nu,"rk4",lo,hi,lo,remesh,0);
                p.pre_run(); p.setShear(); p.run(3);
                check("RK4 repeated shear with/without remeshing",
                      p.stateError(true,std::pow(stability(-nu*0.1),3)));
            }
        }
        // A remesh skip flag must not change the operator used by RK4.
        // Shape filtering is non-idempotent, so stage inconsistency is visible.
        T filteredNorms[2];
        for (int skip=0; skip<2; ++skip) {
            RK4Probe p(1,nr,4096,solver,0,0.1,"vif_rk4_filter",1,0.2,"rk4",lo,hi,lo,0,0);
            p.pre_run(); p.setShear();
            p.prepareRemeshedState(skip != 0);
            p.run(1);
            filteredNorms[skip]=p.strengthNorm();
        }
        check("RK4 consistent filtering after remesh",
              std::abs(filteredNorms[0]-filteredNorms[1]));
        T errors[3];
        // Keep every resolved diffusion mode inside RK4's stability region,
        // including roundoff/NUFFT noise outside the analytic shear mode.
        Vector_t<int,3> shearGrid(8);
        for (int level=0; level<3; ++level) {
            const int steps=10<<level;
            RK4Probe p(steps,shearGrid,4096,solver,0,1./steps,"vif_rk4_order",0,1.,"rk4",lo,hi,lo,0,0);
            p.pre_run(); p.setShear(); p.run(steps);
            errors[level]=p.stateError(true,std::exp(-1.));
            if (ippl::Comm->rank()==0)
                std::cout << "shear temporal error (" << steps << " steps): " << errors[level] << '\n';
        }
        if (!(errors[0]>12*errors[1] && errors[1]>12*errors[2]))
            throw std::runtime_error("RK4 shear does not converge at fourth order");
        // Nonlinear TGV exercises nonzero stretching and moving particles.
        // Compare a global observable to a refined solution of the same spatial ODE.
        T norms[4];
        const int counts[4] = {2,4,8,64};
        for (int level=0; level<4; ++level) {
            const int steps=counts[level];
            RK4Probe p(steps,nr,4096,solver,0,0.8/steps,"vif_rk4_tgv",0,0.02,"rk4",lo,hi,lo,0,0);
            p.pre_run(); p.run(steps);
            norms[level]=p.strengthNorm();
        }
        const T e0=std::abs(norms[0]-norms[3]);
        const T e1=std::abs(norms[1]-norms[3]);
        const T e2=std::abs(norms[2]-norms[3]);
        if (ippl::Comm->rank()==0)
            std::cout << "nonlinear TGV norm errors: " << e0 << ", " << e1 << ", " << e2 << '\n';
        if (!(e0>8*e1 && e1>8*e2))
            throw std::runtime_error("coupled RK4 nonlinear refinement failed");
        RK4Probe clipped(1,nr,4096,solver,0,0.1,"vif_rk4_clipped",0,0.2,"rk4",lo,hi,lo,0,0);
        clipped.pre_run(); clipped.setShear();
        clipped.setAdaptiveLCFL(true); clipped.setLCFL(0.01); clipped.setFinalTime(0.005);
        clipped.run(1);
        check("one clipped timestep in all RK stages",
              clipped.stateError(true,stability(-0.2*0.005)));
    } catch (const std::exception& e) {
        std::cerr << "VIF RK4 regression failed: " << e.what() << '\n'; status=1;
    }
    ippl::finalize();
    return status;
}
