constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "VIFEuler3DTest";
#include "Ippl.h"
#include "datatypes.h"
#include "VortexInFourier3DManager.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

class EulerProbe : public VortexInFourier3DManager<T> {
public:
    using VortexInFourier3DManager<T>::VortexInFourier3DManager;

    T volume() const {
        const auto L = this->rmax_m-this->rmin_m;
        return L[0]*L[1]*L[2]/this->np_m;
    }

    void setShear() {
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView(); auto w = pc->omega.getView();
        const T v = volume();
        Kokkos::parallel_for("vif_euler_test_shear", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                w(p)[0]=0; w(p)[1]=0; w(p)[2]=-v*Kokkos::cos(R(p)[1]);
            });
        Kokkos::fence();
    }

    T remeshTGVError() {
        auto pc=this->pcontainer_m;
        auto R=pc->R.getView(); auto old=pc->R_old.getView();
        auto w=pc->omega.getView(); auto P=pc->P.getView(); auto u=pc->u.getView();
        auto wx=pc->omega_x.getView(); auto wy=pc->omega_y.getView(); auto wz=pc->omega_z.getView();
        auto ux=pc->ux.getView(); auto uy=pc->uy.getView(); auto uz=pc->uz.getView();
        const T v=volume();
        T error=0;
        Kokkos::parallel_reduce("vif_test_ifft_remesh",pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p,T& e) {
                const T x=R(p)[0], y=R(p)[1], z=R(p)[2];
                const auto expected=TaylorGreen3D<T>::vorticity(x,y,z);
                Vector_t<T,3> velocity(0);
                velocity[0]=Kokkos::sin(x)*Kokkos::cos(y)*Kokkos::cos(z);
                velocity[1]=-Kokkos::cos(x)*Kokkos::sin(y)*Kokkos::cos(z);
                for(unsigned d=0;d<3;++d) {
                    e=Kokkos::max(e,Kokkos::abs(w(p)[d]/v-expected[d]));
                    e=Kokkos::max(e,Kokkos::abs(P(p)[d]-velocity[d]));
                    e=Kokkos::max(e,Kokkos::abs(u(p)[d]-P(p)[d]));
                    e=Kokkos::max(e,Kokkos::abs(old(p)[d]-R(p)[d]));
                }
                e=Kokkos::max(e,Kokkos::abs(wx(p)-w(p)[0])/v);
                e=Kokkos::max(e,Kokkos::abs(wy(p)-w(p)[1])/v);
                e=Kokkos::max(e,Kokkos::abs(wz(p)-w(p)[2])/v);
                e=Kokkos::max(e,Kokkos::abs(ux(p)-P(p)[0]));
                e=Kokkos::max(e,Kokkos::abs(uy(p)-P(p)[1]));
                e=Kokkos::max(e,Kokkos::abs(uz(p)-P(p)[2]));
            },Kokkos::Max<T>(error));
        T global=0;
        MPI_Allreduce(&error,&global,1,MPI_DOUBLE,MPI_MAX,ippl::Comm->getCommunicator());
        return global;
    }

    T stateError(const bool shear = false, const T amplitude = 1) {
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView(); auto base = pc->rk4_R0.getView();
        auto old = pc->R_old.getView(); auto w = pc->omega.getView();
        auto wx = pc->omega_x.getView(); auto wy = pc->omega_y.getView();
        auto wz = pc->omega_z.getView();
        const T v=volume(), dt=this->dt_m, nu=this->viscosity_m;
        const auto lower=this->rmin_m;
        const Vector_t<T,3> L=this->rmax_m-this->rmin_m;
        T local=0;
        Kokkos::parallel_reduce("vif_euler_test_state", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p, T& error) {
                Vector_t<T,3> expected(0);
                if (shear) {
                    expected[2]=-amplitude*Kokkos::cos(R(p)[1]);
                } else {
                    const T x=base(p)[0], y=base(p)[1], z=base(p)[2];
                    expected=TaylorGreen3D<T>::vorticity(x,y,z)*(1-3*nu*dt);
                    expected[0]-=dt*Kokkos::sin(2*y)*Kokkos::sin(2*z)/4;
                    expected[1]+=dt*Kokkos::sin(2*x)*Kokkos::sin(2*z)/4;
                    Vector_t<T,3> velocity(0);
                    velocity[0]=Kokkos::sin(x)*Kokkos::cos(y)*Kokkos::cos(z);
                    velocity[1]=-Kokkos::cos(x)*Kokkos::sin(y)*Kokkos::cos(z);
                    for (unsigned d=0; d<3; ++d) {
                        T position=base(p)[d]+dt*velocity[d];
                        position-=L[d]*Kokkos::floor((position-lower[d])/L[d]);
                        error=Kokkos::max(error,Kokkos::abs(R(p)[d]-position));
                        error=Kokkos::max(error,Kokkos::abs(old(p)[d]-base(p)[d]));
                    }
                }
                for (unsigned d=0; d<3; ++d)
                    error=Kokkos::max(error,Kokkos::abs(w(p)[d]/v-expected[d]));
                error=Kokkos::max(error,Kokkos::abs(wx(p)-w(p)[0])/v);
                error=Kokkos::max(error,Kokkos::abs(wy(p)-w(p)[1])/v);
                error=Kokkos::max(error,Kokkos::abs(wz(p)-w(p)[2])/v);
            }, Kokkos::Max<T>(local));
        T global=0;
        MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,ippl::Comm->getCommunicator());
        return global;
    }
};

void check(const char* label, T error) {
    if (ippl::Comm->rank()==0) std::cout << label << ": " << error << '\n';
    if (!std::isfinite(error) || error>2e-7) throw std::runtime_error(label);
}

int main(int argc, char** argv) {
    ippl::initialize(argc,argv);
    int status=0;
    try {
        Vector_t<int,3> nr(8);
        auto lo=TaylorGreen3D<T>::domainMin(), hi=TaylorGreen3D<T>::domainMax();
        std::string solver="FFT";
        const T dt=0.03;
        EulerProbe roundtrip(1,nr,512,solver,0,dt,"vif_ifft_remesh",0,0,"euler",lo,hi,lo,1,0);
        roundtrip.pre_run();
        // No time advance: detect IFFT phase, normalization, indexing, or
        // packing errors independently of Euler's physical evolution.
        roundtrip.remeshParticles3D();
        check("IFFT remesh preserves TGV velocity and strengths",roundtrip.remeshTGVError());
        bool rejected=false;
        try {
            EulerProbe mismatch(1,nr,4096,solver,0,dt,"vif_ifft_mismatch",0,0,"euler",lo,hi,lo,1,0);
            mismatch.pre_run();
        } catch (const std::runtime_error&) { rejected=true; }
        if (!rejected) throw std::runtime_error("IFFT remesh accepted a mismatched lattice");
        for (const T nu : {0.,0.2}) {
            // Two particle densities test strength-volume scaling independently
            // of the Fourier grid size. Check after MPI migration as well.
            for (unsigned np : {512u,4096u}) {
                EulerProbe p(1,nr,np,solver,0,dt,"vif_euler_test",0,nu,"euler",lo,hi,lo,0,0);
                p.pre_run(); p.run(1);
                check("TGV forward Euler from the common old state",p.stateError());
            }
            for (int remesh : {0,1}) {
                EulerProbe p(3,nr,512,solver,0,dt,"vif_euler_shear",0,nu,"euler",lo,hi,lo,remesh,0);
                p.pre_run(); p.setShear(); p.run(3);
                check("explicit shear decay with/without remeshing",
                      p.stateError(true,std::pow(1-nu*dt,3)));
            }
        }
        EulerProbe clipped(1,nr,512,solver,0,0.1,"vif_euler_clipped",0,0.2,"euler",lo,hi,lo,0,0);
        clipped.pre_run(); clipped.setShear();
        clipped.setAdaptiveLCFL(true); clipped.setLCFL(0.01); clipped.setFinalTime(0.005);
        clipped.run(1);
        check("clipped adaptive dt in explicit viscosity",clipped.stateError(true,1-0.2*0.005));
    } catch (const std::exception& e) {
        std::cerr << "VIF Euler regression failed: " << e.what() << '\n'; status=1;
    }
    ippl::finalize();
    return status;
}
