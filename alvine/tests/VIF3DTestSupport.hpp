#pragma once

class VIFProbe : public VortexInFourier3DManager<T> {
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

