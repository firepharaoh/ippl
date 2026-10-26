constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "VIFEuler3DTest";
#include "Ippl.h"
#include "datatypes.h"
#include "VortexInFourier3DManager.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "VIF3DTestSupport.hpp"

int main(int argc, char** argv) {
    ippl::initialize(argc,argv);
    int status=0;
    try {
        Vector_t<int,3> nr(16);
        auto lo=TaylorGreen3D<T>::domainMin(), hi=TaylorGreen3D<T>::domainMax();
        std::string solver="FFT";
        const T dt=0.03;
        for (const T nu : {0.,0.2}) {
            // Two particle densities test strength-volume scaling independently
            // of the Fourier grid size. Check after MPI migration as well.
            for (unsigned np : {4096u,32768u}) {
                VIFProbe p(1,nr,np,solver,0,dt,"vif_euler_test",0,nu,"euler",lo,hi,lo,0,0);
                p.pre_run(); p.run(1);
                check("TGV forward Euler from the common old state",p.stateError());
            }
            for (int remesh : {0,1}) {
                VIFProbe p(3,nr,4096,solver,0,dt,"vif_euler_shear",0,nu,"euler",lo,hi,lo,remesh,0);
                p.pre_run(); p.setShear(); p.run(3);
                check("explicit shear decay with/without remeshing",
                      p.stateError(true,std::pow(1-nu*dt,3)));
            }
        }
        VIFProbe clipped(1,nr,4096,solver,0,0.1,"vif_euler_clipped",0,0.2,"euler",lo,hi,lo,0,0);
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
