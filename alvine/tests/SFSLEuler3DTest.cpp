// Exercise the production FFT/source/particle pipeline on one or more MPI ranks.
constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "SFSLEuler3DTest";

#include "Ippl.h"
#include "datatypes.h"
#include "SpectralFSL3DManager.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "SFSL3DTestSupport.hpp"

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    int status = 0;
    try {
        Vector_t<int,3> nr(8);
        auto lower = TaylorGreen3D<T>::domainMin();
        auto upper = TaylorGreen3D<T>::domainMax();
        std::string solver = "FFT";
        const double dt = 0.03;
        for (const double nu : {0.0, 0.2}) {
            SFSLProbe manager(3, nr, 512, solver, 0, dt, "sfsl_euler_test",
                               0, nu, "euler", lower, upper, lower, 0);
            manager.pre_run();
            check("initial TGV FFT/IFFT normalization (no initial viscous step)",
                  manager.gridError(false, 1));
            manager.computeEulerGridSource3D();
            manager.updateEulerSpectralVorticity3D();
            check("TGV grid stretching and implicit split",
                  manager.gridError(false, 1/(1+3*nu*dt), dt/(1+8*nu*dt)));

            manager.setShear();
            manager.setUseStretching(false);
            const double initialEnergy = manager.energy();
            const double initialEnstrophy = manager.enstrophy();
            manager.run(3);
            const double amplitude = std::pow(1+nu*dt, -3);
            check("three complete Euler steps preserve source-updated strengths",
                  manager.gridError(true, amplitude));
            check("shear kinetic energy decay",
                  std::abs(manager.energy()/initialEnergy - amplitude*amplitude));
            check("shear enstrophy decay",
                  std::abs(manager.enstrophy()/initialEnstrophy - amplitude*amplitude));
        }

        SFSLProbe adaptive(1, nr, 512, solver, 0, 0.1, "sfsl_euler_lcfl_test",
                            0, 0.2, "euler", lower, upper, lower, 0);
        adaptive.pre_run();
        adaptive.setShear();
        adaptive.setAdaptiveLCFL(true);
        adaptive.setLCFL(0.01);
        adaptive.setFinalTime(0.005);
        adaptive.run(1);
        check("final-time clipped dt also used by diffusion",
              adaptive.gridError(true, 1/(1+0.2*0.005)));

        SFSLProbe lcfl(1, nr, 512, solver, 0, 0.1, "sfsl_euler_lcfl_active_test",
                        0, 0.2, "euler", lower, upper, lower, 0);
        lcfl.pre_run();
        lcfl.setShear();
        lcfl.setAdaptiveLCFL(true);
        lcfl.setLCFL(0.01);
        lcfl.run(1);
        // For this shear, ||sym(grad u)||_1=|cos(y)|/2; the grid maximum
        // is cos(pi/8)/2. The selected dt must enter the viscous denominator.
        const double lcflDt = 0.01 / (0.5 * std::cos(std::acos(-1.0)/8));
        check("active LCFL dt used by diffusion",
              lcfl.gridError(true, 1/(1+0.2*lcflDt)));
    } catch (const std::exception& error) {
        std::cerr << "SFSL Euler regression failed: " << error.what() << std::endl;
        status = 1;
    }
    ippl::finalize();
    return status;
}
