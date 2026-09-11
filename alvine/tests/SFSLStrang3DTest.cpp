constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "SFSLStrang3DTest";

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
            SFSLProbe manager(3, nr, 512, solver, 0, dt, "sfsl_strang_test",
                               0, nu, "strang", lower, upper, lower, 0);
            manager.pre_run();
            check("coupled particle RHS and noncumulative synchronized stages",
                  manager.coupledStageError());
            check("shape filter is bypassed inside coupled stages",
                  manager.coupledStageError(true));
            check("no diffusion during initialization", manager.gridError(false, 1));
            manager.applyStrangSpectralDiffusion3D(dt/2);
            check("first exact half-step on physical vorticity",
                  manager.gridError(false, std::exp(-3*nu*dt/2)));
            manager.applyStrangSpectralDiffusion3D(dt/2);
            check("two halves equal one exact step",
                  manager.gridError(false, std::exp(-3*nu*dt)));

            manager.setShear();
            manager.setUseStretching(nu == 0.0);
            const double e0 = manager.energy(), z0 = manager.enstrophy();
            manager.run(3);
            const double amplitude = std::exp(-nu*dt*3);
            check("complete Strang shear diffusion/transport/reset",
                  manager.gridError(true, amplitude));
            check("Strang energy decay", std::abs(manager.energy()/e0-amplitude*amplitude));
            check("Strang enstrophy decay", std::abs(manager.enstrophy()/z0-amplitude*amplitude));
        }

        SFSLProbe adaptive(1, nr, 512, solver, 0, 0.1, "sfsl_strang_lcfl_test",
                            0, 0.2, "strang", lower, upper, lower, 0);
        adaptive.pre_run();
        adaptive.setShear();
        adaptive.setAdaptiveLCFL(true);
        adaptive.setLCFL(0.01);
        adaptive.run(1);
        const double selectedDt = 0.01/(0.5*std::cos(std::acos(-1.0)/8));
        check("same LCFL dt in both diffusion halves",
              adaptive.gridError(true, std::exp(-0.2*selectedDt)));

        SFSLProbe clipped(1, nr, 512, solver, 0, 0.1, "sfsl_strang_clipped_test",
                           0, 0.2, "strang", lower, upper, lower, 0);
        clipped.pre_run();
        clipped.setShear();
        clipped.setFinalTime(0.005);
        clipped.run(1);
        check("clipped final time in both diffusion halves",
              clipped.gridError(true, std::exp(-0.2*0.005)));

        // Isolate spectral stretching: RK4 should converge at order four on
        // the fixed spectral discretization. This catches stale/cumulative k2-k4.
        SFSLProbe source(1, nr, 512, solver, 0, 0.1, "sfsl_strang_source_test",
                          0, 0.0, "rk4", lower, upper, lower, 0);
        source.pre_run();
        auto initial = source.snapshot();
        for (int i=0; i<16; ++i) source.advanceStrangStretchingRK4_3D(0.025, "reference");
        auto reference = source.snapshot();
        double sourceErrors[3];
        for (int level=0; level<3; ++level) {
            source.restoreModes(initial);
            const int steps = 1 << level;
            for (int i=0; i<steps; ++i)
                source.advanceStrangStretchingRK4_3D(0.4/steps, "source");
            sourceErrors[level] = source.modeError(reference);
        }
        if (!(sourceErrors[0] > 8*sourceErrors[1] && sourceErrors[1] > 8*sourceErrors[2]))
            throw std::runtime_error("spectral stretching does not show RK4 refinement");

        // Complete noncommuting TGV composition: expect SECOND order, not RK4
        // order. Use a finer grid to keep remapping/aliasing below time errors.
        Vector_t<int,3> fineNr(16);
        SFSLProbe fine(32, fineNr, 4096, solver, 0, 0.00625, "sfsl_strang_reference",
                        0, 0.02, "strang", lower, upper, lower, 0);
        fine.pre_run();
        fine.run(32);
        auto fullReference = fine.snapshot();
        double errors[3];
        for (int level=0; level<3; ++level) {
            const int steps = 2 << level;
            SFSLProbe run(steps, fineNr, 4096, solver, 0, 0.2/steps, "sfsl_strang_convergence",
                           0, 0.02, "strang", lower, upper, lower, 0);
            run.pre_run();
            run.run(steps);
            errors[level] = run.modeError(fullReference);
            if (ippl::Comm->rank() == 0)
                std::cout << "Strang TGV dt=" << 0.2/steps << " error=" << errors[level] << '\n';
        }
        if (!(errors[0] > 2.5*errors[1] && errors[1] > 2.5*errors[2]))
            throw std::runtime_error("complete Strang composition does not show second-order refinement");
    } catch (const std::exception& error) {
        std::cerr << "SFSL Strang regression failed: " << error.what() << std::endl;
        status = 1;
    }
    ippl::finalize();
    return status;
}
