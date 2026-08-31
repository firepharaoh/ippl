// Spectral Forward Semi-Lagrangian 3D test
// Usage:
//   srun ./SpectralFSL3D <nx> <ny> <nz> <Np> <Nt> <solver> <dump_freq>
//        [--dt value] [--final-time value] [--method label] [--filter 0|1|2|3]
//        [--test-case taylor_green_3d] [--viscosity value]
//        [--diagnostics-freq frequency] [--adaptive-lcfl] [--lcfl value]
//        [--integrator euler|leapfrog|rk4] [--rk4-stage-trace]
//        [--no-stretching] [--overallocate value] [--info level]

constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "SpectralFSL3D";

#include "Ippl.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

#include "datatypes.h"
#include "Utility/IpplTimings.h"
#include "SpectralFSL3DManager.h"
#include "test_cases/TestCaseSelection.hpp"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        if (argc < 8) {
            msg << "Usage: SpectralFSL3D nx ny nz np nt solver dump_freq "
                   "[--dt dt] [--final-time time] [--method label] [--filter 0|1|2|3] "
                   "[--test-case taylor_green_3d] [--viscosity value] "
                   "[--diagnostics-freq frequency] [--adaptive-lcfl] [--lcfl value] "
                   "[--integrator euler|leapfrog|rk4] [--rk4-stage-trace] "
                   "[--no-stretching] "
                   "[--overallocate value] [--info level]"
                << endl;
            ippl::Comm->abort();
        }

        unsigned arg = 1;
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; ++d) {
            nr[d] = std::atoi(argv[arg++]);
        }

        const unsigned np = static_cast<unsigned>(std::atoi(argv[arg++]));
        const unsigned nt = static_cast<unsigned>(std::atoi(argv[arg++]));
        std::string solver = argv[arg++];
        const int dumpFreq = std::atoi(argv[arg++]);

        double dt = 0.05;
        double finalTime = -1.0;
        std::string method = "sfsl3d";
        std::string testCase = "taylor_green_3d";
        int spectralFilter = 0;
        double viscosity = 0.0;
        int diagnosticsFreq = 1;
        bool adaptiveLCFL = false;
        double lcfl = 1.0;
        bool useStretching = true;
        std::string timeIntegrator = "leapfrog";
        bool rk4StageTrace = false;

        while (arg < static_cast<unsigned>(argc)) {
            const std::string option = argv[arg++];
            if (option == "--dt" && arg < static_cast<unsigned>(argc)) {
                dt = std::atof(argv[arg++]);
            } else if (option == "--final-time" && arg < static_cast<unsigned>(argc)) {
                finalTime = std::atof(argv[arg++]);
            } else if (option == "--method" && arg < static_cast<unsigned>(argc)) {
                method = argv[arg++];
            } else if (option == "--test-case" && arg < static_cast<unsigned>(argc)) {
                testCase = alvine::normalizeTestCaseName(argv[arg++]);
            } else if (option == "--filter" && arg < static_cast<unsigned>(argc)) {
                spectralFilter = std::atoi(argv[arg++]);
            } else if (option == "--viscosity" && arg < static_cast<unsigned>(argc)) {
                viscosity = std::atof(argv[arg++]);
            } else if (option == "--diagnostics-freq" && arg < static_cast<unsigned>(argc)) {
                diagnosticsFreq = std::atoi(argv[arg++]);
            } else if (option == "--integrator" && arg < static_cast<unsigned>(argc)) {
                timeIntegrator = argv[arg++];
                std::transform(timeIntegrator.begin(), timeIntegrator.end(),
                               timeIntegrator.begin(),
                               [](unsigned char c) { return std::tolower(c); });
            } else if (option == "--adaptive-lcfl") {
                adaptiveLCFL = true;
            } else if (option == "--rk4-stage-trace") {
                rk4StageTrace = true;
            } else if (option == "--lcfl" && arg < static_cast<unsigned>(argc)) {
                lcfl = std::atof(argv[arg++]);
            } else if (option == "--no-stretching") {
                useStretching = false;
            } else if ((option == "--overallocate" || option == "-b" || option == "--info"
                        || option == "-i" || option == "--timer-fences")
                       && arg < static_cast<unsigned>(argc)) {
                ++arg;
            } else if (option == "--debug" || option == "-g") {
                // Already handled by ippl::initialize().
            } else {
                msg << "Unknown or incomplete option: " << option << endl;
                ippl::Comm->abort();
            }
        }

        if (alvine::normalizeTestCaseName(testCase) != "taylor_green_3d") {
            msg << "SpectralFSL3D currently requires --test-case taylor_green_3d." << endl;
            ippl::Comm->abort();
        }
        if (spectralFilter < 0 || spectralFilter > 3) {
            msg << "Invalid --filter value " << spectralFilter
                << ". Use 0:no filter, 1:shape function, 2:Hou-Li, 3:2/3 cutoff."
                << endl;
            ippl::Comm->abort();
        }
        if (viscosity < 0.0) {
            msg << "Viscosity must be non-negative." << endl;
            ippl::Comm->abort();
        }
        if (diagnosticsFreq < 0) {
            msg << "Diagnostics frequency must be non-negative." << endl;
            ippl::Comm->abort();
        }
        if (lcfl <= 0.0) {
            msg << "LCFL must be positive." << endl;
            ippl::Comm->abort();
        }
        if (finalTime <= 0.0 && finalTime != -1.0) {
            msg << "Final time must be positive, or omitted." << endl;
            ippl::Comm->abort();
        }
        if (timeIntegrator != "euler" && timeIntegrator != "leapfrog"
            && timeIntegrator != "rk4") {
            msg << "Invalid --integrator value " << timeIntegrator
                << ". Use euler, leapfrog, or rk4." << endl;
            ippl::Comm->abort();
        }

        Vector_t<double, Dim> rmin = alvine::domainMinForTestCase<double, Dim>(testCase);
        Vector_t<double, Dim> rmax = alvine::domainMaxForTestCase<double, Dim>(testCase);
        Vector_t<double, Dim> origin = rmin;

        msg << " Grid size: " << nr
            << " No. of virtual particles: " << np
            << " No. of max steps: " << nt
            << " dt max: " << dt
            << " Method: " << method
            << " Test case: " << testCase
            << " Spectral filter: " << spectralFilter
            << " viscosity: " << viscosity
            << " Diagnostics frequency: " << diagnosticsFreq
            << " Adaptive LCFL: " << (adaptiveLCFL ? "true" : "false")
            << " LCFL: " << lcfl
            << " Final time: " << finalTime
            << " Stretching: " << (useStretching ? "true" : "false")
            << " Time integrator: " << timeIntegrator
            << " RK4 stage trace: " << (rk4StageTrace ? "true" : "false") << endl;

        SpectralFSL3DManager<T> manager(nt, nr, np, solver, dumpFreq, dt, method,
                                        spectralFilter, viscosity, timeIntegrator,
                                        rmin, rmax, origin, diagnosticsFreq);
        manager.setAdaptiveLCFL(adaptiveLCFL);
        manager.setLCFL(lcfl);
        manager.setFinalTime(finalTime);
        manager.setUseStretching(useStretching);
        manager.setRK4StageTrace(rk4StageTrace);
        manager.pre_run();
        manager.run(manager.getNt());

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing_spectral_fsl3d.dat"));
    }
    ippl::finalize();
    return 0;
}
