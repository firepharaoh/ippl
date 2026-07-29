constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "VortexInFourier3D";

#include "Ippl.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

#include "datatypes.h"
#include "Utility/IpplTimings.h"
#include "VortexInFourier3DManager.h"
#include "test_cases/TestCaseSelection.hpp"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        if (argc < 8) {
            msg << "Usage: VortexInFourier3D nx ny nz np nt solver dump_freq "
                   "[--dt dt] [--method label] [--filter 0|1|2] "
                   "[--test-case taylor_green_3d] [--viscosity 0.0] "
                   "[--overallocate value] [--info level]"
                << endl;
            ippl::Comm->abort();
        }

        unsigned arg = 1;
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        const unsigned np = static_cast<unsigned>(std::atoi(argv[arg++]));
        const unsigned nt = static_cast<unsigned>(std::atoi(argv[arg++]));
        std::string solver = argv[arg++];
        const int dump_freq = std::atoi(argv[arg++]);

        double dt = 0.05;
        std::string method = "vif3d";
        std::string test_case = "taylor_green_3d";
        int spectral_filter = 0;
        double viscosity = 0.0;
        std::string time_integrator = "leapfrog";

        while (arg < static_cast<unsigned>(argc)) {
            const std::string option = argv[arg++];
            if (option == "--dt" && arg < static_cast<unsigned>(argc)) {
                dt = std::atof(argv[arg++]);
            } else if (option == "--method" && arg < static_cast<unsigned>(argc)) {
                method = argv[arg++];
            } else if (option == "--test-case" && arg < static_cast<unsigned>(argc)) {
                test_case = alvine::normalizeTestCaseName(argv[arg++]);
            } else if (option == "--filter" && arg < static_cast<unsigned>(argc)) {
                spectral_filter = std::atoi(argv[arg++]);
            } else if (option == "--viscosity" && arg < static_cast<unsigned>(argc)) {
                viscosity = std::atof(argv[arg++]);
            } else if (option == "--integrator" && arg < static_cast<unsigned>(argc)) {
                time_integrator = argv[arg++];
                std::transform(time_integrator.begin(), time_integrator.end(),
                               time_integrator.begin(),
                               [](unsigned char c) { return std::tolower(c); });
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

        if (!alvine::isSupportedTestCase(test_case)) {
            msg << "Invalid --test-case value " << test_case
                << ". Supported for this executable: taylor_green_3d." << endl;
            ippl::Comm->abort();
        }

        if (alvine::normalizeTestCaseName(test_case) != "taylor_green_3d") {
            msg << "VortexInFourier3D requires --test-case taylor_green_3d." << endl;
            ippl::Comm->abort();
        }

        if (spectral_filter < 0 || spectral_filter > 2) {
            msg << "Invalid --filter value " << spectral_filter
                << ". Use 0:no filter, 1:shape function, 2:Hou-Li." << endl;
            ippl::Comm->abort();
        }

        Vector_t<double, Dim> rmin = alvine::domainMinForTestCase<double, Dim>(test_case);
        Vector_t<double, Dim> rmax = alvine::domainMaxForTestCase<double, Dim>(test_case);
        Vector_t<double, Dim> origin = rmin;

        msg << " Grid size: " << nr
            << " No. of particles: " << np
            << " No. of smoke steps: " << nt
            << " dt: " << dt
            << " Method: " << method
            << " Test case: " << test_case
            << " Spectral filter: " << spectral_filter
            << " viscosity: " << viscosity << endl;

        VortexInFourier3DManager<T> manager(nt, nr, np, solver, dump_freq, dt, method,
                                            spectral_filter, viscosity, time_integrator,
                                            rmin, rmax, origin);
        manager.pre_run();
        manager.run(manager.getNt());

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing_vif3d.dat"));
    }
    ippl::finalize();
    return 0;
}
