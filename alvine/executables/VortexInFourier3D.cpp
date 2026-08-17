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
            msg << "Usage: VortexInFourier3D nx ny nz np nt solver dump_freq [remesh_freq] "
                   "[--dt dt] [--method label] [--filter 0|1|2|3] "
                   "[--test-case taylor_green_3d] [--viscosity 0.0] "
                   "[--remesh-freq frequency] [--diagnostics-freq frequency] "
                   "[--remesh-round-trip] [--overallocate value] [--info level]"
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
        int remesh_freq = 0;
        int diagnostics_freq = 1;
        bool remesh_round_trip = false;
        if (arg < static_cast<unsigned>(argc) && std::string(argv[arg]).rfind("--", 0) != 0) {
            remesh_freq = std::atoi(argv[arg++]);
        }

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
                if (viscosity < 0.0) {
                    msg << "Viscosity must be non-negative." << endl;
                    ippl::Comm->abort();
                }
            } else if (option == "--remesh-freq" && arg < static_cast<unsigned>(argc)) {
                remesh_freq = std::atoi(argv[arg++]);
            } else if (option == "--diagnostics-freq" && arg < static_cast<unsigned>(argc)) {
                diagnostics_freq = std::atoi(argv[arg++]);
            } else if (option == "--integrator" && arg < static_cast<unsigned>(argc)) {
                time_integrator = argv[arg++];
                std::transform(time_integrator.begin(), time_integrator.end(),
                               time_integrator.begin(),
                               [](unsigned char c) { return std::tolower(c); });
            } else if (option == "--remesh-round-trip") {
                remesh_round_trip = true;
            } else if (option == "--dt") {
                msg << "Missing value after --dt" << endl;
                ippl::Comm->abort();
            } else if (option == "--method") {
                msg << "Missing value after --method" << endl;
                ippl::Comm->abort();
            } else if (option == "--test-case") {
                msg << "Missing value after --test-case" << endl;
                ippl::Comm->abort();
            } else if (option == "--filter") {
                msg << "Missing value after --filter" << endl;
                ippl::Comm->abort();
            } else if (option == "--viscosity") {
                msg << "Missing value after --viscosity" << endl;
                ippl::Comm->abort();
            } else if (option == "--remesh-freq") {
                msg << "Missing value after --remesh-freq" << endl;
                ippl::Comm->abort();
            } else if (option == "--diagnostics-freq") {
                msg << "Missing value after --diagnostics-freq" << endl;
                ippl::Comm->abort();
            } else if (option == "--integrator") {
                msg << "Missing value after --integrator" << endl;
                ippl::Comm->abort();
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

        if (spectral_filter < 0 || spectral_filter > 3) {
            msg << "Invalid --filter value " << spectral_filter
                << ". Use 0:no filter, 1:shape function, 2:Hou-Li, 3:2/3 cutoff."
                << endl;
            ippl::Comm->abort();
        }

        if (time_integrator != "leapfrog" && time_integrator != "rk4") {
            msg << "Invalid --integrator value " << time_integrator
                << ". Use leapfrog or rk4." << endl;
            ippl::Comm->abort();
        }

        if (remesh_freq < 0) {
            msg << "Invalid --remesh-freq value " << remesh_freq
                << ". Use 0 to disable remeshing or a positive frequency." << endl;
            ippl::Comm->abort();
        }

        if (diagnostics_freq < 0) {
            msg << "Invalid --diagnostics-freq value " << diagnostics_freq
                << ". Use 0 to disable diagnostics or a positive frequency." << endl;
            ippl::Comm->abort();
        }

        if (remesh_round_trip && spectral_filter != 0) {
            msg << "--remesh-round-trip requires --filter 0." << endl;
            ippl::Comm->abort();
        }

        if (remesh_round_trip && viscosity != 0.0) {
            msg << "--remesh-round-trip requires --viscosity 0." << endl;
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
            << " viscosity: " << viscosity
            << " Remesh frequency: " << remesh_freq
            << " Diagnostics frequency: " << diagnostics_freq
            << " Remesh round-trip: " << (remesh_round_trip ? "true" : "false")
            << " Time integrator: " << time_integrator << endl;

        VortexInFourier3DManager<T> manager(nt, nr, np, solver, dump_freq, dt, method,
                                            spectral_filter, viscosity, time_integrator,
                                            rmin, rmax, origin, remesh_freq,
                                            diagnostics_freq);
        manager.pre_run();
        if (remesh_round_trip) {
            manager.runIsolatedRemeshRoundTripTest3D();
        } else {
            manager.run(manager.getNt());
        }

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing_vif3d.dat"));
    }
    ippl::finalize();
    return 0;
}
