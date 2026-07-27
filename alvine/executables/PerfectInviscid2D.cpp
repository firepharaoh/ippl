// Perfect 2D inviscid run shell.
// This target is intentionally empty for now; it only enters the Alvine build
// ecosystem with the same basic executable shape as the other 2D drivers.

constexpr unsigned Dim = 2;
using T                = double;
const char* TestName   = "PerfectInviscid2D";

#include "Ippl.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "datatypes.h"

#include "Utility/IpplTimings.h"

#include "PerfectInviscid2DManager.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        if (argc < 7) {
            msg << "Usage: PerfectInviscid2D <nx> <ny> <Np> <Nt> <stype> <dump_freq> "
                   "[--dt value] [--method label] [--test-case taylor_green_2d]"
                << endl;
            ippl::Comm->abort();
        }

        unsigned arg = 1;

        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; ++d) {
            nr[d] = std::atoi(argv[arg++]);
        }

        const int np          = std::atoi(argv[arg++]);
        const int nt          = std::atoi(argv[arg++]);
        std::string stype       = argv[arg++];
        const int dump_freq   = std::atoi(argv[arg++]);

        double dt          = 0.05;
        std::string method = "perfect_2d_inviscid";
        std::string test_case = "taylor_green_2d";

        while (arg < static_cast<unsigned>(argc)) {
            const std::string option = argv[arg++];
            if (option == "--dt" && arg < static_cast<unsigned>(argc)) {
                dt = std::atof(argv[arg++]);
            } else if (option == "--method" && arg < static_cast<unsigned>(argc)) {
                method = argv[arg++];
            } else if (option == "--test-case" && arg < static_cast<unsigned>(argc)) {
                test_case = alvine::normalizeTestCaseName(argv[arg++]);
            } else if (option == "--test-case") {
                msg << "Missing value after --test-case" << endl;
                ippl::Comm->abort();
            } else if ((option == "--overallocate" || option == "-b" || option == "--info"
                        || option == "-i" || option == "--timer-fences")
                       && arg < static_cast<unsigned>(argc)) {
                ++arg;
            } else if (option == "--debug" || option == "-g") {
                // Already handled by ippl::initialize().
            } else {
                msg << "Unknown option: " << option << endl;
                ippl::Comm->abort();
            }
        }

        if (!alvine::isSupportedTestCase(test_case)) {
            msg << "Invalid --test-case value " << test_case
                << ". Currently supported: taylor_green_2d." << endl;
            ippl::Comm->abort();
        }

        msg << " Grid size: " << nr
            << " No. of particles: " << np
            << " No. of time steps: " << nt
            << " dt: " << dt
            << " Solver type: " << stype
            << " Dump frequency: " << dump_freq
            << " Method: " << method
            << " Test case: " << test_case
            << endl;

        Vector_t<double, Dim> rmin = alvine::domainMinForTestCase<double, Dim>(test_case);
        Vector_t<double, Dim> rmax = alvine::domainMaxForTestCase<double, Dim>(test_case);
        Vector_t<double, Dim> origin = rmin;

        PerfectInviscid2DManager<T, Dim> manager(
            nt, nr, np, stype, dump_freq, dt, method, rmin, rmax, origin);

        manager.pre_run();

        manager.dump();

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing_perfect_2d_inviscid.dat"));
    }

    ippl::finalize();

    return 0;
}
