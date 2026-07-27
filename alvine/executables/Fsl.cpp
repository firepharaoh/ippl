// Forward Semi-Lagrangian Test
// Usage:
//   srun ./FSL <nx> <ny> <Np_unused> <Nt> <stype> <dump_freq> [--dt value]
//        [--method label] [--test-case taylor_green_2d] [--integrator leapfrog|rk4]
//        --overallocate 1.0 --info 5

constexpr unsigned Dim = 2;
using T                = double;
const char* TestName   = "FSL";

#include "Ippl.h"

#include <algorithm>
#include <cctype>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "datatypes.h"

#include "Utility/IpplTimings.h"

#include "Manager/PicManager.h"
#include "FslManager.h"
#include "VortexDistributions.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        unsigned arg = 1;

        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        int np = std::atoi(argv[arg++]);   // unused for FSL, but keep argument format
        int nt = std::atoi(argv[arg++]);

        std::string solver = argv[arg++];
        int dump_freq = std::atoi(argv[arg++]);
        double dt = 0.05;
        std::string method = "fsl";
        std::string test_case = "taylor_green_2d";
        std::string time_integrator = "leapfrog";
        while (arg < static_cast<unsigned>(argc)) {
            const std::string option = argv[arg++];
            if (option == "--dt" && arg < static_cast<unsigned>(argc)) {
                dt = std::atof(argv[arg++]);
            } else if (option == "--method" && arg < static_cast<unsigned>(argc)) {
                method = argv[arg++];
            } else if (option == "--test-case" && arg < static_cast<unsigned>(argc)) {
                test_case = alvine::normalizeTestCaseName(argv[arg++]);
            } else if (option == "--integrator" && arg < static_cast<unsigned>(argc)) {
                time_integrator = argv[arg++];
                std::transform(time_integrator.begin(), time_integrator.end(),
                               time_integrator.begin(),
                               [](unsigned char c) { return std::tolower(c); });
            } else if (option == "--dt") {
                msg << "Missing value after --dt" << endl;
                ippl::Comm->abort();
            } else if (option == "--method") {
                msg << "Missing value after --method" << endl;
                ippl::Comm->abort();
            } else if (option == "--test-case") {
                msg << "Missing value after --test-case" << endl;
                ippl::Comm->abort();
            } else if (option == "--integrator") {
                msg << "Missing value after --integrator" << endl;
                ippl::Comm->abort();
            }
        }

        if (!alvine::isSupportedTestCase(test_case)) {
            msg << "Invalid --test-case value " << test_case
                << ". Currently supported: taylor_green_2d." << endl;
            ippl::Comm->abort();
        }

        if (time_integrator != "leapfrog" && time_integrator != "rk4") {
            msg << "Invalid --integrator value " << time_integrator
                << ". Use leapfrog or rk4." << endl;
            ippl::Comm->abort();
        }

        msg << " Grid size: " << nr
            << " No. of virtual particles per step: " << nr[0] * nr[1]
            << " No. of time steps: " << nt << " dt: " << dt
            << " Method: " << method
            << " Test case: " << test_case
            << " Time integrator: " << time_integrator << endl;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        Vector_t<double, Dim> rmin = alvine::domainMinForTestCase<double, Dim>(test_case);
        Vector_t<double, Dim> rmax = alvine::domainMaxForTestCase<double, Dim>(test_case);
        Vector_t<double, Dim> origin = rmin;
        Vector_t<double, Dim> hr = (rmax - rmin) / nr;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        const bool isAllPeriodic = true;

        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);

        FSLManager<T, Dim, Band> manager(
            nt,
            nr,
            np,
            solver,
            dump_freq,
            dt,
            method,
            time_integrator,
            rmin,
            rmax,
            origin,
            FL,
            mesh
        );

        manager.pre_run();
        manager.run(manager.getNt());

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing_fsl.dat"));
    }

    ippl::finalize();

    return 0;
}
