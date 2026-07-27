// Vortex In Fourier Test
//   Usage:
//     srun ./VortexInFourier
//                  <nx> [<ny>...] <Np> <Nt> <stype> <dump_freq> [remesh_freq]
//                  --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = No. of vortex particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     dump_freq= Dumping frequency of particle output
//     remesh_freq= Remeshing frequency. Default 0 disables remeshing.
//     --dt     = Optional timestep. Default 0.05.
//     --method = Optional method label used in diagnostic CSV filenames. Default vif.
//     --test-case = Optional test case: taylor_green_2d. Default taylor_green_2d.
//     --filter = Optional spectral filter: 0 none, 1 shape function, 2 Hou-Li. Default 0.
//     --integrator = Optional time integrator: leapfrog or rk4. Default leapfrog.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     makdir build_*/alvine/data
//     chmod +x data
//     srun ./VortexInFourier 128 128 10000 100 FFT 100 10 --overallocate 1.0 --info 5
//     to build, call 
//          make VortexInFourier 
//     in the build directory to only build this target

constexpr unsigned Dim = 2;
using T                = double;
const char* TestName   = "VortexInFourier";

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
#include "VortexInFourierManager.h"
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

        int np = std::atoi(argv[arg++]);
        int nt  = std::atoi(argv[arg++]);
        std::string solver = argv[arg++];
        int dump_freq  = std::atoi(argv[arg++]);
        int remesh_freq = 0;
        if (arg < static_cast<unsigned>(argc) && std::string(argv[arg]).rfind("--", 0) != 0) {
            remesh_freq = std::atoi(argv[arg++]);
        }
        double dt = 0.05;
        std::string method = "vif";
        std::string test_case = "taylor_green_2d";
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
            } else if (option == "--filter") {
                msg << "Missing value after --filter" << endl;
                ippl::Comm->abort();
            } else if (option == "--integrator") {
                msg << "Missing value after --integrator" << endl;
                ippl::Comm->abort();
            } else if (option == "--viscosity" && arg < static_cast<unsigned>(argc)) {
                viscosity = std::atof(argv[arg++]);
                if(viscosity < 0.0) {
                    msg << "Viscosity must be non-negative." << endl;
                    ippl::Comm->abort();
                }
            } else if (option == "--viscosity") {
                msg << "Missing value after --viscosity" << endl;
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

        if (spectral_filter < 0 || spectral_filter > 2) {
            msg << "Invalid --filter value " << spectral_filter
                << ". Use 0:no filter, 1:shape function, 2:Hou-Li." << endl;
            ippl::Comm->abort();
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
        
        msg << " Grid size: " << nr << " No. of particles: " << np
            << " No. of time steps: " << nt << " dt: " << dt
            << " Method: " << method << " Remesh frequency: " << remesh_freq
            << " Test case: " << test_case
            << " Spectral filter: " << spectral_filter
            << " viscosity: " << viscosity
            << " Time integrator: " << time_integrator << endl;
        
        // ===== CRITICAL: Create mesh and layout with proper MPI decomposition =====
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

        // Create the mesh and layout with MPI communicator
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        
        // Now create manager WITH the layout info
        VortexInFourierManager<T, Dim, Band> manager(nt, nr, np, solver, dump_freq, remesh_freq,
                                                   dt, method, spectral_filter, viscosity,
                                                   time_integrator, rmin, rmax, origin, FL, mesh);

        manager.pre_run();
        manager.run(manager.getNt());
        
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
