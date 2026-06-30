// Spectral Forward Semi-Lagrangian Test
// Usage:
//   srun ./SpectralFSL <nx> <ny> <Np_unused> <Nt> <stype> <dump_freq> [--dt value]
//        [--method label] --overallocate 1.0 --info 5

constexpr unsigned Dim = 2;
using T                = double;
const char* TestName   = "SpectralFSL";

#include "Ippl.h"

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
#include "SpectralFSLManager.h"
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

        int np = std::atoi(argv[arg++]);   // unused for SFSL, but keep argument format
        int nt = std::atoi(argv[arg++]);

        std::string solver = argv[arg++];
        int dump_freq = std::atoi(argv[arg++]);
        double dt = 0.05;
        std::string method = "sfsl";
        while (arg < static_cast<unsigned>(argc)) {
            const std::string option = argv[arg++];
            if (option == "--dt" && arg < static_cast<unsigned>(argc)) {
                dt = std::atof(argv[arg++]);
            } else if (option == "--method" && arg < static_cast<unsigned>(argc)) {
                method = argv[arg++];
            } else if (option == "--dt") {
                msg << "Missing value after --dt" << endl;
                ippl::Comm->abort();
            } else if (option == "--method") {
                msg << "Missing value after --method" << endl;
                ippl::Comm->abort();
            }
        }

        msg << " Grid size: " << nr
            << " No. of virtual particles per step: " << nr[0] * nr[1]
            << " No. of time steps: " << nt << " dt: " << dt
            << " Method: " << method << endl;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        Vector_t<double, Dim> rmin(0.0);
        // Taylor-Green vortex is typically defined on a 2pi x 2pi domain.
        Vector_t<double, Dim> rmax(2.0 * std::acos(-1.0));
        Vector_t<double, Dim> origin = rmin;
        Vector_t<double, Dim> hr = (rmax - rmin) / nr;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        const bool isAllPeriodic = true;

        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);

        SpectralFSLManager<T, Dim, Band> manager(
            nt,
            nr,
            np,
            solver,
            dump_freq,
            dt,
            method,
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
        IpplTimings::print(std::string("timing_spectral_fsl.dat"));
    }

    ippl::finalize();

    return 0;
}
