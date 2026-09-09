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

class EulerProbe : public SpectralFSL3DManager<T> {
public:
    using SpectralFSL3DManager<T>::SpectralFSL3DManager;

    void setShear() {
        // u=(sin(y),0,0), omega=(0,0,-cos(y)): stretching and advection
        // of omega vanish exactly, even though particles move in x.
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto w = pc->omega.getView();
        auto wx = pc->omega_x.getView();
        auto wy = pc->omega_y.getView();
        auto wz = pc->omega_z.getView();
        const T volume = particleVolume3D();
        Kokkos::parallel_for("test_initialize_shear", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                w(p)[0] = wx(p) = 0;
                w(p)[1] = wy(p) = 0;
                w(p)[2] = wz(p) = -Kokkos::cos(R(p)[1]) * volume;
            });
        Kokkos::fence();
        scatterAndSolveCurrentParticles3D();
    }

    double gridError(const bool shear, const T amplitude, const T sourceAmplitude = 0) {
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        auto w = this->fcontainer_m->getOmegaField().getView();
        const int ng = this->fcontainer_m->getOmegaField().getNghost();
        const auto local = this->fcontainer_m->getFL().getLocalNDIndex();
        const auto dx = this->hr_m;
        T localError = 0;
        Kokkos::parallel_reduce("test_euler_grid_error", ippl::getRangePolicy(w, ng),
            KOKKOS_LAMBDA(const int i, const int j, const int k, T& error) {
                const T x = (i - ng + local[0].first() + T(0.5)) * dx[0];
                const T y = (j - ng + local[1].first() + T(0.5)) * dx[1];
                const T z = (k - ng + local[2].first() + T(0.5)) * dx[2];
                Vector_t<T,3> expected(0);
                if (shear) {
                    expected[2] = -amplitude * Kokkos::cos(y);
                } else {
                    expected = TaylorGreen3D<T>::vorticity(x,y,z) * amplitude;
                    // Independently differentiated TGV: Sx=-sin(2y)sin(2z)/4,
                    // Sy=sin(2x)sin(2z)/4, Sz=0. These modes have k^2=8.
                    expected[0] -= sourceAmplitude * Kokkos::sin(2*y) * Kokkos::sin(2*z) / 4;
                    expected[1] += sourceAmplitude * Kokkos::sin(2*x) * Kokkos::sin(2*z) / 4;
                }
                for (unsigned d=0; d<3; ++d) {
                    error = Kokkos::max(error, Kokkos::abs(w(i,j,k)[d] - expected[d]));
                }
            }, Kokkos::Max<T>(localError));
        double globalError = 0;
        MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_MAX,
                      ippl::Comm->getCommunicator());
        return globalError;
    }

    double energy() { return this->computeSpectralEnergy3D(); }
    double enstrophy() { return this->computeSpectralEnstrophy3D(); }
};

void check(const char* label, double error, double tolerance = 2e-7) {
    if (ippl::Comm->rank() == 0) {
        std::cout << label << ": error=" << error << std::endl;
    }
    if (!std::isfinite(error) || error > tolerance) {
        throw std::runtime_error(label);
    }
}

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
            EulerProbe manager(3, nr, 512, solver, 0, dt, "sfsl_euler_test",
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

        EulerProbe adaptive(1, nr, 512, solver, 0, 0.1, "sfsl_euler_lcfl_test",
                            0, 0.2, "euler", lower, upper, lower, 0);
        adaptive.pre_run();
        adaptive.setShear();
        adaptive.setAdaptiveLCFL(true);
        adaptive.setLCFL(0.01);
        adaptive.setFinalTime(0.005);
        adaptive.run(1);
        check("final-time clipped dt also used by diffusion",
              adaptive.gridError(true, 1/(1+0.2*0.005)));

        EulerProbe lcfl(1, nr, 512, solver, 0, 0.1, "sfsl_euler_lcfl_active_test",
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
