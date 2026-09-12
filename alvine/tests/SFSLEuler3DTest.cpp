// Exercise the production FFT/source/particle pipeline on one or more MPI ranks.
constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "SFSLEuler3DTest";

#include "Ippl.h"
#include "datatypes.h"
#include "SpectralFSL3DManager.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

#include "SFSL3DTestSupport.hpp"

class EulerProbe : public SFSLProbe {
public:
    using SFSLProbe::SFSLProbe;

    Modes directTGVStep() {
        // Independent DFT of analytically advected TGV strengths, plus the
        // analytic stretching at ORIGINAL positions. No production source,
        // transport, or IMEX kernel is used to construct this reference.
        auto result = snapshot();
        auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result[0].getView());
        auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result[1].getView());
        auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result[2].getView());
        const auto local = result[0].getLayout().getLocalNDIndex();
        const int ng = result[0].getNghost(), n = this->nr_m[0];
        const T pi = std::acos(-1.0), dt = this->dt_m;
        for (int i=ng; i<int(hx.extent(0))-ng; ++i)
        for (int j=ng; j<int(hx.extent(1))-ng; ++j)
        for (int k=ng; k<int(hx.extent(2))-ng; ++k) {
            const int g[3] = {i-ng+local[0].first(), j-ng+local[1].first(), k-ng+local[2].first()};
            int m[3], wave[3];
            T k2 = 0;
            for (int d=0; d<3; ++d) {
                m[d] = g[d] <= n/2 ? g[d] : g[d]-n;
                wave[d] = g[d] == n/2 ? 0 : m[d];
                k2 += wave[d]*wave[d];
            }
            std::complex<T> q[3] = {};
            if (k2 != 0) {
                for (int a=0; a<n; ++a)
                for (int b=0; b<n; ++b)
                for (int c=0; c<n; ++c) {
                    const T x=(a+T(0.5))*2*pi/n, y=(b+T(0.5))*2*pi/n, z=(c+T(0.5))*2*pi/n;
                    const T ux=std::sin(x)*std::cos(y)*std::cos(z);
                    const T uy=-std::cos(x)*std::sin(y)*std::cos(z);
                    const T omega[3] = {-std::cos(x)*std::sin(y)*std::sin(z),
                        -std::sin(x)*std::cos(y)*std::sin(z), 2*std::sin(x)*std::sin(y)*std::cos(z)};
                    const T source[3] = {-std::sin(2*y)*std::sin(2*z)/4,
                        std::sin(2*x)*std::sin(2*z)/4, 0};
                    const auto advected = std::polar(T(1), -(m[0]*(x+dt*ux)+m[1]*(y+dt*uy)+m[2]*z));
                    const auto original = std::polar(T(1), -(m[0]*x+m[1]*y+m[2]*z));
                    for (int d=0; d<3; ++d) q[d] += omega[d]*advected + dt*source[d]*original;
                }
                const T norm = T(n)*n*n*k2*(1+this->viscosity_m*dt*k2);
                for (auto& value : q) value /= norm;
                // The solver retains its end-step solenoidal projection.
                std::complex<T> dot = T(wave[0])*q[0]+T(wave[1])*q[1]+T(wave[2])*q[2];
                for (int d=0; d<3; ++d) q[d] -= T(wave[d])*dot/k2;
            }
            hx(i,j,k) = Kokkos::complex<T>(q[0].real(),q[0].imag());
            hy(i,j,k) = Kokkos::complex<T>(q[1].real(),q[1].imag());
            hz(i,j,k) = Kokkos::complex<T>(q[2].real(),q[2].imag());
        }
        Kokkos::deep_copy(result[0].getView(),hx);
        Kokkos::deep_copy(result[1].getView(),hy);
        Kokkos::deep_copy(result[2].getView(),hz);
        return result;
    }
};

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
            EulerProbe pipeline(1, nr, 512, solver, 0, dt, "sfsl_euler_order_test",
                                0, nu, "euler", lower, upper, lower, 0);
            pipeline.pre_run();
            auto expected = pipeline.directTGVStep();
            pipeline.run(1);
            check("advection first, original-grid stretching, then implicit diffusion",
                  pipeline.modeError(expected));

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
            check("three complete Euler steps preserve the post-advection IMEX state",
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
