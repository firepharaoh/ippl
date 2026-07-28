#ifndef IPPL_ALVINE_DIAGNOSTICS_ALVINESPECTRALDIAGNOSTICS3D_HPP
#define IPPL_ALVINE_DIAGNOSTICS_ALVINESPECTRALDIAGNOSTICS3D_HPP

    double computeSpectralEnergy3D() {
        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto& layout       = ux_hat_m.getLayout();
        auto& mesh         = ux_hat_m.get_mesh();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = ux_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();
        const T Lx   = dx[0] * Nx;
        const T Ly   = dx[1] * Ny;
        const T Lz   = dx[2] * Nz;
        const T volume = Lx * Ly * Lz;

        double localEnergy = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "compute_spectral_energy_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ux.extent(0)) - nghost,
                         static_cast<int>(ux.extent(1)) - nghost,
                         static_cast<int>(ux.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& lsum) {
                const auto uxMode = ux(i, j, k);
                const auto uyMode = uy(i, j, k);
                const auto uzMode = uz(i, j, k);

                const double ux2 = uxMode.real() * uxMode.real()
                                 + uxMode.imag() * uxMode.imag();
                const double uy2 = uyMode.real() * uyMode.real()
                                 + uyMode.imag() * uyMode.imag();
                const double uz2 = uzMode.real() * uzMode.real()
                                 + uzMode.imag() * uzMode.imag();

                lsum += ux2 + uy2 + uz2;
            },
            localEnergy);

        double globalEnergy = 0.0;
        ippl::Comm->allreduce(localEnergy, globalEnergy, 1, std::plus<double>());

        // The 3D NUFFT type-1 modes represent Fourier integrals because the
        // particle vorticity stores vorticity times particle volume. The 3D
        // Biot-Savart step preserves that scaling, so u_hat_raw = volume * u_hat.
        // Parseval for Fourier-series coefficients is:
        //   E = 0.5 * volume * sum(|u_hat|^2)
        // Therefore with raw modes:
        //   E = 0.5 * sum(|u_hat_raw|^2) / volume.
        return T(0.5) * globalEnergy / volume;
    }

    double computeSpectralEnergy() {
        return computeSpectralEnergy3D();
    }

    double computeSpectralEnstrophy3D() {
        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        auto& mesh         = omega_x_hat_m.get_mesh();
        const auto& domain = omega_x_hat_m.getLayout().getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = omega_x_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();

        const T Lx     = dx[0] * Nx;
        const T Ly     = dx[1] * Ny;
        const T Lz     = dx[2] * Nz;
        const T volume = Lx * Ly * Lz;

        double localEnstrophy = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "compute_spectral_enstrophy_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ox.extent(0)) - nghost,
                         static_cast<int>(ox.extent(1)) - nghost,
                         static_cast<int>(ox.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& lsum) {
                const auto oxMode = ox(i, j, k);
                const auto oyMode = oy(i, j, k);
                const auto ozMode = oz(i, j, k);

                const double ox2 = oxMode.real() * oxMode.real()
                                 + oxMode.imag() * oxMode.imag();
                const double oy2 = oyMode.real() * oyMode.real()
                                 + oyMode.imag() * oyMode.imag();
                const double oz2 = ozMode.real() * ozMode.real()
                                 + ozMode.imag() * ozMode.imag();

                lsum += ox2 + oy2 + oz2;
            },
            localEnstrophy);

        double globalEnstrophy = 0.0;
        ippl::Comm->allreduce(localEnstrophy, globalEnstrophy, 1, std::plus<double>());

        // The 3D NUFFT type-1 modes represent Fourier integrals:
        // omega_hat_raw = volume * omega_hat. With Parseval,
        // Z = 0.5 * volume * sum(|omega_hat|^2), hence
        // Z = 0.5 * sum(|omega_hat_raw|^2) / volume.
        return T(0.5) * globalEnstrophy / volume;
    }

    double computeSpectralEnstrophy() {
        return computeSpectralEnstrophy3D();
    }

    double computeSpectralDivergenceL23D() {
        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto& layout = ux_hat_m.getLayout();
        auto& mesh   = ux_hat_m.get_mesh();

        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = ux_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();

        const T Lx     = dx[0] * Nx;
        const T Ly     = dx[1] * Ny;
        const T Lz     = dx[2] * Nz;
        const T volume = Lx * Ly * Lz;

        const T twoPi = T(2.0 * std::acos(-1.0));
        const Kokkos::complex<T> imag(0.0, 1.0);

        double localDiv2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "compute_spectral_divergence_l2_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ux.extent(0)) - nghost,
                         static_cast<int>(ux.extent(1)) - nghost,
                         static_cast<int>(ux.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& lsum) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();

                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const int mz = (gz <= Nz / 2) ? gz : gz - Nz;

                const bool notMidX = (gx != Nx / 2);
                const bool notMidY = (gy != Ny / 2);
                const bool notMidZ = (gz != Nz / 2);

                const T kx = notMidX * twoPi * mx / Lx;
                const T ky = notMidY * twoPi * my / Ly;
                const T kz = notMidZ * twoPi * mz / Lz;

                const auto divHat =
                    imag * (kx * ux(i, j, k) + ky * uy(i, j, k) + kz * uz(i, j, k));
                lsum += divHat.real() * divHat.real() + divHat.imag() * divHat.imag();
            },
            localDiv2);

        double globalDiv2 = 0.0;
        ippl::Comm->allreduce(localDiv2, globalDiv2, 1, std::plus<double>());

        // div_hat is built from raw velocity modes, and u_hat_raw = volume * u_hat.
        // Therefore ||div u||_L2^2 = sum(|div_hat_raw|^2) / volume.
        return std::sqrt(globalDiv2 / volume);
    }

    double computeSpectralDivergenceL2() {
        return computeSpectralDivergenceL23D();
    }

    void logSpectralDiagnostics3D(
        const std::string& filename = "spectral_diagnostics_3d.csv") {
        const double energy = computeSpectralEnergy3D();
        const double enstrophy = computeSpectralEnstrophy3D();
        const double divergenceL2 = computeSpectralDivergenceL23D();

        const double pi = std::acos(-1.0);
        const double exactEnergy = pi * pi * pi;
        const double exactEnstrophy = 3.0 * pi * pi * pi;
        const double energyRelError =
            std::abs(energy - exactEnergy) / std::max(exactEnergy, 1e-30);
        const double enstrophyRelError =
            std::abs(enstrophy - exactEnstrophy) / std::max(exactEnstrophy, 1e-30);
        const double velocityL2 = std::sqrt(std::max(2.0 * energy, 1e-30));
        const double divergenceNormalized = divergenceL2 / velocityL2;

        if (ippl::Comm->rank() == 0) {
            const bool firstWrite = !spectral_3d_diagnostics_initialized_m;
            std::ofstream out(diagnosticFileName(filename),
                              firstWrite ? std::ios::out : std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            if (firstWrite) {
                out << "method,dt,step,time,spectral_energy,spectral_energy_rel_error,"
                    << "spectral_enstrophy,spectral_enstrophy_rel_error,"
                    << "spectral_divergence_l2,spectral_divergence_normalized\n";
            }

            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << energy << "," << energyRelError << ","
                << enstrophy << "," << enstrophyRelError << ","
                << divergenceL2 << "," << divergenceNormalized << "\n";

            Inform m("spectral_diagnostics_3d ");
            m << "energy = " << energy
              << ", energyRelError = " << energyRelError
              << ", enstrophy = " << enstrophy
              << ", enstrophyRelError = " << enstrophyRelError
              << ", divergenceL2 = " << divergenceL2
              << ", divergenceNormalized = " << divergenceNormalized << endl;
        }

        spectral_3d_diagnostics_initialized_m = true;
    }
#endif
