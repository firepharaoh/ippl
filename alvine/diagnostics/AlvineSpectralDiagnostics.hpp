#ifndef IPPL_ALVINE_DIAGNOSTICS_ALVINESPECTRALDIAGNOSTICS_HPP
#define IPPL_ALVINE_DIAGNOSTICS_ALVINESPECTRALDIAGNOSTICS_HPP

    double computeSpectralDivergenceL2() {
        if constexpr (Dim == 2) {
            auto ux = ux_hat_m.getView();
            auto uy = uy_hat_m.getView();

            auto& layout = ux_hat_m.getLayout();
            auto& mesh   = ux_hat_m.get_mesh();

            const auto& lDom   = layout.getLocalNDIndex();
            const auto& domain = layout.getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = ux_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T Lx   = dx[0] * Nx;
            const T Ly   = dx[1] * Ny;

            const T twoPi = T(2.0 * std::acos(-1.0));
            const Kokkos::complex<T> imag(0.0, 1.0);

            double localDiv2 = 0.0;
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_reduce(
                "compute_spectral_divergence_l2",
                policy_type({nghost, nghost},
                            {static_cast<int>(ux.extent(0)) - nghost,
                             static_cast<int>(ux.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                    const int gx = i - nghost + lDom[0].first();
                    const int gy = j - nghost + lDom[1].first();

                    const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                    const int my = (gy <= Ny / 2) ? gy : gy - Ny;

                    const bool notMidX = (gx != Nx / 2);
                    const bool notMidY = (gy != Ny / 2);

                    const T kx = notMidX * twoPi * mx / Lx;
                    const T ky = notMidY * twoPi * my / Ly;

                    const auto divHat = imag * (kx * ux(i, j) + ky * uy(i, j));
                    lsum += divHat.real() * divHat.real() + divHat.imag() * divHat.imag();
                },
                localDiv2);

            double globalDiv2 = 0.0;
            ippl::Comm->reduce(localDiv2, globalDiv2, 1, std::plus<double>());

            const double N = static_cast<double>(Nx) * static_cast<double>(Ny);
            return std::sqrt(globalDiv2 / N);
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralDivergenceL2 is implemented for 2D VIC only");
        }
    }

    double computeSpectralEnergy() {
        if constexpr (Dim == 2) {
            auto ux = ux_hat_m.getView();
            auto uy = uy_hat_m.getView();

            auto& layout       = ux_hat_m.getLayout();
            auto& mesh         = ux_hat_m.get_mesh();
            const auto& domain = layout.getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = ux_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T Lx   = dx[0] * Nx;
            const T Ly   = dx[1] * Ny;
            const T area = Lx * Ly;

            double localEnergy = 0.0;
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_reduce(
                "compute_spectral_energy",
                policy_type({nghost, nghost},
                            {static_cast<int>(ux.extent(0)) - nghost,
                             static_cast<int>(ux.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                    const auto uxMode = ux(i, j);
                    const auto uyMode = uy(i, j);

                    const double ux2 =
                        uxMode.real() * uxMode.real() + uxMode.imag() * uxMode.imag();
                    const double uy2 =
                        uyMode.real() * uyMode.real() + uyMode.imag() * uyMode.imag();

                    lsum += 0.5 * (ux2 + uy2);
                },
                localEnergy);

            double globalEnergy = 0.0;
            ippl::Comm->allreduce(localEnergy, globalEnergy, 1, std::plus<double>());

            // ux_hat_m and uy_hat_m are Fourier-series coefficients because
            // computeSpectralVelocityModes divides the raw type-1 NUFFT modes by
            // the domain area. Parseval therefore contributes one factor of area.
            return area * globalEnergy;
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralEnergy is implemented for 2D VIC only");
        }
    }

    double computeSpectralEnstrophy() {
        if constexpr (Dim == 2) {
            auto omega = omega_hat_m.getView();

            auto& mesh         = omega_hat_m.get_mesh();
            const auto& domain = omega_hat_m.getLayout().getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = omega_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T Lx   = dx[0] * Nx;
            const T Ly   = dx[1] * Ny;
            const T area = Lx * Ly;

            double localEnstrophy = 0.0;
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_reduce(
                "compute_spectral_enstrophy",
                policy_type({nghost, nghost},
                            {static_cast<int>(omega.extent(0)) - nghost,
                             static_cast<int>(omega.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                    const auto omegaMode = omega(i, j);
                    const double omega2 =
                        omegaMode.real() * omegaMode.real() + omegaMode.imag() * omegaMode.imag();
                    lsum += 0.5 * omega2;
                },
                localEnstrophy);

            double globalEnstrophy = 0.0;
            ippl::Comm->allreduce(localEnstrophy, globalEnstrophy, 1, std::plus<double>());

            // Dividing raw type-1 modes by area gives Fourier-series coefficients,
            // so Parseval contributes one inverse factor of area.
            return globalEnstrophy / area;
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralEnstrophy is implemented for 2D VIC only");
        }
    }

    std::vector<VorticitySpectrumShell> computeSpectralVorticitySpectrum() {
        if constexpr (Dim == 2) {
            auto omega = omega_hat_m.getView();

            auto& layout       = omega_hat_m.getLayout();
            auto& mesh         = omega_hat_m.get_mesh();
            const auto& lDom   = layout.getLocalNDIndex();
            const auto& domain = layout.getDomain();
            const auto& dx     = mesh.getMeshSpacing();
            const int nghost   = omega_hat_m.getNghost();

            const int Nx = domain[0].length();
            const int Ny = domain[1].length();
            const T area = (dx[0] * Nx) * (dx[1] * Ny);
            const int maxShell =
                static_cast<int>(std::floor(std::hypot(Nx / 2.0, Ny / 2.0)));
            const int shellCount = maxShell + 1;

            Kokkos::View<double*> localSpectrum("local_vorticity_spectrum", shellCount);
            Kokkos::View<std::uint64_t*> localModeCounts(
                "local_vorticity_spectrum_mode_counts", shellCount);
            Kokkos::deep_copy(localSpectrum, 0.0);
            Kokkos::deep_copy(localModeCounts, std::uint64_t(0));

            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for(
                "compute_spectral_vorticity_spectrum",
                policy_type({nghost, nghost},
                            {static_cast<int>(omega.extent(0)) - nghost,
                             static_cast<int>(omega.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    const int gx = i - nghost + lDom[0].first();
                    const int gy = j - nghost + lDom[1].first();

                    const int mx    = (gx <= Nx / 2) ? gx : gx - Nx;
                    const int my    = (gy <= Ny / 2) ? gy : gy - Ny;
                    const int shell =
                        static_cast<int>(Kokkos::floor(Kokkos::sqrt(T(mx * mx + my * my))));

                    const auto omegaMode = omega(i, j);
                    const double omega2 =
                        omegaMode.real() * omegaMode.real() + omegaMode.imag() * omegaMode.imag();

                    Kokkos::atomic_add(&localSpectrum(shell), 0.5 * omega2 / area);
                    Kokkos::atomic_add(&localModeCounts(shell), std::uint64_t(1));
                });
            Kokkos::fence();

            auto hostSpectrum =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localSpectrum);
            auto hostModeCounts =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localModeCounts);

            std::vector<double> localValues(shellCount);
            std::vector<double> globalValues(shellCount);
            std::vector<std::uint64_t> localCounts(shellCount);
            std::vector<std::uint64_t> globalCounts(shellCount);
            for (int shell = 0; shell < shellCount; ++shell) {
                localValues[shell] = hostSpectrum(shell);
                localCounts[shell] = hostModeCounts(shell);
            }

            ippl::Comm->allreduce(
                localValues.data(), globalValues.data(), shellCount, std::plus<double>());
            ippl::Comm->allreduce(
                localCounts.data(), globalCounts.data(), shellCount, std::plus<std::uint64_t>());

            const int completeShellLimit = std::min(Nx, Ny) / 2;
            std::vector<VorticitySpectrumShell> spectrum(shellCount);
            for (int shell = 0; shell < shellCount; ++shell) {
                spectrum[shell].enstrophy = globalValues[shell];
                spectrum[shell].modeCount = globalCounts[shell];
                spectrum[shell].complete  = shell < completeShellLimit;
            }
            return spectrum;
        } else {
            throw std::runtime_error(
                "AlvineManager::computeSpectralVorticitySpectrum is implemented for 2D VIC only");
        }
    }

#endif
