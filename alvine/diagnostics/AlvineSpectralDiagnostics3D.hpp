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

        // ux_hat_m, uy_hat_m, and uz_hat_m are Fourier-series velocity
        // coefficients. Parseval contributes one factor of volume.
        return T(0.5) * volume * globalEnergy;
    }

    double computeSpectralEnergy() {
        return computeSpectralEnergy3D();
    }

    double computeSpectralEnstrophy3D() {
        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        auto& layout       = omega_x_hat_m.getLayout();
        auto& mesh         = omega_x_hat_m.get_mesh();
        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = omega_x_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();

        const T Lx     = dx[0] * Nx;
        const T Ly     = dx[1] * Ny;
        const T Lz     = dx[2] * Nz;
        const T volume = Lx * Ly * Lz;
        const T twoPi  = T(2.0 * std::acos(-1.0));

        double localEnstrophy = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "compute_spectral_enstrophy_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ox.extent(0)) - nghost,
                         static_cast<int>(ox.extent(1)) - nghost,
                         static_cast<int>(ox.extent(2)) - nghost}),
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
                const T k2 = kx * kx + ky * ky + kz * kz;

                const auto oxMode = ox(i, j, k);
                const auto oyMode = oy(i, j, k);
                const auto ozMode = oz(i, j, k);

                const double ox2 = k2 * k2 * (oxMode.real() * oxMode.real()
                                            + oxMode.imag() * oxMode.imag());
                const double oy2 = k2 * k2 * (oyMode.real() * oyMode.real()
                                            + oyMode.imag() * oyMode.imag());
                const double oz2 = k2 * k2 * (ozMode.real() * ozMode.real()
                                            + ozMode.imag() * ozMode.imag());

                lsum += ox2 + oy2 + oz2;
            },
            localEnstrophy);

        double globalEnstrophy = 0.0;
        ippl::Comm->allreduce(localEnstrophy, globalEnstrophy, 1, std::plus<double>());

        // omega_*_hat_m stores raw type-1 vorticity modes divided by
        // volume*k^2 after spectral scatter. Multiplying by k^2 recovers the
        // Fourier-series vorticity coefficient used by Parseval.
        return T(0.5) * volume * globalEnstrophy;
    }

    double computeSpectralEnstrophy() {
        return computeSpectralEnstrophy3D();
    }

    std::vector<VorticitySpectrumShell> computeSpectralVorticitySpectrum3D() {
        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        auto& layout       = omega_x_hat_m.getLayout();
        auto& mesh         = omega_x_hat_m.get_mesh();
        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = omega_x_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();

        const T Lx     = dx[0] * Nx;
        const T Ly     = dx[1] * Ny;
        const T Lz     = dx[2] * Nz;
        const T volume = Lx * Ly * Lz;
        const T twoPi  = T(2.0 * std::acos(-1.0));

        const int maxShell = static_cast<int>(std::floor(std::sqrt(
            static_cast<double>(Nx / 2) * static_cast<double>(Nx / 2)
            + static_cast<double>(Ny / 2) * static_cast<double>(Ny / 2)
            + static_cast<double>(Nz / 2) * static_cast<double>(Nz / 2))));
        const int shellCount = maxShell + 1;

        Kokkos::View<double*> localSpectrum("local_vorticity_spectrum_3d", shellCount);
        Kokkos::View<std::uint64_t*> localModeCounts(
            "local_vorticity_spectrum_3d_mode_counts", shellCount);
        Kokkos::deep_copy(localSpectrum, 0.0);
        Kokkos::deep_copy(localModeCounts, std::uint64_t(0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "compute_spectral_vorticity_spectrum_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ox.extent(0)) - nghost,
                         static_cast<int>(ox.extent(1)) - nghost,
                         static_cast<int>(ox.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
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
                const T k2 = kx * kx + ky * ky + kz * kz;

                const T radius2 = T(mx) * T(mx) + T(my) * T(my) + T(mz) * T(mz);
                const int shell = static_cast<int>(Kokkos::floor(Kokkos::sqrt(radius2)));

                const auto oxMode = ox(i, j, k);
                const auto oyMode = oy(i, j, k);
                const auto ozMode = oz(i, j, k);

                const double ox2 = k2 * k2 * (oxMode.real() * oxMode.real()
                                            + oxMode.imag() * oxMode.imag());
                const double oy2 = k2 * k2 * (oyMode.real() * oyMode.real()
                                            + oyMode.imag() * oyMode.imag());
                const double oz2 = k2 * k2 * (ozMode.real() * ozMode.real()
                                            + ozMode.imag() * ozMode.imag());

                Kokkos::atomic_add(&localSpectrum(shell), 0.5 * volume * (ox2 + oy2 + oz2));
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

        const int completeShellLimit = std::min({Nx, Ny, Nz}) / 2;
        std::vector<VorticitySpectrumShell> spectrum(shellCount);
        for (int shell = 0; shell < shellCount; ++shell) {
            spectrum[shell].enstrophy = globalValues[shell];
            spectrum[shell].modeCount = globalCounts[shell];
            spectrum[shell].complete  = shell < completeShellLimit;
        }
        return spectrum;
    }

    void logSpectralVorticitySpectrum3D(
        const std::string& filename = "vorticity_spectrum_3d.csv") {
        const auto spectrum = computeSpectralVorticitySpectrum3D();

        if (ippl::Comm->rank() == 0) {
            const bool firstWrite = !spectral_3d_vorticity_spectrum_initialized_m;
            std::ofstream out(diagnosticFileName(filename),
                              firstWrite ? std::ios::out : std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            if (firstWrite) {
                out << "method,dt,step,time,k,vorticity_spectrum,mode_count,complete\n";
            }

            for (std::size_t shell = 0; shell < spectrum.size(); ++shell) {
                out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                    << shell << "," << spectrum[shell].enstrophy << ","
                    << spectrum[shell].modeCount << ","
                    << (spectrum[shell].complete ? 1 : 0) << "\n";
            }

            Inform m("vorticity_spectrum_3d ");
            m << "step = " << it_m << ", shells = " << spectrum.size() << endl;
        }

        spectral_3d_vorticity_spectrum_initialized_m = true;
    }

    double computeSpectralDivergenceL23D() {
        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto& layout = ux_hat_m.getLayout();

        const auto& lDom   = layout.getLocalNDIndex();
        const int nghost   = ux_hat_m.getNghost();

        const int Nx = nr_m[0];
        const int Ny = nr_m[1];
        const int Nz = nr_m[2];

        const T Lx     = rmax_m[0] - rmin_m[0];
        const T Ly     = rmax_m[1] - rmin_m[1];
        const T Lz     = rmax_m[2] - rmin_m[2];
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

        // ux_hat_m, uy_hat_m, and uz_hat_m are Fourier-series coefficients, so
        // Parseval contributes one factor of volume.
        return std::sqrt(volume * globalDiv2);
    }

    double computeSpectralDivergenceL2() {
        return computeSpectralDivergenceL23D();
    }

    double computeSpectralVorticityDivergenceL23D() {
        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        auto& layout = omega_x_hat_m.getLayout();

        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = omega_x_hat_m.getNghost();

        const int Nx = nr_m[0];
        const int Ny = nr_m[1];
        const int Nz = nr_m[2];

        const T Lx = rmax_m[0] - rmin_m[0];
        const T Ly = rmax_m[1] - rmin_m[1];
        const T Lz = rmax_m[2] - rmin_m[2];
        const T volume = Lx * Ly * Lz;

        const T twoPi = T(2.0 * std::acos(-1.0));
        const Kokkos::complex<T> imag(0.0, 1.0);

        double localDiv2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "compute_spectral_vorticity_divergence_l2_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ox.extent(0)) - nghost,
                         static_cast<int>(ox.extent(1)) - nghost,
                         static_cast<int>(ox.extent(2)) - nghost}),
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
                const T k2 = kx * kx + ky * ky + kz * kz;

                const auto divHat =
                    imag * k2 * (kx * ox(i, j, k) + ky * oy(i, j, k) + kz * oz(i, j, k));
                lsum += divHat.real() * divHat.real() + divHat.imag() * divHat.imag();
            },
            localDiv2);

        double globalDiv2 = 0.0;
        ippl::Comm->allreduce(localDiv2, globalDiv2, 1, std::plus<double>());

        // omega_*_hat_m stores raw type-1 vorticity modes divided by
        // volume*k^2 after spectral scatter. Multiplying by k^2 recovers the
        // Fourier-series vorticity coefficient used by Parseval.
        return std::sqrt(volume * globalDiv2);
    }

    double computeTGVSpectralVelocityProjectionScale3D() {
        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto& layout = ux_hat_m.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = ux_hat_m.getNghost();

        const int Nx = nr_m[0];
        const int Ny = nr_m[1];
        const int Nz = nr_m[2];

        double localDot = 0.0;
        double localRef2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "tgv_spectral_velocity_projection_scale_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ux.extent(0)) - nghost,
                         static_cast<int>(ux.extent(1)) - nghost,
                         static_cast<int>(ux.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& dot, double& ref2) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();

                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const int mz = (gz <= Nz / 2) ? gz : gz - Nz;

                if ((mx == 1 || mx == -1) && (my == 1 || my == -1) &&
                    (mz == 1 || mz == -1)) {
                    const T sx = mx > 0 ? T(1.0) : T(-1.0);
                    const T sy = my > 0 ? T(1.0) : T(-1.0);

                    // For u = sin(x)cos(y)cos(z), v = -cos(x)sin(y)cos(z):
                    // u_hat = -i sign(mx)/8, v_hat = i sign(my)/8, w_hat = 0.
                    const Kokkos::complex<T> uxExact(T(0.0), -sx / T(8.0));
                    const Kokkos::complex<T> uyExact(T(0.0), sy / T(8.0));
                    const Kokkos::complex<T> uzExact(T(0.0), T(0.0));

                    const auto uxMode = ux(i, j, k);
                    const auto uyMode = uy(i, j, k);
                    const auto uzMode = uz(i, j, k);

                    dot += uxMode.real() * uxExact.real() + uxMode.imag() * uxExact.imag()
                         + uyMode.real() * uyExact.real() + uyMode.imag() * uyExact.imag()
                         + uzMode.real() * uzExact.real() + uzMode.imag() * uzExact.imag();
                    ref2 += uxExact.real() * uxExact.real() + uxExact.imag() * uxExact.imag()
                          + uyExact.real() * uyExact.real() + uyExact.imag() * uyExact.imag()
                          + uzExact.real() * uzExact.real() + uzExact.imag() * uzExact.imag();
                }
            },
            Kokkos::Sum<double>(localDot),
            Kokkos::Sum<double>(localRef2));

        double globalDot = 0.0;
        double globalRef2 = 0.0;
        ippl::Comm->allreduce(localDot, globalDot, 1, std::plus<double>());
        ippl::Comm->allreduce(localRef2, globalRef2, 1, std::plus<double>());

        return globalDot / std::max(globalRef2, 1e-30);
    }

    double computeTGVSpectralVorticityProjectionScale3D() {
        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        auto& layout = omega_x_hat_m.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = omega_x_hat_m.getNghost();

        const int Nx = nr_m[0];
        const int Ny = nr_m[1];
        const int Nz = nr_m[2];
        const T Lx = rmax_m[0] - rmin_m[0];
        const T Ly = rmax_m[1] - rmin_m[1];
        const T Lz = rmax_m[2] - rmin_m[2];
        const T twoPi = T(2.0 * std::acos(-1.0));

        double localDot = 0.0;
        double localRef2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "tgv_spectral_vorticity_projection_scale_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ox.extent(0)) - nghost,
                         static_cast<int>(ox.extent(1)) - nghost,
                         static_cast<int>(ox.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& dot, double& ref2) {
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
                const T k2 = kx * kx + ky * ky + kz * kz;

                if ((mx == 1 || mx == -1) && (my == 1 || my == -1) &&
                    (mz == 1 || mz == -1)) {
                    const T sx = mx > 0 ? T(1.0) : T(-1.0);
                    const T sy = my > 0 ? T(1.0) : T(-1.0);
                    const T sz = mz > 0 ? T(1.0) : T(-1.0);

                    // Current TaylorGreen3D convention:
                    // omega_x = -cos(x)sin(y)sin(z) gives omega_x_hat = sy*sz/8
                    // omega_y = -sin(x)cos(y)sin(z) gives omega_y_hat = sx*sz/8
                    // omega_z =  2sin(x)sin(y)cos(z) gives omega_z_hat = -sx*sy/4
                    const Kokkos::complex<T> oxExact(sy * sz / T(8.0), T(0.0));
                    const Kokkos::complex<T> oyExact(sx * sz / T(8.0), T(0.0));
                    const Kokkos::complex<T> ozExact(-sx * sy / T(4.0), T(0.0));

                    const auto oxMode = k2 * ox(i, j, k);
                    const auto oyMode = k2 * oy(i, j, k);
                    const auto ozMode = k2 * oz(i, j, k);

                    dot += oxMode.real() * oxExact.real() + oxMode.imag() * oxExact.imag()
                         + oyMode.real() * oyExact.real() + oyMode.imag() * oyExact.imag()
                         + ozMode.real() * ozExact.real() + ozMode.imag() * ozExact.imag();
                    ref2 += oxExact.real() * oxExact.real() + oxExact.imag() * oxExact.imag()
                          + oyExact.real() * oyExact.real() + oyExact.imag() * oyExact.imag()
                          + ozExact.real() * ozExact.real() + ozExact.imag() * ozExact.imag();
                }
            },
            Kokkos::Sum<double>(localDot),
            Kokkos::Sum<double>(localRef2));

        double globalDot = 0.0;
        double globalRef2 = 0.0;
        ippl::Comm->allreduce(localDot, globalDot, 1, std::plus<double>());
        ippl::Comm->allreduce(localRef2, globalRef2, 1, std::plus<double>());

        return globalDot / std::max(globalRef2, 1e-30);
    }

    double computeParticleDeformationOneNorm3D() {
        auto& pc = *pcontainer_m;
        auto duxdx = pc.duxdx.getView();
        auto duxdy = pc.duxdy.getView();
        auto duxdz = pc.duxdz.getView();
        auto duydx = pc.duydx.getView();
        auto duydy = pc.duydy.getView();
        auto duydz = pc.duydz.getView();
        auto duzdx = pc.duzdx.getView();
        auto duzdy = pc.duzdy.getView();
        auto duzdz = pc.duzdz.getView();
        const auto n = pc.getLocalNum();

        double localMax = 0.0;
        Kokkos::parallel_reduce(
            "compute_particle_deformation_one_norm_3d",
            n,
            KOKKOS_LAMBDA(const size_t p, double& lmax) {
                const T s11 = duxdx(p);
                const T s22 = duydy(p);
                const T s33 = duzdz(p);
                const T s12 = T(0.5) * (duxdy(p) + duydx(p));
                const T s13 = T(0.5) * (duxdz(p) + duzdx(p));
                const T s23 = T(0.5) * (duydz(p) + duzdy(p));

                const T a11 = s11 < T(0) ? -s11 : s11;
                const T a22 = s22 < T(0) ? -s22 : s22;
                const T a33 = s33 < T(0) ? -s33 : s33;
                const T a12 = s12 < T(0) ? -s12 : s12;
                const T a13 = s13 < T(0) ? -s13 : s13;
                const T a23 = s23 < T(0) ? -s23 : s23;

                const T column0 = a11 + a12 + a13;
                const T column1 = a12 + a22 + a23;
                const T column2 = a13 + a23 + a33;
                const T oneNorm = Kokkos::max(column0, Kokkos::max(column1, column2));
                if (static_cast<double>(oneNorm) > lmax) {
                    lmax = static_cast<double>(oneNorm);
                }
            },
            Kokkos::Max<double>(localMax));

        double globalMax = 0.0;
        ippl::Comm->allreduce(localMax, globalMax, 1, std::greater<double>());
        return globalMax;
    }

    void logSpectralDiagnostics3D(
        const std::string& filename = "spectral_diagnostics_3d.csv") {
        const double energy = computeSpectralEnergy3D();
        const double enstrophy = computeSpectralEnstrophy3D();
        const double divergenceL2 = computeSpectralDivergenceL23D();
        const double vorticityDivergenceL2 = computeSpectralVorticityDivergenceL23D();
        const double spectralVelocityProjectionScale =
            computeTGVSpectralVelocityProjectionScale3D();
        const double spectralVorticityProjectionScale =
            computeTGVSpectralVorticityProjectionScale3D();
        const double deformationOneNorm = computeParticleDeformationOneNorm3D();
        const double lcflDt = deformationOneNorm > 0.0
            ? lcfl_m / deformationOneNorm
            : std::numeric_limits<double>::infinity();

        const double Lx = rmax_m[0] - rmin_m[0];
        const double Ly = rmax_m[1] - rmin_m[1];
        const double Lz = rmax_m[2] - rmin_m[2];
        const double volume = Lx * Ly * Lz;
        const double pi = std::acos(-1.0);
        const double exactEnergy = pi * pi * pi;
        const double exactEnstrophy = 3.0 * pi * pi * pi;
        const double energyRelError =
            std::abs(energy - exactEnergy) / std::max(exactEnergy, 1e-30);
        const double enstrophyRelError =
            std::abs(enstrophy - exactEnstrophy) / std::max(exactEnstrophy, 1e-30);
        const double energyDissipationRate =
            2.0 * viscosity_m * enstrophy;

        if (!spectral_3d_dissipation_initialized_m) {
            spectral_3d_initial_energy_m = energy;
            spectral_3d_previous_time_m = time_m;
            spectral_3d_previous_dissipation_rate_m = energyDissipationRate;
            spectral_3d_cumulative_energy_dissipation_m = 0.0;
            spectral_3d_dissipation_initialized_m = true;
        } else {
            const double dtDiagnostic = time_m - spectral_3d_previous_time_m;
            if (dtDiagnostic > 0.0) {
                spectral_3d_cumulative_energy_dissipation_m +=
                    0.5 * (spectral_3d_previous_dissipation_rate_m
                           + energyDissipationRate) * dtDiagnostic;
            }
            spectral_3d_previous_time_m = time_m;
            spectral_3d_previous_dissipation_rate_m = energyDissipationRate;
        }

        const double energyLoss = spectral_3d_initial_energy_m - energy;
        const double energyDissipationBalanceError =
            energyLoss - spectral_3d_cumulative_energy_dissipation_m;
        const double energyDissipationBalanceRelError =
            std::abs(energyDissipationBalanceError)
            / std::max(std::abs(spectral_3d_cumulative_energy_dissipation_m), 1e-30);
        const double velocityL2 = std::sqrt(std::max(2.0 * energy, 1e-30));
        const double divergenceNormalized = divergenceL2 / velocityL2;
        const double vorticityL2 = std::sqrt(std::max(2.0 * enstrophy, 1e-30));
        const double vorticityDivergenceNormalized =
            vorticityDivergenceL2 / vorticityL2;

        if (ippl::Comm->rank() == 0) {
            const bool firstWrite = !spectral_3d_diagnostics_initialized_m;
            std::ofstream out(diagnosticFileName(filename),
                              firstWrite ? std::ios::out : std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            if (firstWrite) {
                out << "method,dt,step,time,spectral_energy,spectral_energy_rel_error,"
                    << "spectral_enstrophy,spectral_enstrophy_rel_error,"
                    << "spectral_energy_dissipation_rate,"
                    << "spectral_cumulative_energy_dissipation,"
                    << "spectral_energy_loss,"
                    << "spectral_energy_dissipation_balance_error,"
                    << "spectral_energy_dissipation_balance_rel_error,"
                    << "spectral_divergence_l2,spectral_divergence_normalized,"
                    << "spectral_vorticity_divergence_l2,"
                    << "spectral_vorticity_divergence_normalized,"
                    << "spectral_velocity_projection_scale,"
                    << "spectral_vorticity_projection_scale,"
                    << "deformation_one_norm,lcfl,lcfl_dt,"
                    << "Lx,Ly,Lz,volume\n";
            }

            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << energy << "," << energyRelError << ","
                << enstrophy << "," << enstrophyRelError << ","
                << energyDissipationRate << ","
                << spectral_3d_cumulative_energy_dissipation_m << ","
                << energyLoss << ","
                << energyDissipationBalanceError << ","
                << energyDissipationBalanceRelError << ","
                << divergenceL2 << "," << divergenceNormalized << ","
                << vorticityDivergenceL2 << "," << vorticityDivergenceNormalized << ","
                << spectralVelocityProjectionScale << ","
                << spectralVorticityProjectionScale << ","
                << deformationOneNorm << ","
                << lcfl_m << ","
                << lcflDt << ","
                << Lx << "," << Ly << "," << Lz << "," << volume << "\n";

            Inform m("spectral_diagnostics_3d ");
            m << "energy = " << energy
              << ", energyRelError = " << energyRelError
              << ", enstrophy = " << enstrophy
              << ", enstrophyRelError = " << enstrophyRelError
              << ", energyDissipationRate = " << energyDissipationRate
              << ", cumulativeEnergyDissipation = "
              << spectral_3d_cumulative_energy_dissipation_m
              << ", energyDissipationBalanceError = "
              << energyDissipationBalanceError
              << ", divergenceL2 = " << divergenceL2
              << ", divergenceNormalized = " << divergenceNormalized
              << ", vorticityDivergenceL2 = " << vorticityDivergenceL2
              << ", vorticityDivergenceNormalized = "
              << vorticityDivergenceNormalized
              << ", spectralVelocityProjectionScale = "
              << spectralVelocityProjectionScale
              << ", spectralVorticityProjectionScale = "
              << spectralVorticityProjectionScale
              << ", deformationOneNorm = " << deformationOneNorm
              << ", lcfl = " << lcfl_m
              << ", lcflDt = " << lcflDt
              << ", L = (" << Lx << ", " << Ly << ", " << Lz
              << "), volume = " << volume << endl;
        }

        spectral_3d_diagnostics_initialized_m = true;
    }
#endif
