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

        // computeSpectralVelocityModes3D divides the raw type-1 NUFFT vorticity
        // modes by volume, so ux_hat_m, uy_hat_m, and uz_hat_m are Fourier-series
        // velocity coefficients. Parseval contributes one factor of volume.
        return T(0.5) * volume * globalEnergy;
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

                    const auto oxMode = ox(i, j, k);
                    const auto oyMode = oy(i, j, k);
                    const auto ozMode = oz(i, j, k);

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

    void logTGVSingleMode3D(const std::string& filename = "tgv_single_mode_3d.csv") {
        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();
        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto& layout = omega_x_hat_m.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = omega_x_hat_m.getNghost();
        const int Nx = nr_m[0];
        const int Ny = nr_m[1];
        const int Nz = nr_m[2];

        double localValues[12] = {};
        Kokkos::parallel_reduce(
            "tgv_single_mode_3d",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {nghost, nghost, nghost},
                {static_cast<int>(ox.extent(0)) - nghost,
                 static_cast<int>(ox.extent(1)) - nghost,
                 static_cast<int>(ox.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& oxRe, double& oxIm,
                          double& oyRe, double& oyIm,
                          double& ozRe, double& ozIm,
                          double& uxRe, double& uxIm,
                          double& uyRe, double& uyIm,
                          double& uzRe, double& uzIm) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();
                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const int mz = (gz <= Nz / 2) ? gz : gz - Nz;

                if (mx == 1 && my == 1 && mz == 1) {
                    const auto oxMode = ox(i, j, k);
                    const auto oyMode = oy(i, j, k);
                    const auto ozMode = oz(i, j, k);
                    const auto uxMode = ux(i, j, k);
                    const auto uyMode = uy(i, j, k);
                    const auto uzMode = uz(i, j, k);
                    oxRe += oxMode.real();
                    oxIm += oxMode.imag();
                    oyRe += oyMode.real();
                    oyIm += oyMode.imag();
                    ozRe += ozMode.real();
                    ozIm += ozMode.imag();
                    uxRe += uxMode.real();
                    uxIm += uxMode.imag();
                    uyRe += uyMode.real();
                    uyIm += uyMode.imag();
                    uzRe += uzMode.real();
                    uzIm += uzMode.imag();
                }
            },
            Kokkos::Sum<double>(localValues[0]),
            Kokkos::Sum<double>(localValues[1]),
            Kokkos::Sum<double>(localValues[2]),
            Kokkos::Sum<double>(localValues[3]),
            Kokkos::Sum<double>(localValues[4]),
            Kokkos::Sum<double>(localValues[5]),
            Kokkos::Sum<double>(localValues[6]),
            Kokkos::Sum<double>(localValues[7]),
            Kokkos::Sum<double>(localValues[8]),
            Kokkos::Sum<double>(localValues[9]),
            Kokkos::Sum<double>(localValues[10]),
            Kokkos::Sum<double>(localValues[11]));

        double values[12] = {};
        for (int n = 0; n < 12; ++n) {
            ippl::Comm->allreduce(localValues[n], values[n], 1, std::plus<double>());
        }

        const double volume = (rmax_m[0] - rmin_m[0])
                            * (rmax_m[1] - rmin_m[1])
                            * (rmax_m[2] - rmin_m[2]);
        const double expected[12] = {
            volume / 8.0, 0.0,
            volume / 8.0, 0.0,
            -volume / 4.0, 0.0,
            0.0, -1.0 / 8.0,
            0.0, 1.0 / 8.0,
            0.0, 0.0
        };

        if (ippl::Comm->rank() == 0) {
            const bool firstWrite = !tgv_single_mode_3d_initialized_m;
            std::ofstream out(diagnosticFileName(filename),
                              firstWrite ? std::ios::out : std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            if (firstWrite) {
                out << "method,dt,step,time,mode_x,mode_y,mode_z,"
                    << "omega_x_re,omega_x_im,omega_x_expected_re,omega_x_expected_im,"
                    << "omega_y_re,omega_y_im,omega_y_expected_re,omega_y_expected_im,"
                    << "omega_z_re,omega_z_im,omega_z_expected_re,omega_z_expected_im,"
                    << "u_x_re,u_x_im,u_x_expected_re,u_x_expected_im,"
                    << "u_y_re,u_y_im,u_y_expected_re,u_y_expected_im,"
                    << "u_z_re,u_z_im,u_z_expected_re,u_z_expected_im\n";
            }

            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ",1,1,1";
            for (int n = 0; n < 12; n += 2) {
                out << "," << values[n] << "," << values[n + 1]
                    << "," << expected[n] << "," << expected[n + 1];
            }
            out << "\n";

            Inform m("tgv_single_mode_3d ");
            m << "omega_x = (" << values[0] << ", " << values[1]
              << "), omega_y = (" << values[2] << ", " << values[3]
              << "), omega_z = (" << values[4] << ", " << values[5]
              << "), u_x = (" << values[6] << ", " << values[7]
              << "), u_y = (" << values[8] << ", " << values[9]
              << "), u_z = (" << values[10] << ", " << values[11] << ")" << endl;
        }

        tgv_single_mode_3d_initialized_m = true;
    }

    void logSpectralDiagnostics3D(
        const std::string& filename = "spectral_diagnostics_3d.csv") {
        const double energy = computeSpectralEnergy3D();
        const double enstrophy = computeSpectralEnstrophy3D();
        const double divergenceL2 = computeSpectralDivergenceL23D();
        const double spectralVelocityProjectionScale =
            computeTGVSpectralVelocityProjectionScale3D();
        const double spectralVorticityProjectionScale =
            computeTGVSpectralVorticityProjectionScale3D();

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
                    << "spectral_divergence_l2,spectral_divergence_normalized,"
                    << "spectral_velocity_projection_scale,"
                    << "spectral_vorticity_projection_scale\n";
            }

            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << energy << "," << energyRelError << ","
                << enstrophy << "," << enstrophyRelError << ","
                << divergenceL2 << "," << divergenceNormalized << ","
                << spectralVelocityProjectionScale << ","
                << spectralVorticityProjectionScale << "\n";

            Inform m("spectral_diagnostics_3d ");
            m << "energy = " << energy
              << ", energyRelError = " << energyRelError
              << ", enstrophy = " << enstrophy
              << ", enstrophyRelError = " << enstrophyRelError
              << ", divergenceL2 = " << divergenceL2
              << ", divergenceNormalized = " << divergenceNormalized
              << ", spectralVelocityProjectionScale = "
              << spectralVelocityProjectionScale
              << ", spectralVorticityProjectionScale = "
              << spectralVorticityProjectionScale << endl;
        }

        spectral_3d_diagnostics_initialized_m = true;
    }
#endif
