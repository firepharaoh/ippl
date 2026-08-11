#ifndef IPPL_ALVINE_SPECTRAL_ALVINESPECTRALOPS3D_HPP
#define IPPL_ALVINE_SPECTRAL_ALVINESPECTRALOPS3D_HPP

    void spectralScatter3D(const bool applyShapeFilter = true) {
        if (!nufftType1_mp) {
            throw std::runtime_error("AlvineManager3D::spectralScatter3D called before initNUFFT3D");
        }

        auto& pc = *this->pcontainer_m;
        auto omega = pc.omega.getView();
        auto ox = pc.omega_x.getView();
        auto oy = pc.omega_y.getView();
        auto oz = pc.omega_z.getView();
        auto nlocal = pc.getLocalNum();

        Kokkos::parallel_for(
            "split_omega_components_3d",
            nlocal,
            KOKKOS_LAMBDA(const size_t p) {
                ox(p) = omega(p)[0];
                oy(p) = omega(p)[1];
                oz(p) = omega(p)[2];
            });
        Kokkos::fence();

        omega_x_hat_m = Kokkos::complex<T>(0.0, 0.0);
        omega_y_hat_m = Kokkos::complex<T>(0.0, 0.0);
        omega_z_hat_m = Kokkos::complex<T>(0.0, 0.0);

        nufftType1_mp->transform(pc.R, pc.omega_x, omega_x_hat_m);
        nufftType1_mp->transform(pc.R, pc.omega_y, omega_y_hat_m);
        nufftType1_mp->transform(pc.R, pc.omega_z, omega_z_hat_m);

        if (applyShapeFilter && useShapeFunctionFilter()) {
            applyShapeFunctionToSpectralVorticityModes3D();
        }
        Kokkos::fence();
    }

    void spectralScatter() {
        spectralScatter3D();
    }

    void applyShapeFunctionToSpectralVorticityModes3D() {
        auto oxModes = omega_x_hat_m.getView();
        auto oyModes = omega_y_hat_m.getView();
        auto ozModes = omega_z_hat_m.getView();
        auto shapeView = Sk_m.getView();
        const int nghost = omega_x_hat_m.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "shape_spectral_vorticity_modes_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(oxModes.extent(0)) - nghost,
                         static_cast<int>(oxModes.extent(1)) - nghost,
                         static_cast<int>(oxModes.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                oxModes(i, j, k) *= shapeView(i, j, k);
                oyModes(i, j, k) *= shapeView(i, j, k);
                ozModes(i, j, k) *= shapeView(i, j, k);
            });
        Kokkos::fence();
    }

    void Hou_Li_filter(ComplexField_t& modes, double alpha = 36.0, int exponent = 36) {
        auto view = modes.getView();

        auto& layout = modes.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost = modes.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();
        const T kxMax = T(Nx) / T(2.0);
        const T kyMax = T(Ny) / T(2.0);
        const T kzMax = T(Nz) / T(2.0);
        const T invSqrtDim = T(1.0) / Kokkos::sqrt(T(3.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "hou_li_filter_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(view.extent(0)) - nghost,
                         static_cast<int>(view.extent(1)) - nghost,
                         static_cast<int>(view.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();

                const T kx = (gx <= Nx / 2) ? T(gx) : T(gx - Nx);
                const T ky = (gy <= Ny / 2) ? T(gy) : T(gy - Ny);
                const T kz = (gz <= Nz / 2) ? T(gz) : T(gz - Nz);

                const T etaX = Kokkos::abs(kx) / kxMax;
                const T etaY = Kokkos::abs(ky) / kyMax;
                const T etaZ = Kokkos::abs(kz) / kzMax;
                const T eta =
                    Kokkos::sqrt(etaX * etaX + etaY * etaY + etaZ * etaZ) * invSqrtDim;
                const T filterFactor =
                    Kokkos::exp(-T(alpha) * Kokkos::pow(eta, exponent));

                view(i, j, k) *= filterFactor;
            });
        Kokkos::fence();
    }

    void twoThirdsFilter3D(ComplexField_t& modes) {
        auto view = modes.getView();

        auto& layout = modes.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost = modes.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();

        const T cutoffX = T(Nx) / T(3.0);
        const T cutoffY = T(Ny) / T(3.0);
        const T cutoffZ = T(Nz) / T(3.0);

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "two_thirds_filter_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(view.extent(0)) - nghost,
                         static_cast<int>(view.extent(1)) - nghost,
                         static_cast<int>(view.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();

                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const int mz = (gz <= Nz / 2) ? gz : gz - Nz;

                if (Kokkos::abs(T(mx)) > cutoffX || Kokkos::abs(T(my)) > cutoffY
                    || Kokkos::abs(T(mz)) > cutoffZ) {
                    view(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                }
            });
        Kokkos::fence();
    }

    void applyConfiguredSpectralFilter3D(ComplexField_t& modes) {
        if (useHouLiFilter()) {
            Hou_Li_filter(modes);
        } else if (useTwoThirdsFilter()) {
            twoThirdsFilter3D(modes);
        }
    }

    void projectSpectralVorticityModes3D() {
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

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "project_spectral_vorticity_modes_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ox.extent(0)) - nghost,
                         static_cast<int>(ox.extent(1)) - nghost,
                         static_cast<int>(ox.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // Convert the local array index to the corresponding global
                // Fourier-grid index for this MPI rank's local slab.
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();

                // Map FFT ordering to signed integer wave numbers.
                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const int mz = (gz <= Nz / 2) ? gz : gz - Nz;

                // Use the same effective wave-vector convention as the
                // velocity reconstruction and diagnostics: the Nyquist mode is
                // treated as zero in its own direction.
                const bool notMidX = (gx != Nx / 2);
                const bool notMidY = (gy != Ny / 2);
                const bool notMidZ = (gz != Nz / 2);

                const T kx = notMidX * twoPi * mx / Lx;
                const T ky = notMidY * twoPi * my / Ly;
                const T kz = notMidZ * twoPi * mz / Lz;

                // k2 is the denominator in the Helmholtz projection onto the
                // plane perpendicular to k. The zero mode has no direction, so
                // keep the previous solver convention and remove mean vorticity.
                const T k2 = kx * kx + ky * ky + kz * kz;
                if (k2 == T(0)) {
                    ox(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    oy(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    oz(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    return;
                }

                // Project omega_hat by removing the component parallel to k:
                // omega_hat <- omega_hat - k * (k dot omega_hat) / |k|^2.
                const auto kDotOmega =
                    kx * ox(i, j, k) + ky * oy(i, j, k) + kz * oz(i, j, k);
                const auto correction = kDotOmega / k2;

                // Each component update is independent for this mode, so the
                // kernel writes only the current local spectral cell.
                ox(i, j, k) -= kx * correction;
                oy(i, j, k) -= ky * correction;
                oz(i, j, k) -= kz * correction;
            });
        Kokkos::fence();
    }

    void computeSpectralVelocityModes3D() {
        // Ensure the vorticity modes are solenoidal before using the
        // Biot-Savart relation. Filtering, remeshing, and particle scatter can
        // introduce a small k-parallel component that would otherwise appear as
        // nonzero div(omega) in diagnostics and downstream reconstructions.
        projectSpectralVorticityModes3D();

        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto& layout = omega_x_hat_m.getLayout();

        const auto& lDom   = layout.getLocalNDIndex();
        const int nghost   = omega_x_hat_m.getNghost();

        const int Nx = nr_m[0];
        const int Ny = nr_m[1];
        const int Nz = nr_m[2];

        const T Lx = rmax_m[0] - rmin_m[0];
        const T Ly = rmax_m[1] - rmin_m[1];
        const T Lz = rmax_m[2] - rmin_m[2];
        const T volume = Lx * Ly * Lz;

        const T twoPi = T(2.0 * std::acos(-1.0));
        const Kokkos::complex<T> imag(0.0, 1.0);

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "compute_spectral_velocity_modes_3d",
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
                if (k2 == T(0)) {
                    ox(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    oy(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    oz(i, j, k) = Kokkos::complex<T>(0.0, 0.0);

                    ux(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    uy(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                    uz(i, j, k) = Kokkos::complex<T>(0.0, 0.0);
                } else {
                    const auto invVolumeK2 = T(1.0) / (volume * k2);

                    ux(i, j, k) = imag * (ky * oz(i, j, k) - kz * oy(i, j, k)) * invVolumeK2;
                    uy(i, j, k) = imag * (kz * ox(i, j, k) - kx * oz(i, j, k)) * invVolumeK2;
                    uz(i, j, k) = imag * (kx * oy(i, j, k) - ky * ox(i, j, k)) * invVolumeK2;
                }

            }
        );
        Kokkos::fence();
    }

    void computeSpectralVelocityModes() {
        computeSpectralVelocityModes3D();
    }

    void initializeShapeFunctionVIF3D() {
        using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        auto Skview = Sk_m.getView();
        auto N = nr_m;
        const int nghost = Sk_m.getNghost();
        const auto& mesh = Sk_m.get_mesh();
        const Vector_t<T, Dim> dx = mesh.getMeshSpacing();
        const Vector_t<T, Dim> Len = rmax_m - rmin_m;
        const T pi = T(3.141592653589793238462643383279502884);
        const int order = shapedegree_m + 1;
        const auto& layout = Sk_m.getLayout();
        const auto& lDom = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "B-spline shape function initialization 3d",
            mdrange_type({nghost, nghost, nghost},
                         {static_cast<int>(Skview.extent(0)) - nghost,
                          static_cast<int>(Skview.extent(1)) - nghost,
                          static_cast<int>(Skview.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                Vector<int, Dim> iVec = {i, j, k};
                for (unsigned d = 0; d < Dim; d++) {
                    iVec[d] = iVec[d] - nghost + lDom[d].first();
                }

                Vector<double, Dim> kVec;
                double Sk = 1.0;
                for (unsigned d = 0; d < Dim; d++) {
                    bool shift = (iVec[d] > (N[d] / 2));
                    kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                    double khbytwo = kVec[d] * dx[d] / 2.0;
                    bool isNotZero = (khbytwo != 0.0);
                    double factor = (1.0 / (khbytwo + ((!isNotZero) * 1.0)));
                    double arg = isNotZero * (Kokkos::sin(khbytwo) * factor) +
                                 (!isNotZero) * 1.0;
                    Sk *= Kokkos::pow(arg, order);
                }
                Skview(i, j, k) = Sk;
            });
        Kokkos::fence();
    }

    void spectralGather3D() {
        if (!nufftType2_mp) {
            throw std::runtime_error("AlvineManager3D::spectralGather3D called before initNUFFT3D");
        }
        auto& pc = *this->pcontainer_m;

        pc.ux = 0.0;
        pc.uy = 0.0;
        pc.uz = 0.0;

        auto uxModes = ux_hat_m.deepCopy();
        auto uyModes = uy_hat_m.deepCopy();
        auto uzModes = uz_hat_m.deepCopy();

        if (useShapeFunctionFilter()) {
            auto uxModeView = uxModes.getView();
            auto uyModeView = uyModes.getView();
            auto uzModeView = uzModes.getView();
            auto shapeView = Sk_m.getView();
            const int nghost = uxModes.getNghost();

            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "shape_spectral_velocity_modes_3d",
                policy_type({nghost, nghost, nghost},
                            {static_cast<int>(uxModeView.extent(0)) - nghost,
                             static_cast<int>(uxModeView.extent(1)) - nghost,
                             static_cast<int>(uxModeView.extent(2)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    uxModeView(i, j, k) *= shapeView(i, j, k);
                    uyModeView(i, j, k) *= shapeView(i, j, k);
                    uzModeView(i, j, k) *= shapeView(i, j, k);
                });
            Kokkos::fence();

        }

        nufftType2_mp->transform(pc.R, pc.ux, uxModes);
        nufftType2_mp->transform(pc.R, pc.uy, uyModes);
        nufftType2_mp->transform(pc.R, pc.uz, uzModes);

        auto P = pc.P.getView();
        auto u = pc.u.getView();
        auto ux = pc.ux.getView();
        auto uy = pc.uy.getView();
        auto uz = pc.uz.getView();
        const auto n = pc.getLocalNum();

        Kokkos::parallel_for(
            "pack_spectral_velocity_3d", n,
            KOKKOS_LAMBDA(const size_t i) {
                P(i)[0] = ux(i);
                P(i)[1] = uy(i);
                P(i)[2] = uz(i);
                u(i)[0] = ux(i);
                u(i)[1] = uy(i);
                u(i)[2] = uz(i);
            });
        Kokkos::fence();

    }

    void spectralGather() {
        spectralGather3D();
    }

    void spectralSolveParticles() {
        spectralScatter3D();
        applyConfiguredSpectralFilter3D(omega_x_hat_m);
        applyConfiguredSpectralFilter3D(omega_y_hat_m);
        applyConfiguredSpectralFilter3D(omega_z_hat_m);
        computeSpectralVelocityModes3D();
        applyConfiguredSpectralFilter3D(ux_hat_m);
        applyConfiguredSpectralFilter3D(uy_hat_m);
        applyConfiguredSpectralFilter3D(uz_hat_m);
        spectralGather3D();
    }

    void computeSpectralViscosityModes3D() {
        auto viscX = viscosity_x_hat_m.getView();
        auto viscY = viscosity_y_hat_m.getView();
        auto viscZ = viscosity_z_hat_m.getView();

        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        const auto& layout = viscosity_x_hat_m.getLayout();
        const auto& mesh   = viscosity_x_hat_m.get_mesh();

        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = viscosity_x_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();

        const T Lx = dx[0] * Nx;
        const T Ly = dx[1] * Ny;
        const T Lz = dx[2] * Nz;
        const T volume = Lx * Ly * Lz;

        const T twoPi = T(2.0 * std::acos(-1.0));
        const T nu = T(viscosity_m);

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "compute_spectral_viscosity_modes_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(viscX.extent(0)) - nghost,
                         static_cast<int>(viscX.extent(1)) - nghost,
                         static_cast<int>(viscX.extent(2)) - nghost}),
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

                viscX(i, j, k) = -nu * k2 * ox(i, j, k) / volume;
                viscY(i, j, k) = -nu * k2 * oy(i, j, k) / volume;
                viscZ(i, j, k) = -nu * k2 * oz(i, j, k) / volume;
            }
        );
        Kokkos::fence();
    }

    void spectralGatherViscosity3D() {
        if (!nufftType2_mp) {
            throw std::runtime_error("AlvineManager3D::spectralGatherViscosity3D called before initNUFFT3D");
        }

        auto& pc = *this->pcontainer_m;

        pc.viscosity_x = 0.0;
        pc.viscosity_y = 0.0;
        pc.viscosity_z = 0.0;

        auto viscXModes = viscosity_x_hat_m.deepCopy();
        auto viscYModes = viscosity_y_hat_m.deepCopy();
        auto viscZModes = viscosity_z_hat_m.deepCopy();

        nufftType2_mp->transform(pc.R, pc.viscosity_x, viscXModes);
        nufftType2_mp->transform(pc.R, pc.viscosity_y, viscYModes);
        nufftType2_mp->transform(pc.R, pc.viscosity_z, viscZModes);

        auto visc = pc.viscosity.getView();
        auto viscX = pc.viscosity_x.getView();
        auto viscY = pc.viscosity_y.getView();
        auto viscZ = pc.viscosity_z.getView();
        const auto n = pc.getLocalNum();

        Kokkos::parallel_for(
            "pack_particle_viscosity_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                visc(p)[0] = viscX(p);
                visc(p)[1] = viscY(p);
                visc(p)[2] = viscZ(p);
            });

        Kokkos::fence();
    }

    void applyParticleViscosity3D() {
        auto& pc = *this->pcontainer_m;
        auto omega = pc.omega.getView();
        auto visc = pc.viscosity.getView();
        auto omegaX = pc.omega_x.getView();
        auto omegaY = pc.omega_y.getView();
        auto omegaZ = pc.omega_z.getView();
        const T dt = T(this->dt_m);
        const auto n = pc.getLocalNum();

        Kokkos::parallel_for(
            "apply_particle_viscosity_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                omega(p)[0] += dt * visc(p)[0];
                omega(p)[1] += dt * visc(p)[1];
                omega(p)[2] += dt * visc(p)[2];
                omegaX(p) = omega(p)[0];
                omegaY(p) = omega(p)[1];
                omegaZ(p) = omega(p)[2];
            });
        Kokkos::fence();
    }

    void applyCellCenteredIfftPhase3D(ComplexField_t& modes) {
        auto view = modes.getView();

        auto& layout = modes.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost = modes.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();
        const T pi = std::acos(T(-1.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "apply_cell_centered_ifft_phase_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(view.extent(0)) - nghost,
                         static_cast<int>(view.extent(1)) - nghost,
                         static_cast<int>(view.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int gz = k - nghost + lDom[2].first();

                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const int mz = (gz <= Nz / 2) ? gz : gz - Nz;

                const T phase = pi * (T(mx) / T(Nx) + T(my) / T(Ny) + T(mz) / T(Nz));
                const Kokkos::complex<T> factor(Kokkos::cos(phase), Kokkos::sin(phase));

                view(i, j, k) *= factor;
            });
        Kokkos::fence();
    }

    void reconstructSpectralVorticity3D(VField_t<T, Dim>& omegaField) {
        if (!spectralFft_mp) {
            throw std::runtime_error(
                "AlvineManager3D::reconstructSpectralVorticity3D called before initNUFFT3D");
        }

        auto oxModes = omega_x_hat_m.deepCopy();
        auto oyModes = omega_y_hat_m.deepCopy();
        auto ozModes = omega_z_hat_m.deepCopy();

        const auto& domain = omega_x_hat_m.getLayout().getDomain();
        const auto& dx = omega_x_hat_m.get_mesh().getMeshSpacing();
        const T volume = (dx[0] * domain[0].length()) *
                         (dx[1] * domain[1].length()) *
                         (dx[2] * domain[2].length());

        oxModes = oxModes / volume;
        oyModes = oyModes / volume;
        ozModes = ozModes / volume;

        applyCellCenteredIfftPhase3D(oxModes);
        applyCellCenteredIfftPhase3D(oyModes);
        applyCellCenteredIfftPhase3D(ozModes);

        spectralFft_mp->transform(ippl::BACKWARD, oxModes);
        spectralFft_mp->transform(ippl::BACKWARD, oyModes);
        spectralFft_mp->transform(ippl::BACKWARD, ozModes);

        auto omegaOut = omegaField.getView();
        auto oxGrid = oxModes.getView();
        auto oyGrid = oyModes.getView();
        auto ozGrid = ozModes.getView();
        const int nghost = omegaField.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "reconstruct_spectral_vorticity_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(omegaOut.extent(0)) - nghost,
                         static_cast<int>(omegaOut.extent(1)) - nghost,
                         static_cast<int>(omegaOut.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                omegaOut(i, j, k)[0] = oxGrid(i, j, k).real();
                omegaOut(i, j, k)[1] = oyGrid(i, j, k).real();
                omegaOut(i, j, k)[2] = ozGrid(i, j, k).real();
            });
        Kokkos::fence();
    }

    void reconstructSpectralVorticity(VField_t<T, Dim>& omegaField) {
        reconstructSpectralVorticity3D(omegaField);
    }

    void reconstructSpectralVelocity3D(VField_t<T, Dim>& uField) {
        if (!spectralFft_mp) {
            throw std::runtime_error(
                "AlvineManager3D::reconstructSpectralVelocity3D called before initNUFFT3D");
        }

        auto uxModes = ux_hat_m.deepCopy();
        auto uyModes = uy_hat_m.deepCopy();
        auto uzModes = uz_hat_m.deepCopy();

        applyCellCenteredIfftPhase3D(uxModes);
        applyCellCenteredIfftPhase3D(uyModes);
        applyCellCenteredIfftPhase3D(uzModes);

        spectralFft_mp->transform(ippl::BACKWARD, uxModes);
        spectralFft_mp->transform(ippl::BACKWARD, uyModes);
        spectralFft_mp->transform(ippl::BACKWARD, uzModes);

        auto uOut = uField.getView();
        auto uxGrid = uxModes.getView();
        auto uyGrid = uyModes.getView();
        auto uzGrid = uzModes.getView();
        const int nghost = uField.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "reconstruct_spectral_velocity_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(uOut.extent(0)) - nghost,
                         static_cast<int>(uOut.extent(1)) - nghost,
                         static_cast<int>(uOut.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                uOut(i, j, k)[0] = uxGrid(i, j, k).real();
                uOut(i, j, k)[1] = uyGrid(i, j, k).real();
                uOut(i, j, k)[2] = uzGrid(i, j, k).real();
            });
        Kokkos::fence();

        const auto uAverage = uField.getVolumeAverage();
        Kokkos::parallel_for(
            "remove_reconstructed_velocity_mean_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(uOut.extent(0)) - nghost,
                         static_cast<int>(uOut.extent(1)) - nghost,
                         static_cast<int>(uOut.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                uOut(i, j, k)[0] -= uAverage[0];
                uOut(i, j, k)[1] -= uAverage[1];
                uOut(i, j, k)[2] -= uAverage[2];
            });
        Kokkos::fence();
    }

    void reconstructSpectralVelocity(VField_t<T, Dim>& uField) {
        reconstructSpectralVelocity3D(uField);
    }

    void computeSpectralVelocityGradientModes3D() {
        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto duxdx = duxdx_hat_m.getView();
        auto duxdy = duxdy_hat_m.getView();
        auto duxdz = duxdz_hat_m.getView();
        auto duydx = duydx_hat_m.getView();
        auto duydy = duydy_hat_m.getView();
        auto duydz = duydz_hat_m.getView();
        auto duzdx = duzdx_hat_m.getView();
        auto duzdy = duzdy_hat_m.getView();
        auto duzdz = duzdz_hat_m.getView();

        auto& layout = ux_hat_m.getLayout();
        const auto& lDom = layout.getLocalNDIndex();
        const int nghost = ux_hat_m.getNghost();

        const int Nx = nr_m[0];
        const int Ny = nr_m[1];
        const int Nz = nr_m[2];

        const T Lx = rmax_m[0] - rmin_m[0];
        const T Ly = rmax_m[1] - rmin_m[1];
        const T Lz = rmax_m[2] - rmin_m[2];

        const T twoPi = T(2.0 * std::acos(-1.0));
        const Kokkos::complex<T> imag(0.0, 1.0);

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_for(
            "compute_spectral_velocity_gradient_modes_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(ux.extent(0)) - nghost,
                         static_cast<int>(ux.extent(1)) - nghost,
                         static_cast<int>(ux.extent(2)) - nghost}),
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

                duxdx(i, j, k) = imag * kx * ux(i, j, k);
                duxdy(i, j, k) = imag * ky * ux(i, j, k);
                duxdz(i, j, k) = imag * kz * ux(i, j, k);

                duydx(i, j, k) = imag * kx * uy(i, j, k);
                duydy(i, j, k) = imag * ky * uy(i, j, k);
                duydz(i, j, k) = imag * kz * uy(i, j, k);

                duzdx(i, j, k) = imag * kx * uz(i, j, k);
                duzdy(i, j, k) = imag * ky * uz(i, j, k);
                duzdz(i, j, k) = imag * kz * uz(i, j, k);
            });
        Kokkos::fence();
    }


    void spectralGatherGradientModes3D() {
        if (!nufftType2_mp) {
            throw std::runtime_error("AlvineManager3D::spectralGatherGradientModes3D called before initNUFFT3D");
        }

        auto& pc = *this->pcontainer_m;

        pc.duxdx = 0.0;
        pc.duxdy = 0.0;
        pc.duxdz = 0.0;
        pc.duydx = 0.0;
        pc.duydy = 0.0;
        pc.duydz = 0.0;
        pc.duzdx = 0.0;
        pc.duzdy = 0.0;
        pc.duzdz = 0.0;

        auto duxdxModes = duxdx_hat_m.deepCopy();
        auto duxdyModes = duxdy_hat_m.deepCopy();
        auto duxdzModes = duxdz_hat_m.deepCopy();
        auto duydxModes = duydx_hat_m.deepCopy();
        auto duydyModes = duydy_hat_m.deepCopy();
        auto duydzModes = duydz_hat_m.deepCopy();
        auto duzdxModes = duzdx_hat_m.deepCopy();
        auto duzdyModes = duzdy_hat_m.deepCopy();
        auto duzdzModes = duzdz_hat_m.deepCopy();

        if (useShapeFunctionFilter()) {
            auto shapeView = Sk_m.getView();
            const int nghost = duxdxModes.getNghost();
            auto duxdxView = duxdxModes.getView();
            auto duxdyView = duxdyModes.getView();
            auto duxdzView = duxdzModes.getView();
            auto duydxView = duydxModes.getView();
            auto duydyView = duydyModes.getView();
            auto duydzView = duydzModes.getView();
            auto duzdxView = duzdxModes.getView();
            auto duzdyView = duzdyModes.getView();
            auto duzdzView = duzdzModes.getView();

            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
            Kokkos::parallel_for(
                "shape_spectral_velocity_gradient_modes_3d",
                policy_type({nghost, nghost, nghost},
                            {static_cast<int>(duxdxView.extent(0)) - nghost,
                             static_cast<int>(duxdxView.extent(1)) - nghost,
                             static_cast<int>(duxdxView.extent(2)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const auto shape = shapeView(i, j, k);
                    duxdxView(i, j, k) *= shape;
                    duxdyView(i, j, k) *= shape;
                    duxdzView(i, j, k) *= shape;
                    duydxView(i, j, k) *= shape;
                    duydyView(i, j, k) *= shape;
                    duydzView(i, j, k) *= shape;
                    duzdxView(i, j, k) *= shape;
                    duzdyView(i, j, k) *= shape;
                    duzdzView(i, j, k) *= shape;
                });
            Kokkos::fence();
        }

        nufftType2_mp->transform(pc.R, pc.duxdx, duxdxModes);
        nufftType2_mp->transform(pc.R, pc.duxdy, duxdyModes);
        nufftType2_mp->transform(pc.R, pc.duxdz, duxdzModes);
        nufftType2_mp->transform(pc.R, pc.duydx, duydxModes);
        nufftType2_mp->transform(pc.R, pc.duydy, duydyModes);
        nufftType2_mp->transform(pc.R, pc.duydz, duydzModes);
        nufftType2_mp->transform(pc.R, pc.duzdx, duzdxModes);
        nufftType2_mp->transform(pc.R, pc.duzdy, duzdyModes);
        nufftType2_mp->transform(pc.R, pc.duzdz, duzdzModes);
    }
    void applyParticleVortexStretching3D() {
        auto& pc = *this->pcontainer_m;
        auto omega = pc.omega.getView();
        auto duxdx = pc.duxdx.getView();
        auto duxdy = pc.duxdy.getView();
        auto duxdz = pc.duxdz.getView();
        auto duydx = pc.duydx.getView();
        auto duydy = pc.duydy.getView();
        auto duydz = pc.duydz.getView();
        auto duzdx = pc.duzdx.getView();
        auto duzdy = pc.duzdy.getView();
        auto duzdz = pc.duzdz.getView();

        const T dt = T(this->dt_m);
        const auto n = pc.getLocalNum();

        Kokkos::parallel_for(
            "apply_particle_vortex_stretching_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                const T omegaX = omega(p)[0];
                const T omegaY = omega(p)[1];
                const T omegaZ = omega(p)[2];

                const T dUxdx = duxdx(p);
                const T dUxdy = duxdy(p);
                const T dUxdz = duxdz(p);
                const T dUydx = duydx(p);
                const T dUydy = duydy(p);
                const T dUydz = duydz(p);
                const T dUzdx = duzdx(p);
                const T dUzdy = duzdy(p);
                const T dUzdz = duzdz(p);

                omega(p)[0] += dt * (omegaX * dUxdx + omegaY * dUxdy + omegaZ * dUxdz);
                omega(p)[1] += dt * (omegaX * dUydx + omegaY * dUydy + omegaZ * dUydz);
                omega(p)[2] += dt * (omegaX * dUzdx + omegaY * dUzdy + omegaZ * dUzdz);
            });
        Kokkos::fence();
    }
#endif
