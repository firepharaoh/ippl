#ifndef IPPL_ALVINE_SPECTRAL_ALVINESPECTRALOPS3D_HPP
#define IPPL_ALVINE_SPECTRAL_ALVINESPECTRALOPS3D_HPP

    void spectralScatter3D() {
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

        if (useShapeFunctionFilter()) {
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
        }
        Kokkos::fence();
    }

    void spectralScatter() {
        spectralScatter3D();
    }

    void computeSpectralVelocityModes3D() {
        auto ox = omega_x_hat_m.getView();
        auto oy = omega_y_hat_m.getView();
        auto oz = omega_z_hat_m.getView();

        auto ux = ux_hat_m.getView();
        auto uy = uy_hat_m.getView();
        auto uz = uz_hat_m.getView();

        auto& layout = omega_x_hat_m.getLayout();
        auto& mesh   = omega_x_hat_m.get_mesh();

        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = omega_x_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const int Nz = domain[2].length();

        const T Lx = dx[0] * Nx;
        const T Ly = dx[1] * Ny;
        const T Lz = dx[2] * Nz;
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

        if (useShapeFunctionFilter()) {
            auto uxModes = ux_hat_m.deepCopy();
            auto uyModes = uy_hat_m.deepCopy();
            auto uzModes = uz_hat_m.deepCopy();
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

            nufftType2_mp->transform(pc.R, pc.ux, uxModes);
            nufftType2_mp->transform(pc.R, pc.uy, uyModes);
            nufftType2_mp->transform(pc.R, pc.uz, uzModes);
        } else {
            nufftType2_mp->transform(pc.R, pc.ux, ux_hat_m);
            nufftType2_mp->transform(pc.R, pc.uy, uy_hat_m);
            nufftType2_mp->transform(pc.R, pc.uz, uz_hat_m);
        }

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
        computeSpectralVelocityModes3D();
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

                viscX(i, j, k) = -nu * k2 * ox(i, j, k);
                viscY(i, j, k) = -nu * k2 * oy(i, j, k);
                viscZ(i, j, k) = -nu * k2 * oz(i, j, k);
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

        nufftType2_mp->transform(pc.R, pc.viscosity_x, viscosity_x_hat_m);
        nufftType2_mp->transform(pc.R, pc.viscosity_y, viscosity_y_hat_m);
        nufftType2_mp->transform(pc.R, pc.viscosity_z, viscosity_z_hat_m);

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
        const T dt = T(this->dt_m);
        const auto n = pc.getLocalNum();

        Kokkos::parallel_for(
            "apply_particle_viscosity_3d",
            n,
            KOKKOS_LAMBDA(const size_t p) {
                omega(p)[0] += dt * visc(p)[0];
                omega(p)[1] += dt * visc(p)[1];
                omega(p)[2] += dt * visc(p)[2];
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

#endif
