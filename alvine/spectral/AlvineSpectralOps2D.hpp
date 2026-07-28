#ifndef IPPL_ALVINE_SPECTRAL_ALVINESPECTRALOPS2D_HPP
#define IPPL_ALVINE_SPECTRAL_ALVINESPECTRALOPS2D_HPP

    // SPECTRAL IMPLEMENTATION
    void spectralScatter2D() {
      if constexpr (Dim == 2) {
        if (!nufftType1_mp) {
          throw std::runtime_error("AlvineManager::spectralScatter called before initNUFFT");
        }

        omega_hat_m = Kokkos::complex<T>(0.0, 0.0);
        nufftType1_mp->transform(
            this->pcontainer_m->R,
            this->pcontainer_m->omega,
            omega_hat_m);

        if (useShapeFunctionFilter()) {
            auto omegaView = omega_hat_m.getView();
            auto shapeView = Sk_m.getView();
            const int nghost = omega_hat_m.getNghost();
            using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
            Kokkos::parallel_for(
                "Multiply with shape function in Fourier space",
                policy_type({nghost, nghost},
                            {static_cast<int>(omegaView.extent(0)) - nghost,
                             static_cast<int>(omegaView.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j) {
                  omegaView(i,j) *= shapeView(i,j);
                });
            Kokkos::fence();
        }
      } else {
        throw std::runtime_error("AlvineManager::spectralScatter is implemented for 2D VIC only");
      }
    }

    void computeSpectralVelocityModes2D() {
      if constexpr (Dim == 2) {
        auto omega = omega_hat_m.getView();
        auto ux    = ux_hat_m.getView();
        auto uy    = uy_hat_m.getView();

        auto& layout = omega_hat_m.getLayout();
        auto& mesh   = omega_hat_m.get_mesh();

        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = omega_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const T Lx   = dx[0] * Nx;
        const T Ly   = dx[1] * Ny;
        const T area = Lx * Ly;

        const T twoPi = T(2.0 * std::acos(-1.0));
        const Kokkos::complex<T> imag(0.0, 1.0);

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "compute_spectral_velocity_modes",
            policy_type({nghost, nghost},
                        {static_cast<int>(omega.extent(0)) - nghost,
                         static_cast<int>(omega.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              const int gx = i - nghost + lDom[0].first();
              const int gy = j - nghost + lDom[1].first();

              const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
              const int my = (gy <= Ny / 2) ? gy : gy - Ny;

              const bool notMidX = (gx != Nx / 2);
              const bool notMidY = (gy != Ny / 2);

              const T laplaceKx = twoPi * mx / Lx;
              const T laplaceKy = twoPi * my / Ly;
              const T k2 = laplaceKx * laplaceKx + laplaceKy * laplaceKy;
              const T derivativeKx = notMidX * laplaceKx;
              const T derivativeKy = notMidY * laplaceKy;

              if (k2 == T(0)) {
                omega(i, j) = Kokkos::complex<T>(0.0, 0.0);
                ux(i, j) = Kokkos::complex<T>(0.0, 0.0);
                uy(i, j) = Kokkos::complex<T>(0.0, 0.0);
              } else {
                const auto psi = omega(i, j) / (area * k2);
                ux(i, j) = imag * derivativeKy * psi;
                uy(i, j) = -imag * derivativeKx * psi;
              }
            });
      } else {
        throw std::runtime_error(
            "AlvineManager::computeSpectralVelocityModes is implemented for 2D VIC only");
      }
    }

    void spectralGather2D() {
        if constexpr (Dim == 2) {
            if (!nufftType2_mp) {
                throw std::runtime_error("spectralGather called before initNUFFT");
            }

            auto& pc = *this->pcontainer_m;

            pc.ux = 0.0;
            pc.uy = 0.0;

            if (useShapeFunctionFilter()) {
                auto uxModes = ux_hat_m.deepCopy();
                auto uyModes = uy_hat_m.deepCopy();
                auto uxModeView = uxModes.getView();
                auto uyModeView = uyModes.getView();
                auto shapeView = Sk_m.getView();
                const int nghost = uxModes.getNghost();

                using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
                Kokkos::parallel_for(
                    "shape_spectral_velocity_modes",
                    policy_type({nghost, nghost},
                                {static_cast<int>(uxModeView.extent(0)) - nghost,
                                 static_cast<int>(uxModeView.extent(1)) - nghost}),
                    KOKKOS_LAMBDA(const int i, const int j) {
                        uxModeView(i, j) *= shapeView(i, j);
                        uyModeView(i, j) *= shapeView(i, j);
                    });
                Kokkos::fence();

                nufftType2_mp->transform(pc.R, pc.ux, uxModes);
                nufftType2_mp->transform(pc.R, pc.uy, uyModes);
            } else {
                nufftType2_mp->transform(pc.R, pc.ux, ux_hat_m);
                nufftType2_mp->transform(pc.R, pc.uy, uy_hat_m);
            }

            auto P       = pc.P.getView();
            auto ux      = pc.ux.getView();
            auto uy      = pc.uy.getView();
            const auto n = pc.getLocalNum();

            Kokkos::parallel_for(
                "pack_spectral_velocity", n,
                KOKKOS_LAMBDA(const size_t i) {
                    P(i)[0] = ux(i);
                    P(i)[1] = uy(i);
                });
            Kokkos::fence();
        } else {
            throw std::runtime_error("AlvineManager::spectralGather is implemented for 2D VIC only");
        }
    }
    void spectralGather() {
        spectralGather2D();
    }
    void spectralGatherViscosity2D(ComplexField_t& visc_hat) {
        if constexpr (Dim == 2) {
            if(!nufftType2_mp){
                throw std::runtime_error("spectralGatherViscosity called before initNUFFT");
            }
            auto & pc = *this->pcontainer_m;
            pc.viscosity = 0.0;
            nufftType2_mp->transform(pc.R, pc.viscosity, visc_hat);
            Kokkos::fence();
        } else {
            throw std::runtime_error("AlvineManager::spectralGatherViscosity is implemented for 2D VIC only");
        }
    }
    void spectralSolveParticles() {
      spectralScatter2D();
      computeSpectralVelocityModes2D();
      spectralGather2D();
    }

    void applyCellCenteredIfftPhase(ComplexField_t& modes) {
      if constexpr (Dim == 2) {
        auto view = modes.getView();

        auto& layout       = modes.getLayout();
        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost   = modes.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const T pi   = std::acos(T(-1.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "apply_cell_centered_ifft_phase",
            policy_type({nghost, nghost},
                        {static_cast<int>(view.extent(0)) - nghost,
                         static_cast<int>(view.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              const int gx = i - nghost + lDom[0].first();
              const int gy = j - nghost + lDom[1].first();

              const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
              const int my = (gy <= Ny / 2) ? gy : gy - Ny;

              const T phase = pi * (T(mx) / T(Nx) + T(my) / T(Ny));
              const Kokkos::complex<T> factor(Kokkos::cos(phase), Kokkos::sin(phase));

              view(i, j) *= factor;
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::applyCellCenteredIfftPhase is implemented for 2D VIC only");
      }
    }

    void Hou_Li_filter(ComplexField_t& modes, double alpha = 36.0, int exponent = 36) {
      if constexpr (Dim == 2) {
        auto view = modes.getView();

        auto& layout       = modes.getLayout();
        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost   = modes.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const T kxMax = T(Nx) / T(2.0);
        const T kyMax = T(Ny) / T(2.0);
        const T invSqrtDim = T(1.0) / Kokkos::sqrt(T(2.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "hou_li_filter",
            policy_type({nghost, nghost},
                        {static_cast<int>(view.extent(0)) - nghost,
                         static_cast<int>(view.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              const int gx = i - nghost + lDom[0].first();
              const int gy = j - nghost + lDom[1].first();

              const T kx = (gx <= Nx / 2) ? T(gx) : T(gx - Nx);
              const T ky = (gy <= Ny / 2) ? T(gy) : T(gy - Ny);

              const T etaX = Kokkos::abs(kx) / kxMax;
              const T etaY = Kokkos::abs(ky) / kyMax;
              const T eta  = Kokkos::sqrt(etaX * etaX + etaY * etaY) * invSqrtDim;
              const T filterFactor =
                  Kokkos::exp(-T(alpha) * Kokkos::pow(eta, exponent));

              view(i, j) *= filterFactor;
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::Hou_Li_filter is implemented for 2D VIC only");
      }
    }

    void initializeShapeFunctionVIF() { //Source is from InitializeShapeFunctionPIF in alpine/ChargedParticles.hpp
        if constexpr (Dim == 2) {
            using mdrange_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
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
                "B-spline shape function initialization",
                mdrange_type({nghost, nghost},
                            {static_cast<int>(Skview.extent(0)) - nghost,
                                static_cast<int>(Skview.extent(1)) - nghost}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    Vector<int, Dim> iVec  = {i,j};
                    for (unsigned d=0;d < Dim; d++) {
                        iVec[d] = iVec[d] - nghost+lDom[d].first();
                    }
                    Vector<double, Dim> kVec;
                    double Sk = 1.0;
                    for (unsigned d=0;d < Dim; d++) {
                        bool shift = (iVec[d] > (N[d]/2));
                        kVec[d] = 2 * pi / Len[d] * (iVec[d] - shift * N[d]);
                        double khbytwo = kVec[d] * dx[d] / 2.0;
                        bool isNotZero = (khbytwo != 0.0);
                        double factor = (1.0 / (khbytwo +((!isNotZero) *1.0))) ;
                        double arg = isNotZero * (Kokkos::sin(khbytwo) * factor) + (!isNotZero) * 1.0;
                        //Fourier Transform of B-spline of order n is (sin(kh/2)/(kh/2))^n, where h is the mesh spacing and k is the wavenumber
                        Sk *= Kokkos::pow(arg, order);
                    }
                    Skview(i,j) = Sk;
                }
            );
        } else {
            throw std::runtime_error(
                "AlvineManager::initializeShapeFunctionVIF is implemented for 2D VIC only");
        }

    }
    void computeSpectralViscosity(ComplexField_t& visc_hat){
      if constexpr (Dim == 2) {
        auto omega = omega_hat_m.getView();
        auto visc  = visc_hat.getView();
        
        auto& layout = omega_hat_m.getLayout();
        auto& mesh   = omega_hat_m.get_mesh();

        const auto& lDom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const auto& dx     = mesh.getMeshSpacing();
        const int nghost   = omega_hat_m.getNghost();

        const int Nx = domain[0].length();
        const int Ny = domain[1].length();
        const T Lx   = dx[0] * Nx;
        const T Ly   = dx[1] * Ny;
        const T area = Lx * Ly;

        const T twoPi = T(2.0 * std::acos(-1.0));

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "Compute spectral viscosity",
            policy_type({nghost, nghost},
                        {static_cast<int>(omega.extent(0)) - nghost,
                         static_cast<int>(omega.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
                const int gx = i - nghost + lDom[0].first();
                const int gy = j - nghost + lDom[1].first();
                const int mx = (gx <= Nx / 2) ? gx : gx - Nx;
                const int my = (gy <= Ny / 2) ? gy : gy - Ny;
                const T laplaceKx = twoPi * mx / Lx;
                const T laplaceKy = twoPi * my / Ly;
                const T k2 = laplaceKx * laplaceKx + laplaceKy * laplaceKy;
                const bool isNotZero = (k2 != T(0));
                visc(i, j) = -T(isNotZero) * viscosity_m * k2 * omega(i, j) / area;
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::computeSpectralViscosity is implemented for 2D VIF only");
      }
    }

    void updateParticleVorticityViscosity(){
        if constexpr (Dim == 2) {
            auto& pc = *this->pcontainer_m;
            auto omega = pc.omega.getView();
            auto visc = pc.viscosity.getView();
            const auto n = pc.getLocalNum();
            const T dt = dt_m;

            const unsigned nxp_global = static_cast<unsigned>(std::sqrt(this->np_m));
            const unsigned nyp_global = this->np_m / nxp_global;
            const T dxp = (this->rmax_m[0] - this->rmin_m[0]) / nxp_global;
            const T dyp = (this->rmax_m[1] - this->rmin_m[1]) / nyp_global;
            const T particleArea = dxp * dyp;

            Kokkos::parallel_for(
                "update_particle_vorticity_viscosity",
                n,
                KOKKOS_LAMBDA(const size_t i){
                    omega(i) += dt * visc(i) * particleArea;
                });
            Kokkos::fence();
        } else {
            throw std::runtime_error("AlvineManager::updateParticleVorticityViscosity is implemented for 2D only");
        }
    }
    void reconstructSpectralVorticity(RealField_t& omegaField) {
      if constexpr (Dim == 2) {
        if (!spectralFft_mp) {
          throw std::runtime_error(
              "AlvineManager::reconstructSpectralVorticity called before initNUFFT");
        }

        auto omegaModes = omega_hat_m.deepCopy();

        const auto& domain = omega_hat_m.getLayout().getDomain();
        const auto& dx     = omega_hat_m.get_mesh().getMeshSpacing();
        const T area =
            (dx[0] * domain[0].length()) * (dx[1] * domain[1].length());

        omegaModes = omegaModes / area;
        applyCellCenteredIfftPhase(omegaModes);

        spectralFft_mp->transform(ippl::BACKWARD, omegaModes);

        auto omegaOut = omegaField.getView();
        auto omegaGrid = omegaModes.getView();
        const int nghost = omegaField.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_for(
            "reconstruct_spectral_vorticity",
            policy_type({nghost, nghost},
                        {static_cast<int>(omegaOut.extent(0)) - nghost,
                         static_cast<int>(omegaOut.extent(1)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j) {
              omegaOut(i, j) = omegaGrid(i, j).real();
            });
        Kokkos::fence();
      } else {
        throw std::runtime_error(
            "AlvineManager::reconstructSpectralVorticity is implemented for 2D VIC only");
      }
    }

    void reconstructSpectralVelocity(VField_t<T, Dim>& uField) {
      if constexpr (Dim == 2) {
        if (!spectralFft_mp) {
          throw std::runtime_error(
              "AlvineManager::reconstructSpectralVelocity called before initNUFFT");
        }

        auto uxModes = ux_hat_m.deepCopy();
        auto uyModes = uy_hat_m.deepCopy();

        applyCellCenteredIfftPhase(uxModes);
        applyCellCenteredIfftPhase(uyModes);

        spectralFft_mp->transform(ippl::BACKWARD, uxModes);
        spectralFft_mp->transform(ippl::BACKWARD, uyModes);

        auto uOut   = uField.getView();
        auto uxGrid = uxModes.getView();
        auto uyGrid = uyModes.getView();
        const int nghost = uField.getNghost();

        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
	        Kokkos::parallel_for(
	            "reconstruct_spectral_velocity",
	            policy_type({nghost, nghost},
	                        {static_cast<int>(uOut.extent(0)) - nghost,
	                         static_cast<int>(uOut.extent(1)) - nghost}),
	            KOKKOS_LAMBDA(const int i, const int j) {
	              uOut(i, j)[0] = uxGrid(i, j).real();
	              uOut(i, j)[1] = uyGrid(i, j).real();
	            });
	        Kokkos::fence();

	        const auto uAverage = uField.getVolumeAverage();
	        Kokkos::parallel_for(
	            "remove_reconstructed_velocity_mean",
	            policy_type({nghost, nghost},
	                        {static_cast<int>(uOut.extent(0)) - nghost,
	                         static_cast<int>(uOut.extent(1)) - nghost}),
	            KOKKOS_LAMBDA(const int i, const int j) {
	              uOut(i, j)[0] -= uAverage[0];
	              uOut(i, j)[1] -= uAverage[1];
	            });
	        Kokkos::fence();
	      } else {
	        throw std::runtime_error(
	            "AlvineManager::reconstructSpectralVelocity is implemented for 2D VIC only");
	      }
	    }

#endif
