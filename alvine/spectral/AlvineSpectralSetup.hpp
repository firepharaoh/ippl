#ifndef IPPL_ALVINE_SPECTRAL_ALVINESPECTRALSETUP_HPP
#define IPPL_ALVINE_SPECTRAL_ALVINESPECTRALSETUP_HPP

    void initNUFFT(double tol = 1e-10) {
        initNUFFT2D(tol);
    }

    void initNUFFT2D(double tol = 1e-10) {
        ippl::ParameterList p1, p2;

        p1.add("tolerance", tol);
        p2.add("tolerance", tol);

        // 2D currently uses native NUFFT path. FINUFFT path is 3D-only here.
        p1.add("use_finufft", false);
        p2.add("use_finufft", false);
        p1.add("use_upsampled_inputs", false);
        p2.add("use_upsampled_inputs", false);
        p1.add("spread_method", "tiled");
        p2.add("gather_method", "atomic_sort");

        auto& FL   = this->fcontainer_m->getFL();
        auto& mesh = this->fcontainer_m->getMesh();

        omega_hat_m.initialize(mesh, FL);
        ux_hat_m.initialize(mesh, FL);
        uy_hat_m.initialize(mesh, FL);
        viscosity_hat_m.initialize(mesh, FL);
        Sk_m.initialize(mesh, FL);
        if (useShapeFunctionFilter()) {
            initializeShapeFunctionVIF();
        }

        ippl::ParameterList fftParams;
        fftParams.add("use_heffte_defaults", true);
        spectralFft_mp = std::make_shared<SpectralFft_t>(FL, fftParams);

        nufftType1_mp =
            std::make_shared<Nufft_t>(FL, this->pcontainer_m->getLocalNum(), 1, p1);

        nufftType2_mp =
            std::make_shared<Nufft_t>(FL, this->pcontainer_m->getLocalNum(), 2, p2);
    }

#endif
