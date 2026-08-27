#ifndef IPPL_ALVINE_SPECTRAL_ALVINESPECTRALSETUP3D_HPP
#define IPPL_ALVINE_SPECTRAL_ALVINESPECTRALSETUP3D_HPP

void initNUFFT3D(double tol = 1e-10) {
    auto& FL   = this->fcontainer_m->getFL();
    auto& mesh = this->fcontainer_m->getMesh();

    omega_x_hat_m.initialize(mesh, FL);
    omega_y_hat_m.initialize(mesh, FL);
    omega_z_hat_m.initialize(mesh, FL);

    ux_hat_m.initialize(mesh, FL);
    uy_hat_m.initialize(mesh, FL);
    uz_hat_m.initialize(mesh, FL);

    duxdx_hat_m.initialize(mesh, FL);
    duxdy_hat_m.initialize(mesh, FL);
    duxdz_hat_m.initialize(mesh, FL);
    duydx_hat_m.initialize(mesh, FL);
    duydy_hat_m.initialize(mesh, FL);
    duydz_hat_m.initialize(mesh, FL);
    duzdx_hat_m.initialize(mesh, FL);
    duzdy_hat_m.initialize(mesh, FL);
    duzdz_hat_m.initialize(mesh, FL);

    viscosity_x_hat_m.initialize(mesh, FL);
    viscosity_y_hat_m.initialize(mesh, FL);
    viscosity_z_hat_m.initialize(mesh, FL);
    Sk_m.initialize(mesh, FL);
    if (useShapeFunctionFilter()) {
        initializeShapeFunctionVIF3D();
    }

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);
    spectralFft_mp = std::make_shared<SpectralFft_t>(FL, fftParams);

    rebuildNUFFTPlans3D(tol);
}

void rebuildNUFFTPlans3D(double tol = 1e-10) {
    ippl::ParameterList p1, p2;
    p1.add("tolerance", tol);
    p2.add("tolerance", tol);

    p1.add("use_finufft", false);
    p2.add("use_finufft", false);
    p1.add("use_upsampled_inputs", false);
    p2.add("use_upsampled_inputs", false);
    p1.add("spread_method", "tiled");
    p2.add("gather_method", "atomic_sort");

    auto& FL = this->fcontainer_m->getFL();

    nufftType1_mp =
        std::make_shared<Nufft_t>(FL, this->pcontainer_m->getLocalNum(), 1, p1);

    nufftType2_mp =
        std::make_shared<Nufft_t>(FL, this->pcontainer_m->getLocalNum(), 2, p2);
}
#endif
