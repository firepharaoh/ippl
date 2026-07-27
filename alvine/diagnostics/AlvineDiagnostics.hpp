#ifndef IPPL_ALVINE_DIAGNOSTICS_ALVINEDIAGNOSTICS_HPP
#define IPPL_ALVINE_DIAGNOSTICS_ALVINEDIAGNOSTICS_HPP

    double relativeError(double value, double reference) const {
        return std::fabs((value - reference) / std::max(std::fabs(reference), 1e-30));
    }

    double computeParticleCirculation() {
        double gamma_local = 0.0;

        auto omega_view = this->pcontainer_m->omega.getView();
        auto nlocal     = this->pcontainer_m->getLocalNum();

        Kokkos::parallel_reduce(
            "particle_circulation", nlocal,
            KOKKOS_LAMBDA(const int i, double& lsum) { lsum += omega_view(i); },
            gamma_local);

        double gamma_global = 0.0;
        ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

        return gamma_global;
    }

    double computeGridCirculation() {
        double gamma_local = 0.0;

        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega_view  = omegaField.getView();

        const double dA  = hr_m[0] * hr_m[1];
        const int nghost = omegaField.getNghost();

        Kokkos::parallel_reduce(
            "grid_circulation", ippl::getRangePolicy(omega_view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                lsum += omega_view(i, j);
            },
            gamma_local);

        gamma_local *= dA;

        double gamma_global = 0.0;
        ippl::Comm->reduce(gamma_local, gamma_global, 1, std::plus<double>());

        return gamma_global;
    }

    void logCirculationDiagnostics(double circulation) {
        if (!circulation_initialized_m) {
            circulation0_m             = circulation;
            circulation_initialized_m  = true;

            if (ippl::Comm->rank() == 0) {
                std::ofstream out(diagnosticFileName("circulation.csv"), std::ios::out);
                out << "method,dt,step,time,circulation,rel_error,normalized_circulation\n";
            }
            ippl::Comm->barrier();
        }

        const double relError = relativeError(circulation, circulation0_m);
        const double normalizedCirculation =
            circulation / (std::fabs(circulation0_m) > 1e-30 ? circulation0_m : 1e-30);

        if (ippl::Comm->rank() == 0) {
            Inform m("circulation ");
            m << "circulation = " << circulation << ", relError = " << relError
              << ", normalizedCirculation = " << normalizedCirculation << endl;

            std::ofstream out(diagnosticFileName("circulation.csv"), std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);
            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << circulation << "," << relError << "," << normalizedCirculation << "\n";
        }
    }

    void checkCirculationConservation(double absError, double relError, Inform& m) {
        size_type TotalParticles = 0;
        size_type localParticles = this->pcontainer_m->getLocalNum();

        ippl::Comm->reduce(localParticles, TotalParticles, 1, std::plus<size_type>());

        if (ippl::Comm->rank() == 0) {
            const double circulationTol = 1e-10;
            if (TotalParticles != np_m || relError > circulationTol) {
                m << "Time step: " << it_m << endl;
                m << "Total particles expected: " << np_m
                  << " after update: " << TotalParticles << endl;
                m << "Abs. error in circulation conservation: " << absError << endl;
                m << "Rel. error in circulation conservation: " << relError << endl;
                ippl::Comm->abort();
            }
        }
    }

    double computeKineticEnergy() {
        double energy_local = 0.0;

        auto& uField = this->fcontainer_m->getUField();
        auto u_view  = uField.getView();

        const double dA  = hr_m[0] * hr_m[1];
        const int nghost = uField.getNghost();

        Kokkos::parallel_reduce(
            "kinetic_energy", ippl::getRangePolicy(u_view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                const double ux = u_view(i, j)[0];
                const double uy = u_view(i, j)[1];
                lsum += 0.5 * (ux * ux + uy * uy);
            },
            energy_local);

        energy_local *= dA;

        double energy_global = 0.0;
        ippl::Comm->reduce(energy_local, energy_global, 1, std::plus<double>());

        return energy_global;
    }

    void logTgvVelocityDiagnostics(const std::string& filename = "tgv_velocity_error.csv") {
        if constexpr (Dim == 2) {
            auto& uField = this->fcontainer_m->getUField();
            auto u_view  = uField.getView();

            const auto& localND = uField.getLayout().getLocalNDIndex();
            const double dA     = hr_m[0] * hr_m[1];
            const int nghost    = uField.getNghost();

            Vector_t<double, Dim> rmin = rmin_m;
            Vector_t<double, Dim> hr   = hr_m;
            const T time = time_m;
            const T viscosity = viscosity_m;

            double localErr2        = 0.0;
            double localOppositeErr2 = 0.0;
            double localRef2        = 0.0;

            Kokkos::parallel_reduce(
                "tgv_velocity_error_l2",
                ippl::getRangePolicy(u_view, nghost),
                KOKKOS_LAMBDA(const int i, const int j,
                              double& err2,
                              double& oppositeErr2,
                              double& ref2) {
                    const int gx = i - nghost + localND[0].first();
                    const int gy = j - nghost + localND[1].first();

                    const double x = rmin[0] + (gx + 0.5) * hr[0];
                    const double y = rmin[1] + (gy + 0.5) * hr[1];

                    const double uxExact =
                        TaylorGreen2D<T>::velocityX(x, y, time, viscosity);
                    const double uyExact =
                        TaylorGreen2D<T>::velocityY(x, y, time, viscosity);

                    const double dux = u_view(i, j)[0] - uxExact;
                    const double duy = u_view(i, j)[1] - uyExact;
                    const double duxOpposite = u_view(i, j)[0] + uxExact;
                    const double duyOpposite = u_view(i, j)[1] + uyExact;

                    err2 += dux * dux + duy * duy;
                    oppositeErr2 += duxOpposite * duxOpposite + duyOpposite * duyOpposite;
                    ref2 += uxExact * uxExact + uyExact * uyExact;
                },
                Kokkos::Sum<double>(localErr2),
                Kokkos::Sum<double>(localOppositeErr2),
                Kokkos::Sum<double>(localRef2));

            double localLinf         = 0.0;
            double localOppositeLinf = 0.0;
            double localRefLinf      = 0.0;

            Kokkos::parallel_reduce(
                "tgv_velocity_error_linf",
                ippl::getRangePolicy(u_view, nghost),
                KOKKOS_LAMBDA(const int i, const int j,
                              double& maxErr,
                              double& maxOppositeErr,
                              double& maxRef) {
                    const int gx = i - nghost + localND[0].first();
                    const int gy = j - nghost + localND[1].first();

                    const double x = rmin[0] + (gx + 0.5) * hr[0];
                    const double y = rmin[1] + (gy + 0.5) * hr[1];

                    const double uxExact =
                        TaylorGreen2D<T>::velocityX(x, y, time, viscosity);
                    const double uyExact =
                        TaylorGreen2D<T>::velocityY(x, y, time, viscosity);

                    const double dux = u_view(i, j)[0] - uxExact;
                    const double duy = u_view(i, j)[1] - uyExact;
                    const double duxOpposite = u_view(i, j)[0] + uxExact;
                    const double duyOpposite = u_view(i, j)[1] + uyExact;

                    const double err = Kokkos::sqrt(dux * dux + duy * duy);
                    const double oppositeErr =
                        Kokkos::sqrt(duxOpposite * duxOpposite + duyOpposite * duyOpposite);
                    const double ref = Kokkos::sqrt(uxExact * uxExact + uyExact * uyExact);

                    if (err > maxErr) {
                        maxErr = err;
                    }
                    if (oppositeErr > maxOppositeErr) {
                        maxOppositeErr = oppositeErr;
                    }
                    if (ref > maxRef) {
                        maxRef = ref;
                    }
                },
                Kokkos::Max<double>(localLinf),
                Kokkos::Max<double>(localOppositeLinf),
                Kokkos::Max<double>(localRefLinf));

            localErr2 *= dA;
            localOppositeErr2 *= dA;
            localRef2 *= dA;

            double globalErr2         = 0.0;
            double globalOppositeErr2 = 0.0;
            double globalRef2         = 0.0;
            double globalLinf         = 0.0;
            double globalOppositeLinf = 0.0;
            double globalRefLinf      = 0.0;

            ippl::Comm->reduce(localErr2, globalErr2, 1, std::plus<double>());
            ippl::Comm->reduce(localOppositeErr2, globalOppositeErr2, 1, std::plus<double>());
            ippl::Comm->reduce(localRef2, globalRef2, 1, std::plus<double>());
            ippl::Comm->reduce(localLinf, globalLinf, 1, std::greater<double>());
            ippl::Comm->reduce(localOppositeLinf, globalOppositeLinf, 1, std::greater<double>());
            ippl::Comm->reduce(localRefLinf, globalRefLinf, 1, std::greater<double>());

            const double l2Error = std::sqrt(globalErr2);
            const double l2OppositeError = std::sqrt(globalOppositeErr2);
            const double l2Reference = std::sqrt(std::max(globalRef2, 1e-30));
            const double linfReference = std::max(globalRefLinf, 1e-30);

            if (ippl::Comm->rank() == 0) {
                const bool firstWrite = !tgv_velocity_diagnostics_initialized_m;
                std::ofstream out(diagnosticFileName(filename),
                                  firstWrite ? std::ios::out : std::ios::app);
                out.precision(16);
                out.setf(std::ios::scientific, std::ios::floatfield);

                if (firstWrite) {
                    out << "method,dt,step,time,l2_error,l2_rel_error,linf_error,linf_rel_error,"
                        << "opposite_sign_l2_error,opposite_sign_l2_rel_error,"
                        << "opposite_sign_linf_error,opposite_sign_linf_rel_error\n";
                }

                out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                    << l2Error << "," << l2Error / l2Reference << ","
                    << globalLinf << "," << globalLinf / linfReference << ","
                    << l2OppositeError << "," << l2OppositeError / l2Reference << ","
                    << globalOppositeLinf << "," << globalOppositeLinf / linfReference << "\n";

                Inform m("tgv_velocity_error ");
                m << "l2Rel = " << l2Error / l2Reference
                  << ", linfRel = " << globalLinf / linfReference
                  << ", oppositeSignL2Rel = " << l2OppositeError / l2Reference
                  << ", oppositeSignLinfRel = " << globalOppositeLinf / linfReference << endl;
            }
            tgv_velocity_diagnostics_initialized_m = true;
        }
    }

    void logTgvVorticityDiagnostics(const std::string& filename = "tgv_vorticity_error.csv") {
        double relL2Error = computeRelativeL2VorticityError(this->fcontainer_m->getOmegaField());

        if (ippl::Comm->rank() == 0) {
            const bool firstWrite = !tgv_vorticity_diagnostics_initialized_m;
            std::ofstream out(diagnosticFileName(filename),
                              firstWrite ? std::ios::out : std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            if (firstWrite) {
                out << "method,dt,step,time,rel_l2_error\n";
            }

            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << relL2Error << "\n";

            Inform m("tgv_vorticity_error ");
            m << "relL2Error = " << relL2Error << endl;
        }
        tgv_vorticity_diagnostics_initialized_m = true;
    }
    double computeRelativeL2VorticityError(Field<T, Dim>& omegaNum){
        
        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega_view = omegaField.getView();
        auto omegaNum_view = omegaNum.getView();
        Field<T, Dim> omegaExact;
        Field<T, Dim> omegaError;

        omegaExact.initialize(this->fcontainer_m->getMesh(), this->fcontainer_m->getFL());
        omegaError.initialize(this->fcontainer_m->getMesh(), this->fcontainer_m->getFL());

        omegaExact = 0.0;
        omegaError = 0.0;
        this->fillExactTGVVorticity(omegaExact, this->time_m);

        omegaError = omegaNum - omegaExact;

        double errorNorm = norm(omegaError, 2);
        double exactNorm = norm(omegaExact, 2);

        return errorNorm / std::max(exactNorm, 1e-30);
       
    } 

    void checkEnergyConservation(double energy, double relError, Inform& m) {
        if (ippl::Comm->rank() == 0) {
            m << "kinetic energy = " << energy << ", relError = " << relError << endl;
        }
    }

    double computeEnstrophy() {
        double enstrophy_local = 0.0;

        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega_view  = omegaField.getView();
        const double dA  = hr_m[0] * hr_m[1];

        const int nghost = omegaField.getNghost();

        Kokkos::parallel_reduce(
            "enstrophy", ippl::getRangePolicy(omega_view, nghost),
            KOKKOS_LAMBDA(const int i, const int j, double& lsum) {
                const double omega = omega_view(i, j);
                lsum += 0.5 * omega * omega;
            },
            enstrophy_local);

        enstrophy_local *= dA;

        double enstrophy_global = 0.0;
        ippl::Comm->reduce(enstrophy_local, enstrophy_global, 1, std::plus<double>());

        return enstrophy_global;
    }

    double computeDivergenceL2() {
        auto& uField = this->fcontainer_m->getUField();
        // uField.fillHalo();

        auto divField = this->fcontainer_m->getOmegaField().deepCopy();

        divField     = div(uField);
        double N     = this->nr_m[0] * this->nr_m[1];
        double div_l2 = norm(divField, 2) / std::sqrt(N);

        // restore omega by recomputing par2grid later if needed
        return div_l2;
    }

#endif
