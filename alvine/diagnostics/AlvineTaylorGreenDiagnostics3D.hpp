#ifndef IPPL_ALVINE_DIAGNOSTICS_ALVINETAYLORGREENDIAGNOSTICS3D_HPP
#define IPPL_ALVINE_DIAGNOSTICS_ALVINETAYLORGREENDIAGNOSTICS3D_HPP

    double computeTGVVelocityError3D() {
        auto& uField = this->fcontainer_m->getUField();
        auto u = uField.getView();

        const auto& localND = uField.getLayout().getLocalNDIndex();
        const int nghost = uField.getNghost();
        const T cellVolume = hr_m[0] * hr_m[1] * hr_m[2];
        const Vector_t<double, Dim> rmin = rmin_m;
        const Vector_t<double, Dim> hr = hr_m;

        double localErr2 = 0.0;
        double localRef2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "tgv_velocity_error_l2_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(u.extent(0)) - nghost,
                         static_cast<int>(u.extent(1)) - nghost,
                         static_cast<int>(u.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& err2, double& ref2) {
                const int gx = i - nghost + localND[0].first();
                const int gy = j - nghost + localND[1].first();
                const int gz = k - nghost + localND[2].first();

                const T x = rmin[0] + (gx + T(0.5)) * hr[0];
                const T y = rmin[1] + (gy + T(0.5)) * hr[1];
                const T z = rmin[2] + (gz + T(0.5)) * hr[2];

                const auto exact = TaylorGreen3D<T>::velocity(x, y, z);
                const T dux = u(i, j, k)[0] - exact[0];
                const T duy = u(i, j, k)[1] - exact[1];
                const T duz = u(i, j, k)[2] - exact[2];

                err2 += dux * dux + duy * duy + duz * duz;
                ref2 += exact[0] * exact[0] + exact[1] * exact[1] + exact[2] * exact[2];
            },
            Kokkos::Sum<double>(localErr2),
            Kokkos::Sum<double>(localRef2));

        localErr2 *= cellVolume;
        localRef2 *= cellVolume;

        double globalErr2 = 0.0;
        double globalRef2 = 0.0;
        ippl::Comm->allreduce(localErr2, globalErr2, 1, std::plus<double>());
        ippl::Comm->allreduce(localRef2, globalRef2, 1, std::plus<double>());

        return std::sqrt(globalErr2 / std::max(globalRef2, 1e-30));
    }

    double computeTGVVelocityProjectionScale3D() {
        auto& uField = this->fcontainer_m->getUField();
        auto u = uField.getView();

        const auto& localND = uField.getLayout().getLocalNDIndex();
        const int nghost = uField.getNghost();
        const T cellVolume = hr_m[0] * hr_m[1] * hr_m[2];
        const Vector_t<double, Dim> rmin = rmin_m;
        const Vector_t<double, Dim> hr = hr_m;

        double localDot = 0.0;
        double localRef2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "tgv_velocity_projection_scale_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(u.extent(0)) - nghost,
                         static_cast<int>(u.extent(1)) - nghost,
                         static_cast<int>(u.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& dot, double& ref2) {
                const int gx = i - nghost + localND[0].first();
                const int gy = j - nghost + localND[1].first();
                const int gz = k - nghost + localND[2].first();

                const T x = rmin[0] + (gx + T(0.5)) * hr[0];
                const T y = rmin[1] + (gy + T(0.5)) * hr[1];
                const T z = rmin[2] + (gz + T(0.5)) * hr[2];

                const auto exact = TaylorGreen3D<T>::velocity(x, y, z);
                dot += u(i, j, k)[0] * exact[0]
                     + u(i, j, k)[1] * exact[1]
                     + u(i, j, k)[2] * exact[2];
                ref2 += exact[0] * exact[0] + exact[1] * exact[1] + exact[2] * exact[2];
            },
            Kokkos::Sum<double>(localDot),
            Kokkos::Sum<double>(localRef2));

        localDot *= cellVolume;
        localRef2 *= cellVolume;

        double globalDot = 0.0;
        double globalRef2 = 0.0;
        ippl::Comm->allreduce(localDot, globalDot, 1, std::plus<double>());
        ippl::Comm->allreduce(localRef2, globalRef2, 1, std::plus<double>());

        return globalDot / std::max(globalRef2, 1e-30);
    }

    Vector_t<double, Dim> computeTGVVelocityComponentErrors3D() {
        auto& uField = this->fcontainer_m->getUField();
        auto u = uField.getView();

        const auto& localND = uField.getLayout().getLocalNDIndex();
        const int nghost = uField.getNghost();
        const T cellVolume = hr_m[0] * hr_m[1] * hr_m[2];
        const Vector_t<double, Dim> rmin = rmin_m;
        const Vector_t<double, Dim> hr = hr_m;

        double localUxErr2 = 0.0;
        double localUyErr2 = 0.0;
        double localUzErr2 = 0.0;
        double localUxRef2 = 0.0;
        double localUyRef2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "tgv_velocity_component_errors_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(u.extent(0)) - nghost,
                         static_cast<int>(u.extent(1)) - nghost,
                         static_cast<int>(u.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& uxErr2, double& uyErr2, double& uzErr2,
                          double& uxRef2, double& uyRef2) {
                const int gx = i - nghost + localND[0].first();
                const int gy = j - nghost + localND[1].first();
                const int gz = k - nghost + localND[2].first();

                const T x = rmin[0] + (gx + T(0.5)) * hr[0];
                const T y = rmin[1] + (gy + T(0.5)) * hr[1];
                const T z = rmin[2] + (gz + T(0.5)) * hr[2];

                const auto exact = TaylorGreen3D<T>::velocity(x, y, z);
                const T dux = u(i, j, k)[0] - exact[0];
                const T duy = u(i, j, k)[1] - exact[1];
                const T duz = u(i, j, k)[2] - exact[2];

                uxErr2 += dux * dux;
                uyErr2 += duy * duy;
                uzErr2 += duz * duz;
                uxRef2 += exact[0] * exact[0];
                uyRef2 += exact[1] * exact[1];
            },
            Kokkos::Sum<double>(localUxErr2),
            Kokkos::Sum<double>(localUyErr2),
            Kokkos::Sum<double>(localUzErr2),
            Kokkos::Sum<double>(localUxRef2),
            Kokkos::Sum<double>(localUyRef2));

        localUxErr2 *= cellVolume;
        localUyErr2 *= cellVolume;
        localUzErr2 *= cellVolume;
        localUxRef2 *= cellVolume;
        localUyRef2 *= cellVolume;

        double globalUxErr2 = 0.0;
        double globalUyErr2 = 0.0;
        double globalUzErr2 = 0.0;
        double globalUxRef2 = 0.0;
        double globalUyRef2 = 0.0;
        ippl::Comm->allreduce(localUxErr2, globalUxErr2, 1, std::plus<double>());
        ippl::Comm->allreduce(localUyErr2, globalUyErr2, 1, std::plus<double>());
        ippl::Comm->allreduce(localUzErr2, globalUzErr2, 1, std::plus<double>());
        ippl::Comm->allreduce(localUxRef2, globalUxRef2, 1, std::plus<double>());
        ippl::Comm->allreduce(localUyRef2, globalUyRef2, 1, std::plus<double>());

        Vector_t<double, Dim> errors;
        errors[0] = std::sqrt(globalUxErr2 / std::max(globalUxRef2, 1e-30));
        errors[1] = std::sqrt(globalUyErr2 / std::max(globalUyRef2, 1e-30));
        errors[2] = std::sqrt(globalUzErr2);
        return errors;
    }

    double computeTGVVorticityError3D() {
        auto& omegaField = this->fcontainer_m->getOmegaField();
        auto omega = omegaField.getView();

        const auto& localND = omegaField.getLayout().getLocalNDIndex();
        const int nghost = omegaField.getNghost();
        const T cellVolume = hr_m[0] * hr_m[1] * hr_m[2];
        const Vector_t<double, Dim> rmin = rmin_m;
        const Vector_t<double, Dim> hr = hr_m;

        double localErr2 = 0.0;
        double localRef2 = 0.0;
        using policy_type = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(
            "tgv_vorticity_error_l2_3d",
            policy_type({nghost, nghost, nghost},
                        {static_cast<int>(omega.extent(0)) - nghost,
                         static_cast<int>(omega.extent(1)) - nghost,
                         static_cast<int>(omega.extent(2)) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k,
                          double& err2, double& ref2) {
                const int gx = i - nghost + localND[0].first();
                const int gy = j - nghost + localND[1].first();
                const int gz = k - nghost + localND[2].first();

                const T x = rmin[0] + (gx + T(0.5)) * hr[0];
                const T y = rmin[1] + (gy + T(0.5)) * hr[1];
                const T z = rmin[2] + (gz + T(0.5)) * hr[2];

                const auto exact = TaylorGreen3D<T>::vorticity(x, y, z);
                const T dox = omega(i, j, k)[0] - exact[0];
                const T doy = omega(i, j, k)[1] - exact[1];
                const T doz = omega(i, j, k)[2] - exact[2];

                err2 += dox * dox + doy * doy + doz * doz;
                ref2 += exact[0] * exact[0] + exact[1] * exact[1] + exact[2] * exact[2];
            },
            Kokkos::Sum<double>(localErr2),
            Kokkos::Sum<double>(localRef2));

        localErr2 *= cellVolume;
        localRef2 *= cellVolume;

        double globalErr2 = 0.0;
        double globalRef2 = 0.0;
        ippl::Comm->allreduce(localErr2, globalErr2, 1, std::plus<double>());
        ippl::Comm->allreduce(localRef2, globalRef2, 1, std::plus<double>());

        return std::sqrt(globalErr2 / std::max(globalRef2, 1e-30));
    }

    void logTaylorGreenDiagnostics3D(
        const std::string& filename = "tgv_3d_field_error.csv") {
        const double velocityRelL2 = computeTGVVelocityError3D();
        const double velocityProjectionScale = computeTGVVelocityProjectionScale3D();
        const Vector_t<double, Dim> velocityComponentErrors =
            computeTGVVelocityComponentErrors3D();
        const double vorticityRelL2 = computeTGVVorticityError3D();

        if (ippl::Comm->rank() == 0) {
            const bool firstWrite = !tgv_3d_diagnostics_initialized_m;
            std::ofstream out(diagnosticFileName(filename),
                              firstWrite ? std::ios::out : std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            if (firstWrite) {
                out << "method,dt,step,time,velocity_rel_l2_error,velocity_projection_scale,"
                    << "ux_rel_l2_error,uy_rel_l2_error,uz_l2_error,"
                    << "vorticity_rel_l2_error\n";
            }

            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << velocityRelL2 << "," << velocityProjectionScale << ","
                << velocityComponentErrors[0] << "," << velocityComponentErrors[1] << ","
                << velocityComponentErrors[2] << ","
                << vorticityRelL2 << "\n";

            Inform m("tgv_3d_field_error ");
            m << "velocityRelL2 = " << velocityRelL2
              << ", velocityProjectionScale = " << velocityProjectionScale
              << ", uxRelL2 = " << velocityComponentErrors[0]
              << ", uyRelL2 = " << velocityComponentErrors[1]
              << ", uzL2 = " << velocityComponentErrors[2]
              << ", vorticityRelL2 = " << vorticityRelL2 << endl;
        }

        tgv_3d_diagnostics_initialized_m = true;
    }

#endif
