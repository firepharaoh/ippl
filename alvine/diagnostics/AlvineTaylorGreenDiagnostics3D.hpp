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
        const double vorticityRelL2 = computeTGVVorticityError3D();

        if (ippl::Comm->rank() == 0) {
            const bool firstWrite = !tgv_3d_diagnostics_initialized_m;
            std::ofstream out(diagnosticFileName(filename),
                              firstWrite ? std::ios::out : std::ios::app);
            out.precision(16);
            out.setf(std::ios::scientific, std::ios::floatfield);

            if (firstWrite) {
                out << "method,dt,step,time,velocity_rel_l2_error,"
                    << "vorticity_rel_l2_error\n";
            }

            out << method_m << "," << dt_m << "," << it_m << "," << time_m << ","
                << velocityRelL2 << "," << vorticityRelL2 << "\n";

            Inform m("tgv_3d_field_error ");
            m << "velocityRelL2 = " << velocityRelL2
              << ", vorticityRelL2 = " << vorticityRelL2 << endl;
        }

        tgv_3d_diagnostics_initialized_m = true;
    }

#endif
