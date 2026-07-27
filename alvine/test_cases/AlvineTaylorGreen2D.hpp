#ifndef IPPL_ALVINE_TEST_CASES_ALVINETAYLORGREEN2D_HPP
#define IPPL_ALVINE_TEST_CASES_ALVINETAYLORGREEN2D_HPP

    void fillExactTGVVorticity(Field<T, Dim>& omegaField) {
        auto omega_view = omegaField.getView();

        auto localND = this->fcontainer_m->getFL().getLocalNDIndex();

        const int i0     = localND[0].first();
        const int i1     = localND[0].last();
        const int j0     = localND[1].first();
        const int j1     = localND[1].last();
        const int nghost = omegaField.getNghost();

        Vector_t<double, Dim> rmin = this->rmin_m;
        Vector_t<double, Dim> hr   = this->hr_m;

        Kokkos::parallel_for(
            "fill_exact_tgv_vorticity",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost},
                                                   {nghost + i1 - i0 + 1,
                                                    nghost + j1 - j0 + 1}),
            KOKKOS_LAMBDA(const int li, const int lj) {
                const int i    = i0 + li - nghost;
                const int j    = j0 + lj - nghost;
                const double x = rmin[0] + (i + 0.5) * hr[0];
                const double y = rmin[1] + (j + 0.5) * hr[1];

                omega_view(li, lj) = 2.0 * cos(x) * cos(y);
            });

        Kokkos::fence();
    }

#endif
