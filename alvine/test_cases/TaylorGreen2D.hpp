#ifndef IPPL_ALVINE_TEST_CASES_TAYLORGREEN2D_HPP
#define IPPL_ALVINE_TEST_CASES_TAYLORGREEN2D_HPP

template <typename T>
struct TaylorGreen2D {
    static Vector_t<T, 2> domainMin() {
        return Vector_t<T, 2>(0.0);
    }

    static Vector_t<T, 2> domainMax() {
        return Vector_t<T, 2>(T(2.0) * T(std::acos(-1.0)));
    }

    KOKKOS_INLINE_FUNCTION
    static T decay(const T time, const T viscosity) {
        return Kokkos::exp(-T(2.0) * viscosity * time);
    }

    KOKKOS_INLINE_FUNCTION
    static T vorticity(const T x, const T y, const T time, const T viscosity) {
        return T(2.0) * Kokkos::cos(x) * Kokkos::cos(y) * decay(time, viscosity);
    }

    KOKKOS_INLINE_FUNCTION
    static T velocityX(const T x, const T y, const T time, const T viscosity) {
        return -Kokkos::cos(x) * Kokkos::sin(y) * decay(time, viscosity);
    }

    KOKKOS_INLINE_FUNCTION
    static T velocityY(const T x, const T y, const T time, const T viscosity) {
        return Kokkos::sin(x) * Kokkos::cos(y) * decay(time, viscosity);
    }

    template <unsigned Dim>
    static void fillVorticity(Field<T, Dim>& omegaField,
                              FieldLayout_t<Dim>& layout,
                              const Vector_t<double, Dim>& rmin,
                              const Vector_t<double, Dim>& hr,
                              const T time,
                              const T viscosity) {
        static_assert(Dim == 2, "TaylorGreen2D::fillVorticity is only valid for Dim == 2");

        auto omega_view = omegaField.getView();
        const auto localND = layout.getLocalNDIndex();

        const int i0     = localND[0].first();
        const int i1     = localND[0].last();
        const int j0     = localND[1].first();
        const int j1     = localND[1].last();
        const int nghost = omegaField.getNghost();

        Kokkos::parallel_for(
            "fill_taylor_green_2d_vorticity",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({nghost, nghost},
                                                   {nghost + i1 - i0 + 1,
                                                    nghost + j1 - j0 + 1}),
            KOKKOS_LAMBDA(const int li, const int lj) {
                const int i = i0 + li - nghost;
                const int j = j0 + lj - nghost;
                const T x   = rmin[0] + (i + T(0.5)) * hr[0];
                const T y   = rmin[1] + (j + T(0.5)) * hr[1];

                omega_view(li, lj) = TaylorGreen2D<T>::vorticity(x, y, time, viscosity);
            });
        Kokkos::fence();
    }
};

#endif
