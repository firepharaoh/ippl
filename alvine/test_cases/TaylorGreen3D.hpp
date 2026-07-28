#ifndef IPPL_ALVINE_TEST_CASES_TAYLORGREEN3D_HPP
#define IPPL_ALVINE_TEST_CASES_TAYLORGREEN3D_HPP

template <typename T>
struct TaylorGreen3D {
    static Vector_t<T, 3> domainMin() {
        return Vector_t<T, 3>(0.0);
    }

    static Vector_t<T, 3> domainMax() {
        return Vector_t<T, 3>(T(2.0) * T(std::acos(-1.0)));
    }

    KOKKOS_INLINE_FUNCTION
    static T velocityX(const T x, const T y, const T z) {
        return Kokkos::sin(x) * Kokkos::cos(y) * Kokkos::cos(z);
    }

    KOKKOS_INLINE_FUNCTION
    static T velocityY(const T x, const T y, const T z) {
        return -Kokkos::cos(x) * Kokkos::sin(y) * Kokkos::cos(z);
    }

    KOKKOS_INLINE_FUNCTION
    static T velocityZ(const T, const T, const T) {
        return T(0.0);
    }

    KOKKOS_INLINE_FUNCTION
    static T vorticityX(const T x, const T y, const T z) {
        return -Kokkos::cos(x) * Kokkos::sin(y) * Kokkos::sin(z);
    }

    KOKKOS_INLINE_FUNCTION
    static T vorticityY(const T x, const T y, const T z) {
        return -Kokkos::sin(x) * Kokkos::cos(y) * Kokkos::sin(z);
    }

    KOKKOS_INLINE_FUNCTION
    static T vorticityZ(const T x, const T y, const T z) {
        return T(2.0) * Kokkos::sin(x) * Kokkos::sin(y) * Kokkos::cos(z);
    }

    KOKKOS_INLINE_FUNCTION
    static Vector_t<T, 3> velocity(const T x, const T y, const T z) {
        return Vector_t<T, 3>(velocityX(x, y, z), velocityY(x, y, z),
                              velocityZ(x, y, z));
    }

    KOKKOS_INLINE_FUNCTION
    static Vector_t<T, 3> vorticity(const T x, const T y, const T z) {
        return Vector_t<T, 3>(vorticityX(x, y, z), vorticityY(x, y, z),
                              vorticityZ(x, y, z));
    }

    template <unsigned Dim>
    static void fillVorticity(VField_t<T, Dim>& omegaField,
                              FieldLayout_t<Dim>& layout,
                              const Vector_t<double, Dim>& rmin,
                              const Vector_t<double, Dim>& hr) {
        static_assert(Dim == 3, "TaylorGreen3D::fillVorticity is only valid for Dim == 3");

        auto omega_view = omegaField.getView();
        const auto localND = layout.getLocalNDIndex();

        const int i0 = localND[0].first();
        const int i1 = localND[0].last();
        const int j0 = localND[1].first();
        const int j1 = localND[1].last();
        const int k0 = localND[2].first();
        const int k1 = localND[2].last();
        const int nghost = omegaField.getNghost();

        Kokkos::parallel_for(
            "fill_taylor_green_3d_vorticity",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nghost, nghost, nghost},
                                                   {nghost + i1 - i0 + 1,
                                                    nghost + j1 - j0 + 1,
                                                    nghost + k1 - k0 + 1}),
            KOKKOS_LAMBDA(const int li, const int lj, const int lk) {
                const int i = i0 + li - nghost;
                const int j = j0 + lj - nghost;
                const int k = k0 + lk - nghost;
                const T x = rmin[0] + (i + T(0.5)) * hr[0];
                const T y = rmin[1] + (j + T(0.5)) * hr[1];
                const T z = rmin[2] + (k + T(0.5)) * hr[2];

                omega_view(li, lj, lk) = TaylorGreen3D<T>::vorticity(x, y, z);
            });
        Kokkos::fence();
    }
};

#endif
