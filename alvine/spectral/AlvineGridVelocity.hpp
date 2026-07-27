#ifndef IPPL_ALVINE_SPECTRAL_ALVINEGRIDVELOCITY_HPP
#define IPPL_ALVINE_SPECTRAL_ALVINEGRIDVELOCITY_HPP

    void computeVelocityField() {
        VField_t<T, Dim> u_field = this->fcontainer_m->getUField();
        u_field = 0.0;

        if constexpr (Dim == 2) {
            const int nghost = u_field.getNghost();
            auto view        = u_field.getView();

            auto omega_view = this->fcontainer_m->getOmegaField().getView();
            this->fcontainer_m->getOmegaField().fillHalo();

            Vector_t<double, Dim> hr = hr_m;
            Kokkos::parallel_for(
                "Assign rhs", ippl::getRangePolicy(view, nghost),
                KOKKOS_LAMBDA(const int i, const int j) {
                    view(i, j) = {
                        (omega_view(i, j + 1) - omega_view(i, j - 1)) / (2 * hr(1)),
                        -(omega_view(i + 1, j) - omega_view(i - 1, j)) / (2 * hr(0))};
                });
        } else if constexpr (Dim == 3) {
            // TODO compute velocity field in 3D, this should be a simple curl operation (one line)
        }
    }

#endif
