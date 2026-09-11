#ifndef IPPL_ALVINE_SFSL3D_TEST_SUPPORT_HPP
#define IPPL_ALVINE_SFSL3D_TEST_SUPPORT_HPP

class SFSLProbe : public SpectralFSL3DManager<T> {
public:
    using SpectralFSL3DManager<T>::SpectralFSL3DManager;

    void setShear() {
        // u=(sin(y),0,0), omega=(0,0,-cos(y)): stretching and advection
        // of omega vanish exactly, even though particles move in x.
        auto pc = this->pcontainer_m;
        auto R = pc->R.getView();
        auto w = pc->omega.getView();
        auto wx = pc->omega_x.getView();
        auto wy = pc->omega_y.getView();
        auto wz = pc->omega_z.getView();
        const T volume = particleVolume3D();
        Kokkos::parallel_for("test_initialize_shear", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p) {
                w(p)[0] = wx(p) = 0;
                w(p)[1] = wy(p) = 0;
                w(p)[2] = wz(p) = -Kokkos::cos(R(p)[1]) * volume;
            });
        Kokkos::fence();
        scatterAndSolveCurrentParticles3D();
    }

    double gridError(const bool shear, const T amplitude, const T sourceAmplitude = 0) {
        this->reconstructSpectralVorticity(this->fcontainer_m->getOmegaField());
        auto w = this->fcontainer_m->getOmegaField().getView();
        const int ng = this->fcontainer_m->getOmegaField().getNghost();
        const auto local = this->fcontainer_m->getFL().getLocalNDIndex();
        const auto dx = this->hr_m;
        T localError = 0;
        Kokkos::parallel_reduce("test_euler_grid_error", ippl::getRangePolicy(w, ng),
            KOKKOS_LAMBDA(const int i, const int j, const int k, T& error) {
                const T x = (i - ng + local[0].first() + T(0.5)) * dx[0];
                const T y = (j - ng + local[1].first() + T(0.5)) * dx[1];
                const T z = (k - ng + local[2].first() + T(0.5)) * dx[2];
                Vector_t<T,3> expected(0);
                if (shear) {
                    expected[2] = -amplitude * Kokkos::cos(y);
                } else {
                    expected = TaylorGreen3D<T>::vorticity(x,y,z) * amplitude;
                    // Independently differentiated TGV: Sx=-sin(2y)sin(2z)/4,
                    // Sy=sin(2x)sin(2z)/4, Sz=0. These modes have k^2=8.
                    expected[0] -= sourceAmplitude * Kokkos::sin(2*y) * Kokkos::sin(2*z) / 4;
                    expected[1] += sourceAmplitude * Kokkos::sin(2*x) * Kokkos::sin(2*z) / 4;
                }
                for (unsigned d=0; d<3; ++d) {
                    error = Kokkos::max(error, Kokkos::abs(w(i,j,k)[d] - expected[d]));
                }
            }, Kokkos::Max<T>(localError));
        double globalError = 0;
        MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_MAX,
                      ippl::Comm->getCommunicator());
        return globalError;
    }

    double energy() { return this->computeSpectralEnergy3D(); }
    double enstrophy() { return this->computeSpectralEnstrophy3D(); }

    double coupledStageError(const bool enableShapeFilter = false) {
        const int savedFilter = this->spectral_filter_m;
        if (enableShapeFilter) {
            this->spectral_filter_m = 1;
            this->initializeShapeFunctionVIF3D();
        }
        auto pc = this->pcontainer_m;
        pc->rk4_R0 = pc->R;
        pc->rk4_omega0 = pc->omega;
        evaluateCoupledStrangRHS3D(pc->rk4_k1, pc->rk4_omega_k1, "test_rhs");
        // Repeating the same temporary stage must not accumulate either R or
        // Gamma. Registered bases/slopes must survive pc->update migration.
        setCoupledStrangStage3D(pc->rk4_k1, pc->rk4_omega_k1, T(0.5));
        setCoupledStrangStage3D(pc->rk4_k1, pc->rk4_omega_k1, T(0.5));
        auto R = pc->R.getView(); auto R0 = pc->rk4_R0.getView();
        auto w = pc->omega.getView(); auto w0 = pc->rk4_omega0.getView();
        auto kr = pc->rk4_k1.getView(); auto kw = pc->rk4_omega_k1.getView();
        auto wx = pc->omega_x.getView(); auto wy = pc->omega_y.getView();
        auto wz = pc->omega_z.getView();
        const T volume = particleVolume3D();
        const T h = T(this->dt_m)/2;
        const auto lower = this->rmin_m;
        const Vector_t<T,3> lengths = this->rmax_m-this->rmin_m;
        T error = 0;
        Kokkos::parallel_reduce("test_coupled_stage", pc->getLocalNum(),
            KOKKOS_LAMBDA(const size_t p, T& e) {
                const T x=R0(p)[0], y=R0(p)[1], z=R0(p)[2];
                Vector_t<T,3> velocity(0), source(0);
                velocity[0] = Kokkos::sin(x)*Kokkos::cos(y)*Kokkos::cos(z);
                velocity[1] = -Kokkos::cos(x)*Kokkos::sin(y)*Kokkos::cos(z);
                source[0] = -Kokkos::sin(2*y)*Kokkos::sin(2*z)/4;
                source[1] = Kokkos::sin(2*x)*Kokkos::sin(2*z)/4;
                for (unsigned d=0; d<3; ++d) {
                    e = Kokkos::max(e, Kokkos::abs(kr(p)[d]-velocity[d]));
                    e = Kokkos::max(e, Kokkos::abs(kw(p)[d]/volume-source[d]));
                    T expected = R0(p)[d]+h*kr(p)[d];
                    expected -= lengths[d]*Kokkos::floor((expected-lower[d])/lengths[d]);
                    e = Kokkos::max(e, Kokkos::abs(R(p)[d]-expected));
                    e = Kokkos::max(e, Kokkos::abs(w(p)[d]-w0(p)[d]-h*kw(p)[d])/volume);
                }
                e = Kokkos::max(e, Kokkos::abs(wx(p)-w(p)[0]));
                e = Kokkos::max(e, Kokkos::abs(wy(p)-w(p)[1]));
                e = Kokkos::max(e, Kokkos::abs(wz(p)-w(p)[2]));
            }, Kokkos::Max<T>(error));
        T global = 0;
        MPI_Allreduce(&error, &global, 1, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
        setCoupledStrangStage3D(pc->rk4_k1, pc->rk4_omega_k1, T(0));
        this->spectralScatter3D(false);
        this->computeSpectralVelocityModes3D();
        this->spectral_filter_m = savedFilter;
        return global;
    }

    using Modes = std::array<ComplexField_t, 3>;

    Modes snapshot() {
        return {this->omega_x_hat_m.deepCopy(), this->omega_y_hat_m.deepCopy(),
                this->omega_z_hat_m.deepCopy()};
    }

    void restoreModes(Modes& state) {
        Kokkos::deep_copy(this->omega_x_hat_m.getView(), state[0].getView());
        Kokkos::deep_copy(this->omega_y_hat_m.getView(), state[1].getView());
        Kokkos::deep_copy(this->omega_z_hat_m.getView(), state[2].getView());
        this->computeSpectralVelocityModes3D(false);
    }

    double modeError(Modes& reference) {
        std::array<ComplexField_t*, 3> fields = {
            &this->omega_x_hat_m, &this->omega_y_hat_m, &this->omega_z_hat_m};
        double sums[2] = {0, 0};
        for (unsigned d = 0; d < 3; ++d) {
            auto q = fields[d]->getView();
            auto ref = reference[d].getView();
            double error = 0, norm = 0;
            Kokkos::parallel_reduce("test_sfsl_modes", ippl::getRangePolicy(q, fields[d]->getNghost()),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& e, double& n) {
                    const auto delta = q(i,j,k) - ref(i,j,k);
                    e += delta.real()*delta.real() + delta.imag()*delta.imag();
                    n += ref(i,j,k).real()*ref(i,j,k).real() + ref(i,j,k).imag()*ref(i,j,k).imag();
                }, error, norm);
            sums[0] += error;
            sums[1] += norm;
        }
        double global[2];
        MPI_Allreduce(sums, global, 2, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
        return std::sqrt(global[0] / global[1]);
    }
};

void check(const char* label, double error, double tolerance = 2e-7) {
    if (ippl::Comm->rank() == 0) {
        std::cout << label << ": error=" << error << std::endl;
    }
    if (!std::isfinite(error) || error > tolerance) {
        throw std::runtime_error(label);
    }
}

#endif
