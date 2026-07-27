#ifndef IPPL_ALVINE_PARTICLES_ALVINESCATTERCIC_HPP
#define IPPL_ALVINE_PARTICLES_ALVINESCATTERCIC_HPP

    void scatterCIC() {
        Inform m("scatter ");

        this->fcontainer_m->getOmegaField() = 0.0;

        if constexpr (Dim == 2) {
            // Scatter particle strengths to grid
            scatter(this->pcontainer_m->omega, this->fcontainer_m->getOmegaField(),
                    this->pcontainer_m->R);

            // Convert deposited circulation to vorticity density
            this->fcontainer_m->getOmegaField() =
                this->fcontainer_m->getOmegaField() / (hr_m[0] * hr_m[1]);

            // Conservation check
            double gammaParticles = computeParticleCirculation();
            double gammaGrid      = computeGridCirculation();

            const double absError = std::fabs(gammaParticles - gammaGrid);
            const double circulationScale =
                std::max(std::max(std::fabs(gammaParticles), std::fabs(gammaGrid)), 1.0);
            const double relError = absError / circulationScale;

            m << "particle circulation = " << gammaParticles
              << ", grid circulation = " << gammaGrid << ", absError = " << absError
              << ", relError = " << relError << endl;

            checkCirculationConservation(absError, relError, m);

        } else if constexpr (Dim == 3) {
            // TODO 3D version
        }
    }

#endif
