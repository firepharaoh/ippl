#ifndef IPPL_ALVINE_TEST_CASES_ALVINETAYLORGREEN2D_HPP
#define IPPL_ALVINE_TEST_CASES_ALVINETAYLORGREEN2D_HPP

    void fillExactTGVVorticity(Field<T, Dim>& omegaField, double time) {
        TaylorGreen2D<T>::fillVorticity(omegaField, this->fcontainer_m->getFL(),
                                        this->rmin_m, this->hr_m, T(time),
                                        T(this->viscosity_m));
    }

    void fillExactTGVVorticity(Field<T, Dim>& omegaField) {
        fillExactTGVVorticity(omegaField, this->time_m);
    }

#endif
