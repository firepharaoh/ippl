#ifndef IPPL_ALVINE_PARTICLES_ALVINEPARTICLEGRIDTRANSFER_HPP
#define IPPL_ALVINE_PARTICLES_ALVINEPARTICLEGRIDTRANSFER_HPP

    void grid2par() override {
        gatherCIC();
    }

    void gatherCIC() {
        this->pcontainer_m->P = 0.0;
        gather(this->pcontainer_m->P, this->fcontainer_m->getUField(), this->pcontainer_m->R);
    }

    void par2grid() override {
        scatterCIC();
    }

#endif
