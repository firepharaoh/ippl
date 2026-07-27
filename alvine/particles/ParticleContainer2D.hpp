#ifndef IPPL_ALVINE_PARTICLE_CONTAINER2D_H
#define IPPL_ALVINE_PARTICLE_CONTAINER2D_H

#include <memory>
#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T>
class ParticleContainer2D : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, 2>>{
public:
    static constexpr unsigned Dim = 2;
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using particle_position_type = typename Base::particle_position_type;

    public:
        particle_position_type P;  
        ippl::ParticleAttrib<T> omega;
        ippl::ParticleAttrib<T> ux;
        ippl::ParticleAttrib<T> uy;
        particle_position_type R_old;
        particle_position_type rk4_R0;
        particle_position_type rk4_k1;
        particle_position_type rk4_k2;
        particle_position_type rk4_k3;
        particle_position_type rk4_k4;
        ippl::ParticleAttrib<T> viscosity;//Viscosity attribute for particles  

    private:
        PLayout_t<T, Dim> pl_m;
    public:
        ParticleContainer2D(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
        }

        ~ParticleContainer2D(){}

        PLayout_t<T, Dim>& getPL() { return pl_m; }
        void setPL(PLayout_t<T, Dim>& pl) { pl_m = pl; }

	void registerAttributes() {
		// register the particle attributes

		this->addAttribute(P);
        this->addAttribute(omega);
        this->addAttribute(ux);
        this->addAttribute(uy);
        this->addAttribute(R_old);
        this->addAttribute(rk4_R0);
        this->addAttribute(rk4_k1);
        this->addAttribute(rk4_k2);
        this->addAttribute(rk4_k3);
        this->addAttribute(rk4_k4);
        this->addAttribute(viscosity); // Register the viscosity attribute
	}
	void setupBCs() { setBCAllPeriodic(); }

    private:
       void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif
