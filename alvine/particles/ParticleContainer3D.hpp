#ifndef IPPL_ALVINE_PARTICLE_CONTAINER3D_H
#define IPPL_ALVINE_PARTICLE_CONTAINER3D_H

#include <memory>
#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T>
class ParticleContainer3D : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, 3>>{
public:
    static constexpr unsigned Dim = 3;
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;
    using particle_position_type = typename Base::particle_position_type;

    public:
        particle_position_type P;  
        particle_position_type omega;
        ippl::ParticleAttrib<T> omega_x;
        ippl::ParticleAttrib<T> omega_y;
        ippl::ParticleAttrib<T> omega_z;
        ippl::ParticleAttrib<T> ux;
        ippl::ParticleAttrib<T> uy;
        ippl::ParticleAttrib<T> uz;
        ippl::ParticleAttrib<T> duxdx;
        ippl::ParticleAttrib<T> duxdy;
        ippl::ParticleAttrib<T> duxdz;
        ippl::ParticleAttrib<T> duydx;
        ippl::ParticleAttrib<T> duydy;
        ippl::ParticleAttrib<T> duydz;
        ippl::ParticleAttrib<T> duzdx;
        ippl::ParticleAttrib<T> duzdy;
        ippl::ParticleAttrib<T> duzdz;
        particle_position_type u; // 3D vector velocity storage for component-based or vector gather paths.
        particle_position_type R_old;
        particle_position_type rk4_R0;
        particle_position_type rk4_k1;
        particle_position_type rk4_k2;
        particle_position_type rk4_k3;
        particle_position_type rk4_k4;

        particle_position_type viscosity;//Viscosity attribute for particles  
        ippl::ParticleAttrib<T> viscosity_x;
        ippl::ParticleAttrib<T> viscosity_y;
        ippl::ParticleAttrib<T> viscosity_z;

        particle_position_type stretching_term; // Stretching term attribute for particles (Still not implemented )
    private:
        PLayout_t<T, Dim> pl_m;
    public:
        ParticleContainer3D(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& FL)
        : pl_m(FL, mesh) {
        this->initialize(pl_m);
        registerAttributes();
        setupBCs();
        }

        ~ParticleContainer3D(){}

        PLayout_t<T, Dim>& getPL() { return pl_m; }
        void setPL(PLayout_t<T, Dim>& pl) { pl_m = pl; }

	void registerAttributes() {
		// register the particle attributes

        this->addAttribute(P);
        this->addAttribute(omega);
        this->addAttribute(omega_x);
        this->addAttribute(omega_y);
        this->addAttribute(omega_z);
        this->addAttribute(viscosity_x);
        this->addAttribute(viscosity_y);
        this->addAttribute(viscosity_z);
        this->addAttribute(ux);
        this->addAttribute(uy);
        this->addAttribute(uz);
        this->addAttribute(duxdx);
        this->addAttribute(duxdy);
        this->addAttribute(duxdz);
        this->addAttribute(duydx);
        this->addAttribute(duydy);
        this->addAttribute(duydz);
        this->addAttribute(duzdx);
        this->addAttribute(duzdy);
        this->addAttribute(duzdz);
        this->addAttribute(u);
        this->addAttribute(R_old);
        this->addAttribute(rk4_R0);
        this->addAttribute(rk4_k1);
        this->addAttribute(rk4_k2);
        this->addAttribute(rk4_k3);
        this->addAttribute(rk4_k4);
        this->addAttribute(viscosity); // Register the viscosity attribute
        this->addAttribute(stretching_term);

    }
	void setupBCs() { setBCAllPeriodic(); }

    private:
       void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif
