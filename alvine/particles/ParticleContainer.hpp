#ifndef IPPL_ALVINE_PARTICLE_CONTAINER_H
#define IPPL_ALVINE_PARTICLE_CONTAINER_H

#include "ParticleContainer2D.hpp"
#include "ParticleContainer3D.hpp"

template <typename T, unsigned Dim>
class ParticleContainer;

template <typename T>
class ParticleContainer<T, 2> : public ParticleContainer2D<T> {
public:
    using ParticleContainer2D<T>::ParticleContainer2D;
};

template <typename T>
class ParticleContainer<T, 3> : public ParticleContainer3D<T> {
public:
    using ParticleContainer3D<T>::ParticleContainer3D;
};

#endif
