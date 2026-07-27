#ifndef IPPL_FIELD_CONTAINER_H
#define IPPL_FIELD_CONTAINER_H

#include "FieldContainer2D.hpp"
#include "FieldContainer3D.hpp"

template <typename T, unsigned Dim>
class FieldContainer;

template <typename T>
class FieldContainer<T, 2> : public FieldContainer2D<T> {
public:
    using FieldContainer2D<T>::FieldContainer2D;
};

template <typename T>
class FieldContainer<T, 3> : public FieldContainer3D<T> {
public:
    using FieldContainer3D<T>::FieldContainer3D;
};

#endif
