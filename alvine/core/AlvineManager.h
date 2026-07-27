#ifndef IPPL_ALVINE_CORE_ALVINE_MANAGER_H
#define IPPL_ALVINE_CORE_ALVINE_MANAGER_H

#include "AlvineManager2D.hpp"
#include "AlvineManager3D.hpp"

template <typename T, unsigned Dim>
class AlvineManager;

template <typename T>
class AlvineManager<T, 2> : public AlvineManager2D<T> {
public:
    using AlvineManager2D<T>::AlvineManager2D;
};

template <typename T>
class AlvineManager<T, 3> : public AlvineManager3D<T> {
public:
    using AlvineManager3D<T>::AlvineManager3D;
};

#endif
