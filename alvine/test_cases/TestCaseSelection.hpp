#ifndef IPPL_ALVINE_TEST_CASES_TESTCASESELECTION_HPP
#define IPPL_ALVINE_TEST_CASES_TESTCASESELECTION_HPP

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "TaylorGreen2D.hpp"
#include "TaylorGreen3D.hpp"

namespace alvine {

inline std::string normalizeTestCaseName(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (char& c : name) {
        if (c == '-' || c == ' ' || c == '.') {
            c = '_';
        }
    }

    if (name == "tgv" || name == "taylor_green" || name == "taylor_green_vortex") {
        return "taylor_green_2d";
    }
    if (name == "tgv3d" || name == "taylor_green_3d" ||
        name == "taylor_green_vortex_3d") {
        return "taylor_green_3d";
    }
    return name;
}

inline bool isSupportedTestCase(const std::string& name) {
    const std::string normalizedName = normalizeTestCaseName(name);
    return normalizedName == "taylor_green_2d" || normalizedName == "taylor_green_3d";
}

template <typename T, unsigned Dim>
Vector_t<T, Dim> domainMinForTestCase(const std::string& name) {
    const std::string normalizedName = normalizeTestCaseName(name);
    if constexpr (Dim == 2) {
        if (normalizedName != "taylor_green_2d") {
            throw std::runtime_error("Unsupported Alvine 2D test case: " + name);
        }
        static_assert(Dim == 2, "Taylor-Green 2D requires Dim == 2");
        return TaylorGreen2D<T>::domainMin();
    }
    if constexpr (Dim == 3) {
        if (normalizedName != "taylor_green_3d") {
            throw std::runtime_error("Unsupported Alvine 3D test case: " + name);
        }
        static_assert(Dim == 3, "Taylor-Green 3D requires Dim == 3");
        return TaylorGreen3D<T>::domainMin();
    }
    throw std::runtime_error("Unsupported Alvine test case: " + name);
}

template <typename T, unsigned Dim>
Vector_t<T, Dim> domainMaxForTestCase(const std::string& name) {
    const std::string normalizedName = normalizeTestCaseName(name);
    if constexpr (Dim == 2) {
        if (normalizedName != "taylor_green_2d") {
            throw std::runtime_error("Unsupported Alvine 2D test case: " + name);
        }
        static_assert(Dim == 2, "Taylor-Green 2D requires Dim == 2");
        return TaylorGreen2D<T>::domainMax();
    }
    if constexpr (Dim == 3) {
        if (normalizedName != "taylor_green_3d") {
            throw std::runtime_error("Unsupported Alvine 3D test case: " + name);
        }
        static_assert(Dim == 3, "Taylor-Green 3D requires Dim == 3");
        return TaylorGreen3D<T>::domainMax();
    }
    throw std::runtime_error("Unsupported Alvine test case: " + name);
}

} // namespace alvine

#endif
