#ifndef IPPL_ALVINE_TEST_CASES_TESTCASESELECTION_HPP
#define IPPL_ALVINE_TEST_CASES_TESTCASESELECTION_HPP

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

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
    return name;
}

inline bool isSupportedTestCase(const std::string& name) {
    return normalizeTestCaseName(name) == "taylor_green_2d";
}

template <typename T, unsigned Dim>
Vector_t<T, Dim> domainMinForTestCase(const std::string& name) {
    const std::string normalizedName = normalizeTestCaseName(name);
    if (normalizedName == "taylor_green_2d") {
        static_assert(Dim == 2, "Taylor-Green 2D requires Dim == 2");
        return TaylorGreen2D<T>::domainMin();
    }
    throw std::runtime_error("Unsupported Alvine test case: " + name);
}

template <typename T, unsigned Dim>
Vector_t<T, Dim> domainMaxForTestCase(const std::string& name) {
    const std::string normalizedName = normalizeTestCaseName(name);
    if (normalizedName == "taylor_green_2d") {
        static_assert(Dim == 2, "Taylor-Green 2D requires Dim == 2");
        return TaylorGreen2D<T>::domainMax();
    }
    throw std::runtime_error("Unsupported Alvine test case: " + name);
}

} // namespace alvine

#endif
