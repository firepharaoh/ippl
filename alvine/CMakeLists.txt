file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

find_package(Heffte CONFIG REQUIRED)
set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (VortexInCell executables/VortexInCell.cpp)
add_executable (VortexInFourier executables/VortexInFourier.cpp)
add_executable (VortexInFourier3D executables/VortexInFourier3D.cpp)
add_executable(Fsl executables/Fsl.cpp)
add_executable(SpectralFSL executables/SpectralFSL.cpp)
add_executable(SpectralFSL3D executables/SpectralFSL3D.cpp)
add_executable(PerfectInviscid2D executables/PerfectInviscid2D.cpp)
target_include_directories(VortexInCell PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(VortexInCell ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

target_include_directories(Fsl PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(Fsl ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

target_include_directories(SpectralFSL PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(SpectralFSL ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

target_include_directories(SpectralFSL3D PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(SpectralFSL3D ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

if(IPPL_ENABLE_TESTS)
    add_executable(VIFEuler3DTest tests/VIFEuler3DTest.cpp)
    target_include_directories(VIFEuler3DTest PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine)
    target_link_libraries(VIFEuler3DTest ${IPPL_LIBS} MPI::MPI_CXX Heffte::Heffte)
    add_executable(SFSLEuler3DTest tests/SFSLEuler3DTest.cpp)
    add_executable(SFSLStrang3DTest tests/SFSLStrang3DTest.cpp)
    target_include_directories(SFSLStrang3DTest PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine)
    target_link_libraries(SFSLStrang3DTest ${IPPL_LIBS} MPI::MPI_CXX Heffte::Heffte)
    target_include_directories(SFSLEuler3DTest PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine)
    target_link_libraries(SFSLEuler3DTest ${IPPL_LIBS} MPI::MPI_CXX Heffte::Heffte)
    foreach(ranks IN ITEMS 1 2)
        add_test(NAME vif_euler_3d_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:VIFEuler3DTest> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(vif_euler_3d_${ranks}_ranks PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 180 LABELS "alvine;vif;euler")
        add_test(NAME sfsl_euler_3d_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:SFSLEuler3DTest> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(sfsl_euler_3d_${ranks}_ranks PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 180 LABELS "alvine;sfsl;euler")
        add_test(NAME sfsl_strang_3d_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:SFSLStrang3DTest> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(sfsl_strang_3d_${ranks}_ranks PROPERTIES
            PROCESSORS ${ranks} TIMEOUT 600 LABELS "alvine;sfsl;strang")
    endforeach()
endif()

target_include_directories(VortexInFourier PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(VortexInFourier ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

target_include_directories(VortexInFourier3D PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(VortexInFourier3D ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

target_include_directories(PerfectInviscid2D PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/alvine )
target_link_libraries(PerfectInviscid2D ${IPPL_LIBS} ippl MPI::MPI_CXX Heffte::Heffte)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
