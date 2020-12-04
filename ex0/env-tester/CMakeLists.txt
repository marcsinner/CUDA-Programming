cmake_minimum_required(VERSION 3.5)
project(env-tester)

set(CMAKE_CXX_STANDARD 11)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

find_package(CUDA REQUIRED)


find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    message(STATUS "PkgConfig Found")
    pkg_check_modules(GMock QUIET gmock)
    if (GMock_FOUND)
        message(STATUS "GMock Found. Testing will be enabled")
    else()
        message(WARNING "cannot find GMock (Google Mock and Google Test)."
                " Note, Testing will be disabled for all worksheets!")
    endif()

else()
    message(WARNING "cannot find PkgConfig."
            " Therefore, GMock (Google Mock and Google Test) cannot be found."
            " Note, Testing will be disabled for all worksheets!")

    message(WARNING "cannot find PkgConfig."
            " Therefore, libjpeg cannot be found."
            " Note, libjpeg is required for WS_0")
endif()


pkg_check_modules(JPEG QUIET libjpeg)
if (JPEG_FOUND)
    message(STATUS "libjpeg is found")
else()
    message(WARNING "libjpeg cannot be found. Note, libjpeg is required for WS_0")
endif()


find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP C++ is found.")
else()
    message(WARNING "OpenMP C++ is not found. Note, OpenMP C++ is required for WS_0")
endif()


find_package(pybind11 QUIET)
if (pybind11_FOUND)
    message(STATUS "pybind11 is found. Python bindings will be enabled")
else()
    message(WARNING "cannot find pybind11. Python bindings will be disabled for all worksheets!")
endif()


add_executable(${CMAKE_PROJECT_NAME} src/main.cpp)
