cmake_minimum_required(VERSION 3.15...3.27)

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_library(PCAP_LIB pcap REQUIRED)

# --- include vendored rs_driver headers ---
set(RS_DRIVER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/rs_driver)
if(NOT EXISTS "${RS_DRIVER_ROOT}/src")
  message(FATAL_ERROR "rs_driver submodule missing. Run: git submodule update --init --recursive")
endif()
include_directories(${RS_DRIVER_ROOT}/src)

# --- build pybind module ---
pybind11_add_module(rs_driver src/main.cpp)

# --- link dependencies ---
target_link_libraries(rs_driver PRIVATE Threads::Threads ${PCAP_LIB})

install(TARGETS rs_driver LIBRARY DESTINATION .)
