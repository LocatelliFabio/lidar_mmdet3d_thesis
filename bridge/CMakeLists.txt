cmake_minimum_required(VERSION 3.16)
project(rs_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# pybind11
find_package(pybind11 REQUIRED)

# threads
find_package(Threads REQUIRED)

# rs_driver root
set(RS_DRIVER_ROOT "/usr/local/rs_driver")

pybind11_add_module(rs_bridge rs_bridge.cpp)

# include dirs
target_include_directories(rs_bridge PRIVATE
  ${RS_DRIVER_ROOT}/include
)

# libpcap via pkg-config (robusto)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCAP REQUIRED libpcap)

target_include_directories(rs_bridge PRIVATE ${PCAP_INCLUDE_DIRS})
# NB: aggiungo anche "pcap" esplicito per sicurezza
# e Threads + dl (dl spesso serve con plugin/driver)
target_link_libraries(rs_bridge PRIVATE
  ${PCAP_LIBRARIES}
  pcap
  Threads::Threads
  dl
)

# ---- link rs_driver ----
find_library(RS_DRIVER_LIB
  NAMES rs_driver librs_driver
  PATHS ${RS_DRIVER_ROOT}/lib /usr/local/lib /usr/lib
)

if(RS_DRIVER_LIB)
  message(STATUS "Found rs_driver lib: ${RS_DRIVER_LIB}")
  target_link_libraries(rs_bridge PRIVATE ${RS_DRIVER_LIB})
else()
  message(WARNING "rs_driver library not found. If rs_driver is header-only, ignore. Otherwise set RS_DRIVER_LIB manually.")
endif()