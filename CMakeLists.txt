cmake_minimum_required(VERSION 3.19)

# ipt~ for Pure Data. Consumes only libipt's C ABI (libipt/include/ipt.h);
# libipt (git submodule) owns the inference core, r8brain and libtorch.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
#   cmake --build build -j 8
#
# Output: dist/ipt~/  (external, libipt + torch dylibs, help patch, README).

# macOS, Apple Silicon only for now: libtorch is arm64-only here.
set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "" FORCE)

# Same reason as in libipt: the Command Line Tools SDK ships a libc++ that calls
# __builtin_clzg, which the libtorch 2.4 headers cannot compile with Apple
# clang 16. Pin the sysroot to Xcode's SDK before the toolchain is detected.
if(CMAKE_HOST_APPLE AND NOT DEFINED CMAKE_OSX_SYSROOT)
    execute_process(COMMAND xcode-select -p
                    OUTPUT_VARIABLE _xcode_dev OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(_xcode_sdk "${_xcode_dev}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    if(EXISTS "${_xcode_sdk}/usr/include/c++/v1")
        set(CMAKE_OSX_SYSROOT "${_xcode_sdk}" CACHE PATH "" FORCE)
    endif()
endif()

project(ipt_pd VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT APPLE)
    message(FATAL_ERROR "ipt_pd: only macOS (Apple Silicon) is supported for now")
endif()

# Printed in the ipt~ console banner.
set(PACKAGE_VERSION "${PROJECT_VERSION}")

# --- libipt ----------------------------------------------------------------------------
# Defines ipt_api and exports IPT_TORCH_LIB_DIR to this scope.
add_subdirectory(libipt)

# --- m_pd.h ----------------------------------------------------------------------------
# From the newest /Applications/Pd-*.app unless given with -DPD_INCLUDE_DIR=...
set(PD_INCLUDE_DIR "" CACHE PATH "Directory containing m_pd.h")
if(NOT PD_INCLUDE_DIR)
    file(GLOB _pd_src_dirs "/Applications/Pd-*.app/Contents/Resources/src")
    if(_pd_src_dirs)
        list(SORT _pd_src_dirs COMPARE NATURAL ORDER DESCENDING)
        list(GET _pd_src_dirs 0 _pd_src_dir)
        set(PD_INCLUDE_DIR "${_pd_src_dir}" CACHE PATH "Directory containing m_pd.h" FORCE)
    endif()
endif()
if(NOT EXISTS "${PD_INCLUDE_DIR}/m_pd.h")
    message(FATAL_ERROR
        "m_pd.h not found. Install Pd vanilla in /Applications or pass "
        "-DPD_INCLUDE_DIR=/path/to/pd/src")
endif()
message(STATUS "m_pd.h from ${PD_INCLUDE_DIR}")

# --- ipt~ ------------------------------------------------------------------------------
set(IPT_PD_DIST "${CMAKE_CURRENT_SOURCE_DIR}/dist/ipt~")

add_library(ipt_tilde MODULE src/ipt_tilde.cpp)
target_include_directories(ipt_tilde PRIVATE "${PD_INCLUDE_DIR}")
target_link_libraries(ipt_tilde PRIVATE ipt_api)
target_compile_definitions(ipt_tilde PRIVATE IPT_PACKAGE_VERSION="${PACKAGE_VERSION}")
# Pd's symbols are resolved against the running pd binary at load time.
target_link_options(ipt_tilde PRIVATE -undefined dynamic_lookup)

# .d_arm64: arm64-only external, loaded by Pd 0.57 (which also accepts .d_fat
# and .pd_darwin) and skipped cleanly by an x86_64 Pd.
set_target_properties(ipt_tilde PROPERTIES
    PREFIX ""
    OUTPUT_NAME "ipt~"
    SUFFIX ".d_arm64"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${IPT_PD_DIST}>"   # no per-config subfolder
    CXX_VISIBILITY_PRESET hidden                       # only ipt_tilde_setup is exported
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path")

# Make dist/ipt~ self-contained: libipt.dylib and the three torch dylibs sit
# next to the external, found through @loader_path. The copied libipt.dylib
# loses its absolute build rpath to the torch tree, so the bundle cannot
# silently load torch from this machine's build tree.
set(_torch_dylibs
    "${IPT_TORCH_LIB_DIR}/libtorch.dylib"
    "${IPT_TORCH_LIB_DIR}/libtorch_cpu.dylib"
    "${IPT_TORCH_LIB_DIR}/libc10.dylib")

add_custom_command(TARGET ipt_tilde POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:ipt_api>" ${_torch_dylibs} "${IPT_PD_DIST}/"
    COMMAND install_name_tool -delete_rpath "${IPT_TORCH_LIB_DIR}" "${IPT_PD_DIST}/libipt.dylib"
    COMMAND install_name_tool -add_rpath "@loader_path" "${IPT_PD_DIST}/libipt.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/help/ipt~-help.pd"
            "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
            "${CMAKE_CURRENT_SOURCE_DIR}/libipt/tests/dummy.ts"
            "${IPT_PD_DIST}/"
    COMMAND codesign -s - --force
            "${IPT_PD_DIST}/libipt.dylib"
            "${IPT_PD_DIST}/libtorch.dylib"
            "${IPT_PD_DIST}/libtorch_cpu.dylib"
            "${IPT_PD_DIST}/libc10.dylib"
            "$<TARGET_FILE:ipt_tilde>"
    VERBATIM)
