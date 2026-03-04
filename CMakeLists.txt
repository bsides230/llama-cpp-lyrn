cmake_minimum_required(VERSION 3.21)
project(llama_cpp)

# --- Install helper: copy built shared libs into both:
#   1) repo-local llama_cpp/lib (useful for local runs)
#   2) the wheel/platlib llama_cpp/lib (for installed package)
function(llama_cpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
        RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib
    )
    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib
    )

    # RPATH so the Python module can load the bundled libs
    if(UNIX)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "@loader_path"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    endif()
endfunction()

# --- Build settings
set(BUILD_SHARED_LIBS ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_SKIP_RPATH FALSE)

# --- llama.cpp build options (keep minimal / no server/tools/tests/examples)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "llama.cpp: build common library" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "llama.cpp: enable curl" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: build examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama.cpp: build server" FORCE)

# --- llama.cpp source resolution
# Preferred: vendored source at vendor/llama.cpp (for deterministic pinned builds)
# Fallback: FetchContent clone (for clean installs where submodules aren't present)
set(LLAMA_CPP_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")

if (EXISTS "${LLAMA_CPP_VENDOR_DIR}/CMakeLists.txt")
    message(STATUS "Using vendored llama.cpp: ${LLAMA_CPP_VENDOR_DIR}")
    add_subdirectory("${LLAMA_CPP_VENDOR_DIR}" llama_cpp_vendor)
else()
    message(STATUS "Vendored llama.cpp not found at vendor/llama.cpp. Fetching ggml-org/llama.cpp...")
    include(FetchContent)
    FetchContent_Declare(
        llama_cpp_upstream
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        # TODO: pin to a known-good commit/tag for determinism
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(llama_cpp_upstream)
    add_subdirectory("${llama_cpp_upstream_SOURCE_DIR}" llama_cpp_vendor)
endif()

# Windows: export symbols for shared libs when needed
if (WIN32)
    if (TARGET llama)
        set_target_properties(llama PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

# Install the primary libs + optional backend libs (no-op if target doesn't exist)
llama_cpp_python_install_target(llama)
llama_cpp_python_install_target(ggml)
llama_cpp_python_install_target(ggml-base)
llama_cpp_python_install_target(ggml-cpu)
llama_cpp_python_install_target(ggml-amx)
llama_cpp_python_install_target(ggml-blas)
llama_cpp_python_install_target(ggml-can)
llama_cpp_python_install_target(ggml-cuda)
llama_cpp_python_install_target(ggml-hip)
llama_cpp_python_install_target(ggml-kompute)
llama_cpp_python_install_target(ggml-metal)
llama_cpp_python_install_target(ggml-musa)
llama_cpp_python_install_target(ggml-rpc)
llama_cpp_python_install_target(ggml-sycl)
llama_cpp_python_install_target(ggml-vulkan)

# Windows: include runtime DLL dependencies for the built targets (when applicable)
if (WIN32)
    if (TARGET llama)
        install(FILES $<TARGET_RUNTIME_DLLS:llama> DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib)
        install(FILES $<TARGET_RUNTIME_DLLS:llama> DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib)
    endif()
    if (TARGET ggml)
        install(FILES $<TARGET_RUNTIME_DLLS:ggml> DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib)
        install(FILES $<TARGET_RUNTIME_DLLS:ggml> DESTINATION ${SKBUILD_PLATLIB_DIR}/llama_cpp/lib)
    endif()
endif()
