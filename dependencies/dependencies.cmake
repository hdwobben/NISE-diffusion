include(FetchContent)

#---------------------------------------------------------------------------------------
# eigen dependency
#---------------------------------------------------------------------------------------

#find_package (Eigen3 CONFIG QUIET)

if(NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.3.8)
    
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)   # name is lowercased
        FetchContent_Populate(eigen)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    add_library(Eigen3::Eigen ALIAS eigen)
endif()

#---------------------------------------------------------------------------------------
# json dependency
#---------------------------------------------------------------------------------------

find_package(nlohmann_json QUIET)

if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
        GIT_TAG v3.9.1)

    FetchContent_GetProperties(json)
    if(NOT json_POPULATED)
        FetchContent_Populate(json)
    endif()
endif()