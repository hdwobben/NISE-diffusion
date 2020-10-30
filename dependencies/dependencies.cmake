include(FetchContent)

#---------------------------------------------------------------------------------------
# eigen dependency
#---------------------------------------------------------------------------------------

#find_package (Eigen3 CONFIG)
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
