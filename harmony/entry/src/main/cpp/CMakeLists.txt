cmake_minimum_required(VERSION 3.5.0)
project(riskguard_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(riskguard_native SHARED riskguard_native.cpp)

target_link_libraries(riskguard_native PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
    libmindspore_lite_ndk.so
)
