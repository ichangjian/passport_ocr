# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC ")


get_filename_component(OCR_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
MESSAGE(STATUS ${OCR_CMAKE_DIR})
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "current platform: Linux ")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(DL_api_shared 
    ${OCR_CMAKE_DIR}/Thirdparty/DeepLearning/libs/Linux/libDL_api_shared.so
    #  ${OCR_CMAKE_DIR}/Thirdparty/DeepLearning/libs/Linux/libiomp5.so
    )
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    MESSAGE(STATUS "current platform: Windows")
    add_definitions(-DAPI_EXPORT_DLL)
    set(DL_api_shared ${OCR_CMAKE_DIR}/Thirdparty/DeepLearning/libs/windows/libDL_api_shared.so)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Android")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4")
    set(DL_api_shared ${OCR_CMAKE_DIR}/Thirdparty/DeepLearning/libs/${ANDROID_ABI}/libDL_api_shared.so log)
	MESSAGE(STATUS "current platform: Android ${ANDROID_ABI}")
ELSE ()
	MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

include_directories(${OCR_CMAKE_DIR}/Thirdparty/DeepLearning/include)
include_directories(include)

message(STATUS "opencv dir: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(src SOURCES)


add_library(pp_ocr  ${SOURCES})
target_link_libraries(pp_ocr ${DL_api_shared} ${OpenCV_LIBS} )

add_executable(runner ./test.cpp )
target_link_libraries(runner pp_ocr)

