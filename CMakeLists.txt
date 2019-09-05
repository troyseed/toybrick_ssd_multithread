cmake_minimum_required(VERSION 2.8)

project(ssd_demo)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

if(${OpenCV_VERSION_MAJOR} MATCHES 2)
	set(LIB_OPENCV_VIDEO "opencv_video")
else()
	set(LIB_OPENCV_VIDEO "opencv_videoio")
endif()

link_directories(/usr/local/lib)
set(link_libs	opencv_core
		opencv_highgui
		opencv_imgproc
		${LIB_OPENCV_VIDEO}
		rknn_api
		pthread
)

include_directories(${CMAKE_SOURCE_DIR}/common /usr/local/include)
aux_source_directory(${CMAKE_SOURCE_DIR}/common common_file)

add_executable(ssd_demo  ssd_demo.cpp ${common_file})
target_link_libraries(ssd_demo ${link_libs})
