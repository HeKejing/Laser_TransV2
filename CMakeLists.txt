cmake_minimum_required(VERSION 2.8)
project( Laser_TransV2 )


#---------------------------------------------------------------------------------------
# add source files
#---------------------------------------------------------------------------------------
include_directories("src")
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
MESSAGE(STATUS ${DIR_SRCS})

FILE(GLOB DIR_HEADERS src/*.h)

#---------------------------------------------------------------------------------------
# OpenCV
#---------------------------------------------------------------------------------------
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#---------------------------------------------------------------------------------------
# Levmar 
#---------------------------------------------------------------------------------------
set(Levmar_INCLUDE_DIRS "C:/Tools_Lib/levmar-2.6")
set(Levmar_LIBS_DIR "C:/Tools_Lib/levmar-2.6/Lib")
set(Clapack_LIBS_DIR "C:/Tools_Lib/clapack/lib")

# find_package( Levmar REQUIRED )

include_directories( ${Levmar_INCLUDE_DIRS} )
link_directories(${Levmar_LIBS_DIR})
link_directories(${Clapack_LIBS_DIR})


#---------------------------------------------------------------------------------------
# daheng
#---------------------------------------------------------------------------------------
set(DAHENG_INCLUDE "${CMAKE_SOURCE_DIR}/3rdParty/windows/daheng/VC SDK/inc")
set(DAHENG_LIBS_DIR "${CMAKE_SOURCE_DIR}/3rdParty/windows/daheng/VC SDK/lib/x64")

include_directories(${DAHENG_INCLUDE})
link_directories(${DAHENG_LIBS_DIR})

#---------------------------------------------------------------------------------------
# OpenMP
#---------------------------------------------------------------------------------------
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


#---------------------------------------------------------------------------------------
# PCL
#---------------------------------------------------------------------------------------
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


#---------------------------------------------------------------------------------------
# Laser_TransV2 
#---------------------------------------------------------------------------------------
add_executable( Laser_TransV2 ${DIR_SRCS} ${DIR_HEADERS} )
target_link_libraries( Laser_TransV2 ${OpenCV_LIBS} )
target_link_libraries( Laser_TransV2 GxIAPI.lib DxImageProc.lib )
target_link_libraries( Laser_TransV2 ${PCL_LIBRARIES} )
target_link_libraries( Laser_TransV2 levmar.lib blas.lib lapack.lib libf2c.lib tmglib.lib)
target_link_libraries( Laser_TransV2 levmard.lib blasd.lib lapackd.lib libf2cd.lib tmglibd.lib)

#---------------------------------------------------------------------------------------
# set groups
#---------------------------------------------------------------------------------------
source_group("Header Files" FILES ${DIR_HEADERS})
