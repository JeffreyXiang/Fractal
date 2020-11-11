cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Fractal VERSION 1.0)
message(STATUS "[Project] ${PROJECT_NAME} ${PROJECT_VERSION}")

include(cmake/XCMake.cmake)

Xi_projectInit()

Xi_findPackage(GLFW3)
Xi_findPackage(OpenGL)
enable_language(CUDA)

Xi_addAllSubDir(src)

