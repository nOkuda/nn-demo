# https://rix0r.nl/blog/2015/08/13/cmake-guide/
cmake_minimum_required (VERSION 3.2)
project (nn-demo)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 14)

enable_testing()
add_subdirectory(nndemo)
add_subdirectory(app)
