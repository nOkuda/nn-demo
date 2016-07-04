project(nndemo)

add_library(nndemo
    src/Data.cpp
    src/FileReader.cpp
    src/NeuralNet.cpp)

target_include_directories(nndemo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

install(TARGETS nndemo
    ARCHIVE DESTINATION nndemo
    LIBRARY DESTINATION nndemo
    RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

export(TARGETS nndemo FILE MyLibraryConfig.cmake)

add_executable(testlib
    test/testlib.cpp)

target_link_libraries(testlib
    nndemo)

add_test(testlib testlib)
