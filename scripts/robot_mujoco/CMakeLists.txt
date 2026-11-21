cmake_minimum_required(VERSION 3.16)
project(robot_mujoco_local LANGUAGES CXX)

# Build the copied upstream simulator against this host's libraries.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${ROBOT_ROOT}/unitree_sdk2" sdk2 EXCLUDE_FROM_ALL)

set(SIM_SOURCE "${ROBOT_ROOT}/unitree_mujoco/simulate/src")
add_executable(unitree_mujoco
    "${SIM_SOURCE}/main.cc"
    "${SIM_SOURCE}/joystick/joystick.cc"
    "${SIM_SOURCE}/lodepng/lodepng.cpp"
    "${MUJOCO_ROOT}/simulate/glfw_adapter.cc"
    "${MUJOCO_ROOT}/simulate/glfw_dispatch.cc"
    "${MUJOCO_ROOT}/simulate/platform_ui_adapter.cc"
    "${MUJOCO_ROOT}/simulate/simulate.cc")
target_include_directories(unitree_mujoco PRIVATE
    "${MUJOCO_ROOT}/include" "${MUJOCO_ROOT}/simulate"
    "${GLFW_INCLUDE_DIR}" "${LOCAL_INCLUDE_DIR}" "${SIM_SOURCE}/lodepng")
target_link_libraries(unitree_mujoco PRIVATE
    "${MUJOCO_ROOT}/lib/libmujoco.so.3.3.6" "${GLFW_LIBRARY}"
    yaml-cpp Boost::program_options fmt::fmt unitree_sdk2 Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(unitree_mujoco PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${MUJOCO_ROOT}/lib;${GLFW_LIBRARY_DIR};${ROBOT_ROOT}/unitree_sdk2/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}")
