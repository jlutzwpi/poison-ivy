
cmake_minimum_required(VERSION 3.12)

set(MODEL_FOLDER .)
set(EI_SDK_FOLDER edge-impulse-sdk)
 
include(pico_sdk_import.cmake)

project(poison_ivy C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
pico_sdk_init()

add_executable(poison_ivy_int8 "")

include(${MODEL_FOLDER}/edge-impulse-sdk/cmake/utils.cmake)

target_include_directories(poison_ivy_int8
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/.
    ${CMAKE_CURRENT_LIST_DIR}/classifer
    ${CMAKE_CURRENT_LIST_DIR}/tflite-model
    ${CMAKE_CURRENT_LIST_DIR}/model-parameters
    )

set_target_properties(
    poison_ivy_int8
    PROPERTIES
    COMPILE_FLAGS -fno-rtti
    COMPILE_FLAGS -fno-exceptions
    COMPILE_FLAGS -fno-threadsafe-statics
    COMPILE_FLAGS -nostdlib
)

target_link_libraries(
  poison_ivy_int8
  hardware_pwm
  hardware_pio
  pico_stdlib
  arducam_hm01b0
  LCD_st7735

)

add_subdirectory(${MODEL_FOLDER}/edge-impulse-sdk/cmake/zephyr)

target_include_directories(poison_ivy_int8 PRIVATE
    ${EI_SDK_FOLDER}
    ${EI_SDK_FOLDER}/third_party/ruy
    ${EI_SDK_FOLDER}/third_party/gemmlowp
    ${EI_SDK_FOLDER}/third_party/flatbuffers/include
    ${EI_SDK_FOLDER}/third_party
    ${EI_SDK_FOLDER}/tensorflow
    ${EI_SDK_FOLDER}/dsp
    ${EI_SDK_FOLDER}/classifier
    ${EI_SDK_FOLDER}/anomaly
    ${EI_SDK_FOLDER}/CMSIS/NN/Include
    ${EI_SDK_FOLDER}/CMSIS/DSP/PrivateInclude
    ${EI_SDK_FOLDER}/CMSIS/DSP/Include
    ${EI_SDK_FOLDER}/CMSIS/Core/Include
)
 
include_directories(${INCLUDES})
 
# find model source files
RECURSIVE_FIND_FILE(MODEL_FILES "${MODEL_FOLDER}/tflite-model" "*.cpp")
RECURSIVE_FIND_FILE(SOURCE_FILES "${EI_SDK_FOLDER}" "*.cpp")
RECURSIVE_FIND_FILE(CC_FILES "${EI_SDK_FOLDER}" "*.cc")
RECURSIVE_FIND_FILE(S_FILES "${EI_SDK_FOLDER}" "*.s")
RECURSIVE_FIND_FILE(C_FILES "${EI_SDK_FOLDER}" "*.c")
list(APPEND SOURCE_FILES ${S_FILES})
list(APPEND SOURCE_FILES ${C_FILES})
list(APPEND SOURCE_FILES ${CC_FILES})
list(APPEND SOURCE_FILES ${MODEL_FILES})

target_sources(poison_ivy_int8
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/ei_classifier_porting.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection_responder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/image_provider.cpp
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/main_functions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection_responder.h
    ${CMAKE_CURRENT_LIST_DIR}/image_provider.h
    ${CMAKE_CURRENT_LIST_DIR}/main_functions.h
    ${SOURCE_FILES}
    )

pico_enable_stdio_uart(poison_ivy_int8 0)
pico_enable_stdio_usb(poison_ivy_int8 1)
pico_add_extra_outputs(poison_ivy_int8)


