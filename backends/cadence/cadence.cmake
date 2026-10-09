# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(XTENSA_TOOLCHAIN_PATH $ENV{XTENSA_TOOLCHAIN})

if(NOT EXISTS ${XTENSA_TOOLCHAIN_PATH})
  message(
    FATAL_ERROR
      "Nothing found at XTENSA_TOOLCHAIN_PATH: '${XTENSA_TOOLCHAIN_PATH}'"
  )
endif()

set(TOOLCHAIN_HOME ${XTENSA_TOOLCHAIN_PATH}/$ENV{TOOLCHAIN_VER}/XtensaTools)

set(LINKER ld)
set(BINTOOLS gnu)

set(CROSS_COMPILE_TARGET xt)
set(SYSROOT_TARGET xtensa-elf)

set(CROSS_COMPILE ${TOOLCHAIN_HOME}/bin/${CROSS_COMPILE_TARGET}-)
set(SYSROOT_DIR ${TOOLCHAIN_HOME}/${SYSROOT_TARGET})

set(NOSYSDEF_CFLAG "")

list(APPEND TOOLCHAIN_C_FLAGS -fms-extensions)

set(TOOLCHAIN_HAS_NEWLIB
    OFF
    CACHE BOOL "True if toolchain supports newlib"
)

set(COMPILER xt-clang)
# set(CC clang) set(C++ clang++)
set(LINKER xt-ld)

set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_C_COMPILER ${TOOLCHAIN_HOME}/bin/${CROSS_COMPILE_TARGET}-clang)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_HOME}/bin/${CROSS_COMPILE_TARGET}-clang++)

set(CMAKE_C_FLAGS_INIT
    "-mtext-section-literals -mlongcalls -DET_ENABLE_ENUM_STRINGS=0 -ffunction-sections -fdata-sections -fsigned-char -INLINE:requested -fmessage-length=0 -fno-zero-initialized-in-bss -fno-unwind-tables -fno-asynchronous-unwind-tables"
)
set(CMAKE_CXX_FLAGS_INIT
    "${CMAKE_C_FLAGS_INIT} -stdlib=libc++ -fno-strict-aliasing -fno-exceptions -fno-rtti"
)

set(XTENSA_DEBUG_FLAGS "-g -Og -DDEBUG")
set(XTENSA_RELEASE_FLAGS "-O3 -DNDEBUG -mcoproc -LNO:simd")

set(CMAKE_C_FLAGS_DEBUG
    "${XTENSA_DEBUG_FLAGS}"
    CACHE STRING "Xtensa C debug flags" FORCE
)
set(CMAKE_CXX_FLAGS_DEBUG
    "${XTENSA_DEBUG_FLAGS}"
    CACHE STRING "Xtensa CXX debug flags" FORCE
)

set(CMAKE_C_FLAGS_RELEASE
    "${XTENSA_RELEASE_FLAGS}"
    CACHE STRING "Xtensa C release flags" FORCE
)
set(CMAKE_CXX_FLAGS_RELEASE
    "${XTENSA_RELEASE_FLAGS}"
    CACHE STRING "Xtensa CXX release flags" FORCE
)

set(CMAKE_SYSROOT ${TOOLCHAIN_HOME}/${SYSROOT_TARGET})
set(CMAKE_LINKER ${TOOLCHAIN_HOME}/bin/xt-ld)
add_link_options(
  -lm -stdlib=libc++ -Wl,--no-as-needed -static -Wl,--gc-sections
)
message(STATUS "Found toolchain: xt-clang (${XTENSA_TOOLCHAIN_PATH})")

function(Lock_Xtensa_Flags variable access value current_list_file stack)
  if(access STREQUAL "MODIFIED_ACCESS")
    if("${variable}" MATCHES "_RELEASE$")
      set(expected_value "${XTENSA_RELEASE_FLAGS}")
    elseif("${variable}" MATCHES "_DEBUG$")
      set(expected_value "${XTENSA_DEBUG_FLAGS}")
    endif()

    if(DEFINED expected_value AND NOT value STREQUAL expected_value)
      set(${variable}
          "${expected_value}"
          PARENT_SCOPE
      )
    endif()
  endif()
endfunction()

variable_watch(CMAKE_C_FLAGS_RELEASE Lock_Xtensa_Flags)
variable_watch(CMAKE_CXX_FLAGS_RELEASE Lock_Xtensa_Flags)
variable_watch(CMAKE_C_FLAGS_DEBUG Lock_Xtensa_Flags)
variable_watch(CMAKE_CXX_FLAGS_DEBUG Lock_Xtensa_Flags)
