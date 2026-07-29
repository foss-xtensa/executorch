# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    hifi_core = read_config("cadence", "hifi_core", "")

    dynamic_deps = []
    if hifi_core == "hifi5":
        dynamic_deps.append("fbsource//third-party/nnlib-hifi5/xa_nnlib:libxa_nnlib")
    else:
        dynamic_deps.append("fbsource//third-party/nnlib-hifi4/xa_nnlib:libxa_nnlib")

    runtime.cxx_library(
        name = "nnlib-extensions",
        srcs = native.glob(["*.c", "*.cpp"]),
        exported_headers = glob(["*.h"]),
        visibility = ["PUBLIC"],
        compatible_with = ["ovr_config//cpu:xtensa"],
        compiler_flags = [
            "-Wno-pointer-sign",
            "-Wno-incompatible-pointer-types-discards-qualifiers",
        ],
        deps = dynamic_deps,
    )
