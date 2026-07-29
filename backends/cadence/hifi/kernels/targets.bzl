load("@fbsource//tools/build_defs:platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    common_deps = [
        "//executorch/runtime/kernel:kernel_includes",
    ]

    hifi_core = read_config("cadence", "hifi_core", "")

    dynamic_exported_deps = []
    if hifi_core == "hifi5":
        dynamic_exported_deps.append(
            "fbsource//third-party/nnlib-hifi5/xa_nnlib:libxa_nnlib_common"
        )
    else:
        dynamic_exported_deps.append(
            "fbsource//third-party/nnlib-hifi4/xa_nnlib:libxa_nnlib_common"
        )

    runtime.cxx_library(
        name = "kernels",
        srcs = ["kernels.cpp"],
        exported_headers = [
            "kernels.h",
        ],
        deps = common_deps,
        compatible_with = ["ovr_config//cpu:xtensa"],
        visibility = ["PUBLIC"],
        exported_deps = dynamic_exported_deps,
        platforms = CXX,
    )
