load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_library(
    name = "shared",
    srcs = ["shared.cc"],
    hdrs = ["shared.h"],
    visibility = [":__subpackages__"],
    deps = [
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@local_config_cuda//cuda",
        "@mkldnn",
    ],
)

cuda_library(
    name = "shared_cuda",
    srcs = ["shared.cu"],
    hdrs = ["shared.cuh"],
    visibility = [":__subpackages__"],
    deps = [
        ":shared",
        "@local_config_cuda//cuda",
    ],
)


cc_library(
    name = "segmented_line",
    srcs = [
        "segmented_line.cc",
    ],
    hdrs = ["segmented_line.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":shared",
        ":shared_cuda",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cuda_library(
    name = "upolytope_cuda",
    srcs = ["upolytope.cu"],
    hdrs = ["upolytope.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        "@tbb",
    ],
)

cc_library(
    name = "upolytope",
    srcs = ["upolytope.cc"],
    hdrs = ["upolytope.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":upolytope_cuda",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        "@tbb",
    ],
)


cc_library(
    name = "transformer",
    srcs = ["transformer.cc"],
    hdrs = ["transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":upolytope",
    ],
)

cuda_library(
    name = "pwl_transformer_cuda",
    srcs = ["pwl_transformer.cu"],
    hdrs = ["pwl_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@tbb",
    ],
)

cc_library(
    name = "pwl_transformer",
    srcs = ["pwl_transformer.cc"],
    hdrs = ["pwl_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":pwl_transformer_cuda",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
        "@tbb",
    ],
)

cuda_library(
    name = "relu_transformer_cuda",
    srcs = ["relu_transformer.cu"],
    hdrs = ["relu_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@local_config_cuda//cuda",
    ],
)

cc_library(
    name = "relu_transformer",
    srcs = ["relu_transformer.cc"],
    hdrs = ["relu_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":relu_transformer_cuda",
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
        "@local_config_cuda//cuda",
    ],
)

cuda_library(
    name = "argmax_transformer_cuda",
    srcs = ["argmax_transformer.cu"],
    hdrs = ["argmax_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "argmax_transformer",
    srcs = ["argmax_transformer.cc"],
    hdrs = ["argmax_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":argmax_transformer_cuda",
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "hard_tanh_transformer",
    srcs = ["hard_tanh_transformer.cc"],
    hdrs = ["hard_tanh_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "affine_transformer",
    srcs = ["affine_transformer.cc"],
    hdrs = ["affine_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cuda_library(
    name = "fullyconnected_transformer_cuda",
    srcs = ["fullyconnected_transformer.cu"],
    hdrs = ["fullyconnected_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "fullyconnected_transformer",
    srcs = ["fullyconnected_transformer.cc"],
    hdrs = ["fullyconnected_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":fullyconnected_transformer_cuda",
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cuda_library(
    name = "normalize_transformer_cuda",
    srcs = ["normalize_transformer.cu"],
    hdrs = ["normalize_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "normalize_transformer",
    srcs = ["normalize_transformer.cc"],
    hdrs = ["normalize_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":normalize_transformer_cuda",
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "strided_window_data",
    srcs = ["strided_window_data.cc"],
    hdrs = ["strided_window_data.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":shared",
        ":shared_cuda",
        "//syrenn_proto:syrenn_cc_grpc",
        "@mkldnn",
    ],
)

cuda_library(
    name = "conv2d_transformer_cuda",
    srcs = ["conv2d_transformer.cu"],
    hdrs = ["conv2d_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":strided_window_data",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@local_config_cuda//cuda",
    ],
)

cc_library(
    name = "conv2d_transformer",
    srcs = ["conv2d_transformer.cc"],
    hdrs = ["conv2d_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":conv2d_transformer_cuda",
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":strided_window_data",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
        "@local_config_cuda//cuda",
    ],
)

cc_library(
    name = "maxpool_transformer",
    srcs = ["maxpool_transformer.cc"],
    hdrs = ["maxpool_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":strided_window_data",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
        "@local_config_cuda//cuda",
    ],
)

cuda_library(
    name = "relu_maxpool_transformer_cuda",
    srcs = ["relu_maxpool_transformer.cu"],
    hdrs = ["relu_maxpool_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":maxpool_transformer",
        ":relu_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@local_config_cuda//cuda",
    ],
)

cc_library(
    name = "relu_maxpool_transformer",
    srcs = ["relu_maxpool_transformer.cc"],
    hdrs = ["relu_maxpool_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":relu_maxpool_transformer_cuda",
        ":maxpool_transformer",
        ":relu_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
        "@local_config_cuda//cuda",
    ],
)

cc_library(
    name = "averagepool_transformer",
    srcs = ["averagepool_transformer.cc"],
    hdrs = ["averagepool_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":strided_window_data",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
        "@local_config_cuda//cuda",
    ],
)

cc_library(
    name = "concat_transformer",
    srcs = ["concat_transformer.cc"],
    hdrs = ["concat_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":conv2d_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":transformer",
        "@eigen",
    ],
)

cc_binary(
    name = "server",
    srcs = ["server.cc"],
    copts = ["-Iexternal/gtest/include"],
    visibility = [":__subpackages__"],
    deps = [
        ":argmax_transformer",
        # ":averagepool_transformer",
        # ":concat_transformer",
        ":conv2d_transformer",
        ":fullyconnected_transformer",
        # ":hard_tanh_transformer",
        # ":maxpool_transformer",
        ":normalize_transformer",
        # ":relu_maxpool_transformer",
        ":relu_transformer",
        ":segmented_line",
        ":shared",
        ":shared_cuda",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
    ],
)
