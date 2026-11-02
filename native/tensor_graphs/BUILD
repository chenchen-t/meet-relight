# Built inside the LiteRT workspace: third_party/LiteRT/tensor/meet_relight is a
# symlink to this directory (see native/tensor_graphs/build.sh).
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "meet_relight_graphs",
    srcs = ["meet_relight_graphs.cc"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
    ],
)

cc_binary(
    name = "static_batch",
    srcs = ["static_batch.cc"],
    deps = [
        "//tflite/schema:schema_fbs_with_mutable",
        "//tflite/converter/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)
