// Meet-Relight segmentation pre/post-processing graphs, authored with the LiteRT
// Tensor API and serialized to plain .tflite files (integration pattern 1:
// "author a model and ship the .tflite"). The browser runs them with LiteRT.js
// on WebGPU around MediaPipe's selfie_multiclass_256x256.tflite, chaining GPU
// buffers end to end — no CPU readback.
//
//   frame  [1,H,W,4] f32 RGBA in [0,1]   (written by our WGSL from the camera)
//     └─ seg_pre.tflite  → normalized_image [1,256,256,3] in [-1,1]
//          └─ selfie_multiclass_256x256.tflite → logits [1,256,256,6]
//               └─ seg_post.tflite → mask [1,256,256,4] soft
//                    r = person (1 − p_bg), g = skin (body+face), b = face skin, a = hair
//
// Usage:
//   bazel run //tensor/meet_relight:meet_relight_graphs -- --out_dir=/abs/path --h=288 --w=512

#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

ABSL_FLAG(std::string, out_dir, "/tmp", "Directory for the generated .tflite files.");
ABSL_FLAG(int, h, 288, "Height of the frame buffer written by the renderer.");
ABSL_FLAG(int, w, 512, "Width of the frame buffer written by the renderer.");

namespace {

using namespace litert::tensor;  // NOLINT
using T = Tensor<TfLiteMixinTag>;

constexpr int kModelSize = 256;
constexpr int kClasses = 6;  // bg, hair, body-skin, face-skin, clothes, others

absl::Status WritePreprocess(int h, int w, const std::string& path) {
  T frame({.name = "frame", .type = Type::kFP32, .shape = {1, h, w, 4}});
  T rgb = Slice(frame, {0, 0, 0, 0}, {1, h, w, 3});
  // MediaPipe stretches the full frame to the square model input.
  T resized = ResizeBilinear(rgb, {kModelSize, kModelSize});
  T normalized = Add(Mul(resized, 2.0f), -1.0f);
  normalized.SetName("normalized_image");

  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature({normalized}, "seg_pre"));
  return factory.Save(path);
}

absl::Status WritePostprocess(const std::string& path) {
  T logits({.name = "logits", .type = Type::kFP32, .shape = {1, kModelSize, kModelSize, kClasses}});
  T probs = Softmax(logits, 1.0f);
  // Class → channel mixing as one FullyConnected (weights are [out=4, in=6]):
  //   person = Σ p_1..5 (= 1 − p_bg), skin = p_body + p_face, face = p_face, hair = p_hair.
  // clang-format off
  T mix({.type = Type::kFP32, .shape = {4, kClasses}, .buffer = std::vector<float>{
      0, 1, 1, 1, 1, 1,   // person
      0, 0, 1, 1, 0, 0,   // skin
      0, 0, 0, 1, 0, 0,   // face skin
      0, 1, 0, 0, 0, 0,   // hair
  }});
  // clang-format on
  T mask = FullyConnected(probs, mix);  // keep_num_dims → [1,256,256,4]
  mask.SetName("mask");

  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature({mask}, "seg_post"));
  return factory.Save(path);
}

// ---------------------------------------------------------------------------
// Face pipeline (MediaPipe face_detector + face_landmarks_detector)
//
//   frame [1,H,W,4] ─face_det_pre→ [1,128,128,3] (letterboxed, [-1,1])
//     ─face_detector.tflite→ regressors [1,896,16], classificators [1,896,1]
//       ─face_det_post→ det [1,17] = best box (cx,cy,w,h), 6 keypoints, score
//                       (letterboxed-square normalized coords; top-1, no NMS —
//                        a video call has one primary face)
//   ROI warp (WGSL) → [1,256,256,3] ─face_landmarks_detector.tflite→
//     landmarks [1,1,1,1434], presence [1,1,1,1]
//       ─face_lm_post(roi)→ face [1,1435] = 478 × (x/W, y/H, z/W) + presence prob
// ---------------------------------------------------------------------------

constexpr int kDetSize = 128;
constexpr int kAnchors = 896;
constexpr int kLmSize = 256;
constexpr int kLandmarks = 478;

// Concatenate along axis 1.
T Cat(std::vector<T> v) { return Concatenation(absl::MakeSpan(v), 1); }

// BlazeFace short-range SSD anchors: strides {8,16,16,16} on 128², fixed size,
// 2 anchors / cell at stride 8 (16×16) and 6 / cell at stride 16 (8×8).
// Returns a [896,16] table: (ax, ay, 0, 0, ax, ay × 6) so decoding is one Add.
std::vector<float> AnchorTable() {
  std::vector<float> t;
  t.reserve(kAnchors * 16);
  auto emit = [&](int grid, int per_cell) {
    for (int y = 0; y < grid; ++y)
      for (int x = 0; x < grid; ++x)
        for (int a = 0; a < per_cell; ++a) {
          const float ax = (x + 0.5f) / grid, ay = (y + 0.5f) / grid;
          t.insert(t.end(), {ax, ay, 0.f, 0.f});
          for (int k = 0; k < 6; ++k) t.insert(t.end(), {ax, ay});
        }
  };
  emit(16, 2);
  emit(8, 6);
  return t;
}

absl::Status WriteFaceDetPre(int h, int w, const std::string& path) {
  if (w < h) return absl::InvalidArgumentError("expects landscape frames");
  T frame({.name = "frame", .type = Type::kFP32, .shape = {1, h, w, 4}});
  T rgb = Slice(frame, {0, 0, 0, 0}, {1, h, w, 3});
  const int pad = (w - h) / 2;  // letterbox to a square, like MediaPipe
  T paddings({.type = Type::kI32, .shape = {4, 2},
              .buffer = std::vector<int32_t>{0, 0, pad, w - h - pad, 0, 0, 0, 0}});
  T square = Pad(rgb, paddings);
  T resized = ResizeBilinear(square, {kDetSize, kDetSize});
  T input = Add(Mul(resized, 2.0f), -1.0f);
  input.SetName("det_input");
  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature({input}, "face_det_pre"));
  return factory.Save(path);
}

absl::Status WriteFaceDetPost(const std::string& path) {
  T regressors({.name = "regressors", .type = Type::kFP32, .shape = {1, kAnchors, 16}});
  T classificators({.name = "classificators", .type = Type::kFP32, .shape = {1, kAnchors, 1}});
  T anchors({.type = Type::kFP32, .shape = {kAnchors, 16}, .buffer = AnchorTable()});

  T scores = Reshape(classificators, {kAnchors});
  T best = Reshape(ArgMax(scores, 0, Type::kI32), {1});
  T raw = Gather(Reshape(regressors, {kAnchors, 16}), best, 0);  // [1,16]
  T anchor = Gather(anchors, best, 0);                            // [1,16]
  T decoded = Add(Div(raw, static_cast<float>(kDetSize)), anchor);
  T score = Logistic(Reshape(Gather(scores, best, 0), {1, 1}));
  T det = Cat({decoded, score});       // [1,17]
  det.SetName("det");
  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature({det}, "face_det_post"));
  return factory.Save(path);
}

absl::Status WriteFaceLmPost(int h, int w, const std::string& path) {
  T landmarks({.name = "landmarks", .type = Type::kFP32, .shape = {1, 1, 1, kLandmarks * 3}});
  T presence({.name = "presence", .type = Type::kFP32, .shape = {1, 1, 1, 1}});
  // roi = (cx, cy, size, angle) in frame pixels / radians; crop pixel (u,v) ↦ c + R(angle)·(uv−½)·size.
  T roi({.name = "roi", .type = Type::kFP32, .shape = {1, 4}});

  T pts = Reshape(landmarks, {kLandmarks, 3});
  T uv = Add(Div(Slice(pts, {0, 0}, {kLandmarks, 2}), static_cast<float>(kLmSize)), -0.5f);
  T z = Div(Slice(pts, {0, 2}, {kLandmarks, 1}), static_cast<float>(kLmSize));

  T center = Slice(roi, {0, 0}, {1, 2});
  T size = Slice(roi, {0, 2}, {1, 1});
  T angle = Slice(roi, {0, 3}, {1, 1});
  T c = Cos(angle), s = Sin(angle);
  // Row-vector rotation: [u v]·[[c, s], [−s, c]] = (u·c − v·s, u·s + v·c).
  T rot = Reshape(Cat({c, s, Neg(s), c}), {2, 2});
  T pix = Add(Mul(BatchMatMul(uv, rot), size), center);           // [478,2] frame pixels
  T inv_wh({.type = Type::kFP32, .shape = {2}, .buffer = std::vector<float>{1.0f / w, 1.0f / h}});
  T xy = Mul(pix, inv_wh);
  T zn = Mul(z, Div(size, static_cast<float>(w)));                 // MediaPipe: z in x-normalized units
  T flat = Reshape(Cat({xy, zn}), {1, kLandmarks * 3});
  T prob = Logistic(Reshape(presence, {1, 1}));
  T face = Cat({flat, prob});           // [1,1435]
  face.SetName("face");
  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(factory.AddSignature({face}, "face_lm_post"));
  return factory.Save(path);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string dir = absl::GetFlag(FLAGS_out_dir);
  const int h = absl::GetFlag(FLAGS_h);
  const int w = absl::GetFlag(FLAGS_w);

  struct Job { const char* name; absl::Status status; };
  const Job jobs[] = {
      {"seg_pre", WritePreprocess(h, w, dir + "/seg_pre.tflite")},
      {"seg_post", WritePostprocess(dir + "/seg_post.tflite")},
      {"face_det_pre", WriteFaceDetPre(h, w, dir + "/face_det_pre.tflite")},
      {"face_det_post", WriteFaceDetPost(dir + "/face_det_post.tflite")},
      {"face_lm_post", WriteFaceLmPost(h, w, dir + "/face_lm_post.tflite")},
  };
  int failed = 0;
  for (const auto& j : jobs) {
    if (j.status.ok()) {
      std::cout << "wrote " << dir << "/" << j.name << ".tflite\n";
    } else {
      std::cerr << j.name << ": " << j.status << "\n";
      ++failed;
    }
  }
  return failed ? 1 : 0;
}
