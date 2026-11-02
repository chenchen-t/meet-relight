// Rewrites a .tflite so every tensor uses its static `shape` (drops
// `shape_signature`, i.e. dynamic -1 batch dims) and pins -1 dims to 1.
// LiteRT.js binds GPUBuffer-backed tensors only to exactly-matching static
// types, and MediaPipe's face_landmarks_detector ships with batch = -1.
//
// Usage: static_batch <in.tflite> <out.tflite>

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "tflite/schema/schema_generated.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: static_batch <in.tflite> <out.tflite>\n";
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
  if (bytes.empty()) {
    std::cerr << "cannot read " << argv[1] << "\n";
    return 1;
  }
  std::unique_ptr<tflite::ModelT> model(tflite::GetModel(bytes.data())->UnPack());
  int changed = 0;
  for (auto& sg : model->subgraphs) {
    for (auto& t : sg->tensors) {
      for (int& d : t->shape) {
        if (d < 0) { d = 1; ++changed; }
      }
      if (!t->shape_signature.empty()) {
        t->shape_signature.clear();
        ++changed;
      }
    }
  }
  flatbuffers::FlatBufferBuilder fbb;
  tflite::FinishModelBuffer(fbb, tflite::Model::Pack(fbb, model.get()));
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize());
  std::cout << "wrote " << argv[2] << " (" << changed << " dynamic dims pinned)\n";
  return 0;
}
