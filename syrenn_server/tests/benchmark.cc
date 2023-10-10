#include "gtest/gtest.h"
#include "syrenn_proto/syrenn.grpc.pb.h"
// #include "syrenn_server/conv2d_transformer.h"
// #include "syrenn_server/strided_window_data.h"
#include "syrenn_server/fullyconnected_transformer.h"
#include "syrenn_server/pwl_transformer.h"
#include "syrenn_server/relu_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/tests/helpers.h"

TEST(Benchmark, matmul) {
  CUDAShared::init();
  RMMatrixXfDevice a(100000, 200000);
  RMMatrixXfDevice b(200000, 300000);
  a *= b;
}
// DEBUG(zhetao): std::srand(9), 1024, { 512, 512, 512, 256 }
// DEBUG(zhetao): std::srand(9), 1024, { 512, 512, 152, 256 }
// TEST(Benchmark, TransfromPlaneDevice) {

//   std::srand(9);
//   const size_t n_points = 1024;
//   std::vector<size_t> dims = { 512, 512, 512, 256 };

//   UPolytope inout = CreateUPolytopeCircleDevice(n_points, dims[0], 5000.);

//   std::vector<std::unique_ptr<FullyConnectedTransformer>> fc;
//   ReLUTransformer relu;

//   for (int i = 0; i < dims.size() - 1; i++) {
//     fc.emplace_back(RandomFC(dims[i], dims[i+1]));
//   }

//   Timer t0; t0.Reset();
//   Timer t; t.Reset();
//   for (int i = 0; i < dims.size() - 1; i++) {
//     fc[i]->TransformUPolytope(&inout);
//     std::cerr << "FC   " << i << ": " << t.Ticks() << std::endl; t.Reset();
//     relu.TransformUPolytope(&inout);
//     std::cerr << "ReLU " << i << ": " << t.Ticks() << std::endl; t.Reset();
//     std::cerr << "n vertices: " << inout.n_vertices() << std::endl;
//     std::cerr << "n faces: " << inout.n_polytopes() << std::endl;
//     std::cerr << "n edges: " << inout.n_edges() << std::endl;
//     std::cerr << "---------------------" << std::endl;
//   }
//   std::cerr << "Total: " << t0.Ticks() << " ms\n";
//   std::cerr << "---------------------" << std::endl;
// }
