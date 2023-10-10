#include "syrenn_server/pwl_transformer.h"
#include <memory>
#include <vector>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

thrust::device_vector<int> PWLTransformer::point_sign_device(2000000);
thrust::device_vector<size_t> PWLTransformer::todo_edge_idxes(4000000);
thrust::device_vector<size_t> PWLTransformer::result_vertex_idxes(8192);

std::vector<int> PWLTransformer::point_sign(2000000);
std::vector<size_t> PWLTransformer::intersected_edge_idxes(8192);
std::vector<size_t> PWLTransformer::intersect_vertex_idxes(8192);
std::vector<int> PWLTransformer::intersected_faces(16384);

void PWLTransformer::ReserveMemory() {
    PWLTransformer::point_sign_device.resize(2000000);
    PWLTransformer::todo_edge_idxes.resize(4000000);
    PWLTransformer::result_vertex_idxes.resize(8192);
}

void PWLTransformer::ReleaseMemory() {
    PWLTransformer::point_sign_device.clear();
    PWLTransformer::point_sign_device.shrink_to_fit();
    PWLTransformer::todo_edge_idxes.clear();
    PWLTransformer::todo_edge_idxes.shrink_to_fit();
    PWLTransformer::result_vertex_idxes.clear();
    PWLTransformer::result_vertex_idxes.shrink_to_fit();
}
