#include "syrenn_server/relu_transformer.h"
#include <memory>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

// TODO(zhetao): https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__
void PointSignKernel(const int n_vertices,
                     const int n_dims,
                     int* results,
                     const float* vertices,
                     const size_t plane_idx) {
  const int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex_idx < n_vertices) {
    if (vertices[vertex_idx * n_dims + plane_idx] == 0) {
      results[vertex_idx] = 0;
    } else {
      results[vertex_idx] = vertices[vertex_idx * n_dims + plane_idx] > 0 ? +1 : -1;
    }
  }
}

__host__
thrust::device_vector<int> & ReLUTransformer::PointSignDevice(
                              const RMMatrixXfDevice &vertices,
                              const size_t plane_idx) const {

  const int n_vertices = vertices.rows();
  const int n_dims = vertices.cols();

  // struct cudaDeviceProp properties;
  // cudaGetDeviceProperties(&properties, 0);
  // std::cerr<<"using "<<properties.multiProcessorCount<<" multiprocessors"<<std::endl;
  // std::cerr<<"max threads per processor: "<<properties.maxThreadsPerMultiProcessor<<std::endl;

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_vertices / threadsPerBlock.x) + 1);
  // assert (nBlocks.x * threadsPerBlock.x >= n_vertices);

  if (PWLTransformer::point_sign_device.capacity() < n_vertices) {
    PWLTransformer::point_sign_device.resize(n_vertices);
    std::cerr << "resize " << n_vertices << " for point_sign_device\n";
  }

#ifdef DEBUG_TRANSFORM
  // std::cerr << "n blocks: " << nBlocks.x << "* n threads per block: " << threadsPerBlock.x << std::endl;
  // std::cerr << "compute signs for n vertices: " << n_vertices << std::endl;
  assert (nBlocks.x * threadsPerBlock.x >= n_vertices);
#endif

  PointSignKernel<<<nBlocks, threadsPerBlock>>>(
    n_vertices, n_dims,
    raw_pointer_cast(PWLTransformer::point_sign_device.data()),
    vertices.data(),
    plane_idx
  );

  return PWLTransformer::point_sign_device;
}

__host__
int ReLUTransformer::PointSign(const RMMatrixXfDevice &vertices,
                               size_t vertex_idx,
                               const size_t face) const {
  float point = vertices.device()[vertex_idx * vertices.cols() + face];
  if (point == 0) {
    return 0;
  }
  return point > 0 ? +1 : -1;
}

// __device__ float ReLUTransformer::CrossingRatio(const size_t n_dims,
//                                const float* vertices,
//                                const size_t from_idx,
//                                const size_t to_idx,
//                                const size_t plane_idx) {
//   const float from = vertices[from_idx * n_dims + plane_idx];
//   const float to = vertices[to_idx * n_dims + plane_idx];
//   const float ratio = -from / (to - from);
//   return ratio;
// }

__global__
void CrossingRatioKernel(float* results,
                         const int n_edges,
                         const float* vertices,
                         const int n_dims,
                         const size_t* from_idxes,
                         const size_t* to_idxes,
                         const size_t plane_idx) {
  const int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_idx < n_edges) {
    const int from_idx = from_idxes[edge_idx * n_dims + plane_idx];
    const int to_idx = to_idxes[edge_idx * n_dims + plane_idx];
    const float from = vertices[from_idx];
    const float to = vertices[to_idx];
    results[edge_idx] = -from / (to - from);
  }
}

// NOTE(zhetao): Not using this anywhere.
std::vector<float> ReLUTransformer::CrossingRatio(
                                      const RMMatrixXfDevice &vertices,
                                      const std::vector<size_t> &from_idxes,
                                      const std::vector<size_t> &to_idxes,
                                      const size_t plane_idx) const {

  assert (false);

  const int n_edges = from_idxes.size();
  const int n_dims = vertices.cols();

  thrust::device_vector<size_t> from_idxes_device(from_idxes.size());
  thrust::copy(from_idxes.begin(), from_idxes.end(), from_idxes_device.begin());
  thrust::device_vector<size_t> to_idxes_device(to_idxes.size());
  thrust::copy(to_idxes.begin(), to_idxes.end(), to_idxes_device.begin());

  thrust::device_vector<float> results_device(n_edges);
  std::vector<float> results(n_edges);

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_edges / threadsPerBlock.x) + 1);
  // assert (nBlocks.x * threadsPerBlock.x >= n_edges);

  CrossingRatioKernel<<<nBlocks, threadsPerBlock>>>(
    raw_pointer_cast(results_device.data()),
    n_edges,
    vertices.data(),
    n_dims,
    raw_pointer_cast(from_idxes_device.data()),
    raw_pointer_cast(to_idxes_device.data()),
    plane_idx);

  thrust::copy(results_device.begin(), results_device.end(), results.begin());
  return results;
}

__host__
double ReLUTransformer::CrossingRatio(const RMMatrixXfDevice &vertices,
                                      const size_t from_idx,
                                      const size_t to_idx,
                                      const size_t face) const {
  // TODO(zhetao): Just a workaround.
  const float from = vertices.device()[from_idx * vertices.cols() + face];
  const float to = vertices.device()[to_idx * vertices.cols() + face];
  return -from / (to - from);
}

void ReLUTransformer::ComputeDevice(RMMatrixXfDevice *inout) const {

  checkCUDNN( cudnnSetTensor4dDescriptor(inout_on_device_desc,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         static_cast<int>(inout->size()),
                                         1, 1, 1) );

  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN( cudnnActivationForward(CUDAShared::cudnn_handle,
                                     activation_descriptor,
                                     &alpha,
                                     inout_on_device_desc,
                                     inout->data(),
                                     &beta,
                                     inout_on_device_desc,
                                     inout->data()) );

  return;
}

__global__
void UpdateEndpointsKernelReLU(
  const size_t n_todos,
  const size_t n_original_vertices,
  const size_t* todo_edge_idxes,
  const size_t* result_vertex_idxes,
  size_t* edge_endpoints
) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_todos && result_vertex_idxes[i] >= n_original_vertices) {
    const size_t edge_idx = todo_edge_idxes[i];
    edge_endpoints[edge_idx*2 + 1] = result_vertex_idxes[i];
  }
}

__global__
void InterpolateVertex(
  const size_t n_todos,
  const size_t* todo_edge_idxes,
  size_t* result_vertex_idxes,
  float* vertices, const size_t n_dims,
  float* combinations, const size_t n_combination_dims,
  const size_t n_original_vertices,
  const size_t* edge_endpoints,
  const int* point_sign,
  const size_t plane_idx
) {
  const size_t todo_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (todo_idx < n_todos) {
    const size_t edge_idx = todo_edge_idxes[todo_idx];
    const size_t v1_idx = edge_endpoints[edge_idx*2];
    const size_t v2_idx = edge_endpoints[edge_idx*2 + 1];
    if (point_sign[v1_idx] == 0) {
      result_vertex_idxes[todo_idx] = v1_idx;
    } else if (point_sign[v2_idx] == 0) {
      result_vertex_idxes[todo_idx] = v2_idx;
    } else {
      // TODO(zhetao): Enforce deterministicity.
      // FIXME(zhetao): CrossRatio for different PWL layers.
      // const float ratio = CrossingRatio(n_dims, vertices, v1_idx, v2_idx, plane_idx);
      const float from = vertices[v1_idx * n_dims + plane_idx];
      const float to = vertices[v2_idx * n_dims + plane_idx];
      const float ratio = -from / (to - from);
      const float ratio2 = 1.0f - ratio;

      const size_t vertex_idx = n_original_vertices + todo_idx;

      // TODO(zhetao): Maybe use cublasSaxpy here.
      for (size_t dim = 0; dim < n_dims; dim++) {
        const float from = vertices[v1_idx * n_dims + dim];
        const float to   = vertices[v2_idx * n_dims + dim];
        vertices[vertex_idx * n_dims + dim] = (ratio2 * from) + (ratio * to);
      }

      for (size_t dim = 0; dim < n_combination_dims; dim++) {
        const float from = combinations[v1_idx * n_combination_dims + dim];
        const float to   = combinations[v2_idx * n_combination_dims + dim];
        combinations[vertex_idx * n_combination_dims + dim] = (ratio2 * from) + (ratio * to);
      }

      // assumes that the first `n_new_vertices` edges need interpolation
      result_vertex_idxes[todo_idx] = vertex_idx;
    }
  }
  return;
}

void ReLUTransformer::Intersect(
  UPolytope* inout,
  const size_t plane_idx,
  std::vector<int> & result_point_sign,
  std::vector<size_t> & result_intersected_edge_idxes,
  std::vector<size_t> & result_intersection_vertex_idxes,
  size_t & n_intersected_edges,
  size_t & n_new_vertices
) const {

#ifdef TIME_INTERSECTION
  Timer t;
#endif

  thrust::device_vector<size_t> & edge_endpoints = inout->endpoints();

#ifdef TIME_INTERSECTION
  std::cerr << ">>> time get endpoints: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  thrust::device_vector<int> & point_sign = PointSignDevice(inout->vertices_device(), plane_idx);

#ifdef TIME_INTERSECTION
  std::cerr << ">>> time calc signs   : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  Intersect(
    inout->vertices_device(),
    inout->combinations_device(),
    inout->n_edges(),
    edge_endpoints,
    point_sign,
    plane_idx,
    result_point_sign,
    result_intersected_edge_idxes,
    result_intersection_vertex_idxes,
    n_intersected_edges,
    n_new_vertices
  );

#ifdef TIME_INTERSECTION
  std::cerr << ">>> time intersection : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  return;
}

void ReLUTransformer::Intersect(
  RMMatrixXfDevice& vertices,
  RMMatrixXfDevice& combinations,
  const size_t n_edges,
  thrust::device_vector<size_t> & edge_endpoints,
  const thrust::device_vector<int> & point_sign,
  const size_t plane_idx,
  std::vector<int> & result_point_sign,
  std::vector<size_t> & result_intersected_edge_idxes,
  std::vector<size_t> & result_intersection_vertex_idxes,
  size_t & n_intersected_edges,
  size_t & n_new_vertices
) const {

#ifdef TIME_INTERSECTION
  Timer t;
#endif

  const size_t n_original_vertices = vertices.rows();
  const size_t n_dims = vertices.cols();

  if (todo_edge_idxes.capacity() < n_edges) {
    todo_edge_idxes.resize(n_edges);
    std::cerr << "resize " << n_edges << " for todo_edge_idxes\n";
  }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time create seq    : " << t.Ticks() << std::endl;
  t.Reset();
#endif

#ifdef NO_THRUST_PARTITION
  auto todo_edge_idxes_end = thrust::copy_if(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(n_edges),
    todo_edge_idxes.begin(),
    need_interpolation(raw_pointer_cast(edge_endpoints.data()),
                     raw_pointer_cast(point_sign.data())));
  n_new_vertices = todo_edge_idxes_end - todo_edge_idxes.begin();

  todo_edge_idxes_end = thrust::copy_if(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(n_edges),
    todo_edge_idxes.begin() + n_new_vertices,
    no_interpolation(raw_pointer_cast(edge_endpoints.data()),
                     raw_pointer_cast(point_sign.data())));
  n_intersected_edges = todo_edge_idxes_end - todo_edge_idxes.begin();
#else
  // NOTE: We need the first `n_new_vertices` elements be the edges to
  // interpolate, because that's how we resize the vertices matrix and determine
  // the indexes of new vertices.
  auto todo_edge_idxes_end = thrust::copy_if(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(n_edges),
    todo_edge_idxes.begin(),
    has_intersection(raw_pointer_cast(edge_endpoints.data()),
                     raw_pointer_cast(point_sign.data())));
  n_intersected_edges = todo_edge_idxes_end - todo_edge_idxes.begin();

  // TODO(zhetao): Try to optimize this partition.
  // NOTE(zhetao): We need all intersected edges to determine which plane to
  // split. Seems we anyway need to filter out edges which does not cross the
  // hyperplane. Because I think it might be better to only interpolate edges
  // crossing the hyperplane because we are interpolating new vertices before
  // the actual split of planes, there edges which are on the hyperplane (i.e.,
  // both endpoints are on the hyperplane) should not be splitted and
  // interpolated.
  todo_edge_idxes_end = thrust::partition(
    todo_edge_idxes.begin(),
    todo_edge_idxes.begin() + n_intersected_edges,
    need_interpolation(raw_pointer_cast(edge_endpoints.data()),
                       raw_pointer_cast(point_sign.data())));
  n_new_vertices = todo_edge_idxes_end - todo_edge_idxes.begin();
#endif

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time create todo   : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  if (result_vertex_idxes.capacity() < n_new_vertices) {
    result_vertex_idxes.resize(n_new_vertices);
    std::cerr << "resize " << n_new_vertices << " for result_vertex_idxes\n";
  }

  // std::cerr << n_original_vertices << "," << n_new_vertices << "\n";
  const size_t n_preallocate_vertices = n_original_vertices + n_new_vertices;
  // if (vertices.capacity() < n_preallocate_vertices * vertices.cols() ||
  //     combinations.capacity() < n_preallocate_vertices * combinations.cols()) {
  //   // boost capcacity
  //   vertices.reserve_rows(n_preallocate_vertices);
  //   combinations.reserve_rows(n_preallocate_vertices);
  //   std::cerr << "boost " << n_preallocate_vertices << " for vertices and combinations\n";
  // }
  vertices.resize_rows(n_preallocate_vertices);
  combinations.resize_rows(n_preallocate_vertices);

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time reserve space : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_new_vertices / threadsPerBlock.x) + 1);
  assert (nBlocks.x * threadsPerBlock.x  >= n_new_vertices);

  InterpolateVertex<<<nBlocks, threadsPerBlock>>>(
    n_new_vertices,
    raw_pointer_cast(todo_edge_idxes.data()),
    raw_pointer_cast(result_vertex_idxes.data()),
    vertices.data(), n_dims,
    combinations.data(), combinations.cols(),
    n_original_vertices,
    raw_pointer_cast(edge_endpoints.data()),
    raw_pointer_cast(point_sign.data()),
    plane_idx
  );

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time interpolation : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  UpdateEndpointsKernelReLU<<<nBlocks, threadsPerBlock>>>(
    n_new_vertices,
    n_original_vertices,
    raw_pointer_cast(todo_edge_idxes.data()),
    raw_pointer_cast(result_vertex_idxes.data()),
    raw_pointer_cast(edge_endpoints.data())
  );

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time upd endpoints : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  if (result_intersected_edge_idxes.capacity() < n_intersected_edges) {
    result_intersected_edge_idxes.resize(n_intersected_edges);
    std::cerr << "resize " << n_intersected_edges << " for result_intersected_edge_idxes\n";
  }
  thrust::copy(todo_edge_idxes.begin(),
               todo_edge_idxes.begin() + n_intersected_edges,
               result_intersected_edge_idxes.begin());

  if (result_intersection_vertex_idxes.capacity() < n_new_vertices) {
    result_intersection_vertex_idxes.resize(n_new_vertices);
    std::cerr << "resize " << n_new_vertices << " for result_intersection_vertex_idxes\n";
  }
  thrust::copy(result_vertex_idxes.begin(),
               result_vertex_idxes.begin() + n_new_vertices,
               result_intersection_vertex_idxes.begin());

  if (result_point_sign.capacity() < n_original_vertices) {
    result_point_sign.resize(n_original_vertices);
    std::cerr << "resize " << n_original_vertices << " for result_point_sign\n";
  }
  thrust::copy(point_sign.begin(),
               point_sign.begin() + n_original_vertices,
               result_point_sign.begin());

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time copy results  : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  return;
}
