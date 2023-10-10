
#include "syrenn_server/argmax_transformer.h"
#include <memory>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

RMMatrixXiDevice ArgMaxTransformer::square_index;
int ArgMaxTransformer::computed_square_index_for = -1;

std::pair<size_t, size_t> SquareIndexFromTriangular2(const size_t face,
                                                    const size_t dims) {
  // Formula derived & tested here:
  // https://stackoverflow.com/questions/27086195
  size_t i = dims - 2 - std::floor(
      (std::sqrt((-8 * face) + (4 * dims * (dims - 1) - 7))
      / 2.0) - 0.5);
  size_t j = face + i + 1 - (dims * (dims - 1) / 2) +
              ((dims - i) * ((dims - i) - 1)) / 2;
  return std::make_pair(i, j);
  // Iterative version:
  // for (size_t i = 0; i < dims; i++) {
  //   for (size_t j = i + 1; j < dims; j++, f++) {
  //     if (f == face) {
  //       return std::make_pair(i, j);
  //     }
  //   }
  // }
}

__global__ void ComputeSquareIndexFromTriangularKernel(
                  const size_t n_faces,
                  const size_t n_dims,
                  int* square_index) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face < n_faces) {
    square_index[face * 2] = n_dims - 2 - floorf(
        (sqrtf((-8 * face) + (4 * n_dims * (n_dims - 1) - 7))
        / 2.0) - 0.5);
    square_index[face * 2 + 1] = face + square_index[face * 2] + 1 - (n_dims * (n_dims - 1) / 2) +
                ((n_dims - square_index[face * 2]) * ((n_dims - square_index[face * 2]) - 1)) / 2;
  }
}

void ArgMaxTransformer::ComputeSquareIndexFromTriangular(const size_t n_dims) const {
  if (computed_square_index_for == n_dims) {
    return;
  }
  computed_square_index_for = n_dims;
  const size_t n_faces = n_piece_faces(n_dims);
  square_index.resize(n_faces, 2);

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_faces / threadsPerBlock.x) + 1);
  ComputeSquareIndexFromTriangularKernel<<<nBlocks, threadsPerBlock>>>(
    n_faces,
    n_dims,
    square_index.data());
}

// TODO(zhetao): Leave stubs for now
// Single.
double ArgMaxTransformer::CrossingRatio(const RMMatrixXfDevice &vertices,
                      size_t from_idx,
                      size_t to_idx,
                      const size_t face) const {
  assert (false);
}

// Selective.
std::vector<float> ArgMaxTransformer::CrossingRatio(const RMMatrixXfDevice &vertices,
                              const std::vector<size_t> &from_idxes,
                              const std::vector<size_t> &to_idxes,
                              const size_t face) const {
  assert (false);
}

// Single.
int ArgMaxTransformer::PointSign(const RMMatrixXfDevice &vertices,
              size_t vertex_idx,
              const size_t face) const {
  return PointSign(vertices.row(vertex_idx), face);
}

__global__ void PointSignKernel(
                  const size_t n_vertices,
                  const size_t n_dims,
                  const size_t face,
                  const float* vertices,
                  const int* square_index,
                  int* point_sign) {
  const int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex_idx < n_vertices) {
    const size_t i = square_index[face*2];
    const size_t j = square_index[face*2+1];
    const float point_i = vertices[vertex_idx * n_dims + i];
    const float point_j = vertices[vertex_idx * n_dims + j];
    if (point_i == point_j) {
      point_sign[vertex_idx] = 0;
    } else {
      point_sign[vertex_idx] = (point_i > point_j) ? +1 : -1;
    }
  }
}

// Actual.
thrust::device_vector<int> & ArgMaxTransformer::PointSignDevice(
                            const RMMatrixXfDevice &vertices,
                            const size_t face) const {
  const size_t n_vertices = vertices.rows();
  const size_t n_dims = vertices.cols();
  ComputeSquareIndexFromTriangular(n_dims);

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_vertices / threadsPerBlock.x) + 1);
  PointSignKernel<<<nBlocks, threadsPerBlock>>>(
    n_vertices,
    n_dims,
    face,
    vertices.data(),
    square_index.data(),
    raw_pointer_cast(PWLTransformer::point_sign_device.data())
  );

  return PWLTransformer::point_sign_device;
}

__global__ void ArgMaxKernel(const size_t n_vertices,
                             const size_t n_dims,
                             const float* vertices,
                             float* result) {
  const int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex_idx < n_vertices) {
    float max = vertices[vertex_idx * n_dims];
    result[vertex_idx] = 0;
    for (size_t dim = 1; dim < n_dims; dim++) {
      if (vertices[vertex_idx * n_dims + dim] > max) {
        max = vertices[vertex_idx * n_dims + dim];
        result[vertex_idx] = (float)dim;
      }
    }
  }
}

void ArgMaxTransformer::ComputeDevice(RMMatrixXfDevice *inout) const {

  // NOTE(zhetao): Possible alternative: thrust::cuda_cub::cub::DeviceReduce::ArgMax.

  const size_t n_vertices = inout->rows();
  const size_t n_dims = inout->cols();
  const size_t n_faces = n_piece_faces(n_dims);
  ComputeSquareIndexFromTriangular(n_dims);

  CUDAShared::output.resize(n_vertices, 1);

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_vertices / threadsPerBlock.x) + 1);
  ArgMaxKernel<<<nBlocks, threadsPerBlock>>>(
    n_vertices,
    n_dims,
    inout->data(),
    CUDAShared::output.data());

  inout->swap(CUDAShared::output);

}

__global__
void UpdateEndpointsKernelArgMax(
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
  const int* square_index,
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
      const size_t i = square_index[plane_idx*2];
      const size_t j = square_index[plane_idx*2+1];
      const float from_i = vertices[v1_idx * n_dims + i];
      const float from_j = vertices[v1_idx * n_dims + j];
      const float to_i = vertices[v2_idx * n_dims + i];
      const float to_j = vertices[v2_idx * n_dims + j];
      const float i_delta = to_i - from_i;
      const float j_delta = to_j - from_j;
      const float ratio = (from_j - from_i) / (i_delta - j_delta);
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

void ArgMaxTransformer::Intersect(
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

void ArgMaxTransformer::Intersect(
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

  if (todo_edge_idxes.size() < n_edges) { todo_edge_idxes.resize(n_edges); }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time create seq    : " << t.Ticks() << std::endl;
  t.Reset();
#endif

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

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time create todo   : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  if (result_vertex_idxes.size() < n_new_vertices) { result_vertex_idxes.resize(n_new_vertices); }
  vertices.resize_rows(n_original_vertices + n_new_vertices);
  combinations.resize_rows(n_original_vertices + n_new_vertices);

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time reserve space : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_new_vertices / threadsPerBlock.x) + 1);

  // assert (nBlocks.x * threadsPerBlock.x  >= n_new_vertices);

  InterpolateVertex<<<nBlocks, threadsPerBlock>>>(
    n_new_vertices,
    raw_pointer_cast(todo_edge_idxes.data()),
    raw_pointer_cast(result_vertex_idxes.data()),
    vertices.data(), n_dims,
    combinations.data(), combinations.cols(),
    square_index.data(),
    n_original_vertices,
    raw_pointer_cast(edge_endpoints.data()),
    raw_pointer_cast(point_sign.data()),
    plane_idx
  );

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time interpolation : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  UpdateEndpointsKernelArgMax<<<nBlocks, threadsPerBlock>>>(
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

  if (result_intersected_edge_idxes.size() < n_intersected_edges) {
    result_intersected_edge_idxes.resize(n_intersected_edges);
  }
  thrust::copy(todo_edge_idxes.begin(),
               todo_edge_idxes.begin() + n_intersected_edges,
               result_intersected_edge_idxes.begin());

  if (result_intersection_vertex_idxes.size() < n_new_vertices) {
    result_intersection_vertex_idxes.resize(n_new_vertices);
  }
  thrust::copy(result_vertex_idxes.begin(),
               result_vertex_idxes.begin() + n_new_vertices,
               result_intersection_vertex_idxes.begin());

  if (result_point_sign.size() < n_original_vertices) {
    result_point_sign.resize(n_original_vertices);
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
