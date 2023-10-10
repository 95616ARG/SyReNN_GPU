#include "syrenn_server/upolytope.h"
#include <memory>
#include <vector>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void UPolytope::to(const int target_device) {
  FlushPending();

  if (device_ == target_device) {
    return;
  }

  if (device_ == SYRENN_DEVICE_CPU && target_device >= 0) {
    vertices_device_.fill(vertices_);
    combinations_device_.fill(combinations_);
    // std::cerr << "Warning: Transporting a UPolytope from host to device is incomplete now, further plane transformation will be incorrect.\n";
    device_ = target_device;
    return;
  }

  if (device_ >= 0 && target_device == SYRENN_DEVICE_CPU) {
    vertices_ = vertices_device_.eigen();
    combinations_ = combinations_device_.eigen();
    device_ = target_device;
    return;
  }

  throw "Error: Unsupported UPolytope transportation request.";
}

UPolytope::UPolytope(RMMatrixXf *vertices, size_t subspace_dimensions,
                     std::vector<std::vector<size_t>> polytopes,
                     const int device)
    : vertices_(0, 0), combinations_(0, 0),
      vertices_device_(0, 0), combinations_device_(0, 0),
      faces_of_edges_(0, -1), endpoints_(0), new_endpoints_(0),
      subspace_dimensions_(subspace_dimensions),
      polytopes_(),
      device_(device) {

  if (device_ == -1) {
    vertices_.swap(*vertices);
    input_plane_ = RMMatrixXf(vertices_);
    combinations_.resize(vertices_.rows(), vertices_.rows());
    combinations_.setIdentity();
    polytopes_ = tbb::concurrent_vector<std::vector<size_t>>(polytopes.begin(), polytopes.end());

    faces_of_edges_ = std::vector<int>(80000000, -1);
    endpoints_.resize(80000000);
    new_endpoints_.resize(8192);

    for (auto const & polytope: polytopes) {
      if (polytope.size() <= 1) {
        continue;
      }
      const size_t face_idx = NewFace();
      face(face_idx).vertex_idxes.reserve(polytope.size());
      for (size_t i = 0; i < polytope.size(); i++) {
        if (polytope[i] == polytope[(i + 1) % polytope.size()]) {
          continue;
        }
        const size_t v1_idx = polytope[i];
        const size_t v2_idx = polytope[(i + 1) % polytope.size()];
        face(face_idx).vertex_idxes.emplace_back(v1_idx);
        AttachEdgeToFace(
          HasEdge(v1_idx, v2_idx) ? edge_idx(v1_idx, v2_idx) : AppendEdge(v1_idx, v2_idx),
          face_idx);
      }
    }

    // std::cerr << this->vertices().block(0,0,3,3) << std::endl;
  } else {
    vertices_device_ = RMMatrixXfDevice(*vertices);
    input_plane_device_ = RMMatrixXfDevice(vertices_device_);
    vertices_device_.set_capacity(1200000 * 200);

    RMMatrixXf tmp(vertices_device_.rows(), vertices_device_.rows());
    tmp.setIdentity();
    combinations_device_ = RMMatrixXfDevice(tmp, 1200000 * 3);

    faces_of_edges_ = std::vector<int>(80000000, -1);
    endpoints_.resize(80000000);
    new_endpoints_.resize(8192);

    for (auto const & polytope: polytopes) {
      if (polytope.size() <= 1) {
        continue;
      }
      const size_t face_idx = NewFace();
      face(face_idx).vertex_idxes.reserve(polytope.size());
      for (size_t i = 0; i < polytope.size(); i++) {
        if (polytope[i] == polytope[(i + 1) % polytope.size()]) {
          continue;
        }
        const size_t v1_idx = polytope[i];
        const size_t v2_idx = polytope[(i + 1) % polytope.size()];
        face(face_idx).vertex_idxes.emplace_back(v1_idx);
        AttachEdgeToFace(
          HasEdge(v1_idx, v2_idx) ? edge_idx(v1_idx, v2_idx) : AppendEdge(v1_idx, v2_idx),
          face_idx);
      }
    }
  }
}

// UPolytope::UPolytope(RMMatrixXfDevice *vertices, size_t subspace_dimensions,
//                      std::vector<std::vector<size_t>> polytopes)
//     : vertices_(0, 0), combinations_(0, 0),
//       vertices_device_(), combinations_device_(),
//       faces_of_edges_(80000000, -1), endpoints_(80000000), new_endpoints_(8192),
//       subspace_dimensions_(subspace_dimensions),
//       polytopes_(),
//       use_gpu_(true) {

//   vertices_device_.swap(*vertices);
//   RMMatrixXf tmp(vertices_device_.rows(), vertices_device_.rows());
//   tmp.setIdentity();
//   combinations_device_ = RMMatrixXfDevice(tmp, 1200000 * 3);

//   input_plane_device_ = RMMatrixXfDevice(vertices_device_);
//   vertices_device_.set_capacity(1200000 * 200);

//   for (auto const & polytope: polytopes) {
//     if (polytope.size() <= 1) {
//       continue;
//     }
//     const size_t face_idx = NewFace();
//     face(face_idx).vertex_idxes.reserve(polytope.size());
//     for (size_t i = 0; i < polytope.size(); i++) {
//       if (polytope[i] == polytope[(i + 1) % polytope.size()]) {
//         continue;
//       }
//       const size_t v1_idx = polytope[i];
//       const size_t v2_idx = polytope[(i + 1) % polytope.size()];
//       face(face_idx).vertex_idxes.emplace_back(v1_idx);
//       AttachEdgeToFace(
//         HasEdge(v1_idx, v2_idx) ? edge_idx(v1_idx, v2_idx) : AppendEdge(v1_idx, v2_idx),
//         face_idx);
//     }
//   }
// }

thrust::device_vector<size_t> & UPolytope::endpoints() {

  const size_t n_endpoints = n_edges() * 2;
  const size_t n_new_edges = n_edges() - n_finished_edges_;
  const size_t n_new_endpoints = n_new_edges * 2;

#ifdef TIME_INTERSECTION
  Timer t;
#endif

  if (endpoints_.size() < n_endpoints) {
    endpoints_.resize(n_endpoints);
  }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time reserve device space: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  if (new_endpoints_.size() < n_new_endpoints) {
    new_endpoints_.resize(n_new_endpoints);
  }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time reserve host space: " << t.Ticks() << std::endl;
  t.Reset();
  std::cerr << "working on n new edges: " << n_new_edges << std::endl;
#endif

  for (size_t i = 0; i < n_new_edges; i++) {
    new_endpoints_[i * 2] = edges_[n_finished_edges_ + i].v1_idx;
    new_endpoints_[i * 2 + 1] = edges_[n_finished_edges_ + i].v2_idx;
  }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time setup: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  thrust::copy(new_endpoints_.begin(),
               new_endpoints_.begin() + n_new_endpoints,
               endpoints_.begin() + n_finished_edges_ * 2);
  n_finished_edges_ = n_edges();

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time copy: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  return endpoints_;
}

void UPolytope::ComputePreImage() {
  if (device_ == SYRENN_DEVICE_CPU) {
    FlushPending();
    combinations_ *= input_plane_;
  } else {
    combinations_device_ *= input_plane_device_;
  }
  computed_preimage_ = true;
  return;
}

template<class T>
__global__ void ArrayAddKernel(const size_t len,
                               const T* a,
                               const T* b,
                               T* c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
      c[i] = a[i] + b[i];
    }
}

template<class T>
__device__ void ArrayAdd(const size_t len,
                               const T* a,
                               const T* b,
                               T* c) {
    for (int i = 0; i < len; i++) {
      c[i] = a[i] + b[i];
    }
}

template<class T>
__global__ void ArrayDivByScalarKernel(const size_t len,
                                       const T k,
                                       T* a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
      a[i] /= k;
    }
}

template<class T>
__device__ void ArrayDivByScalar(const size_t len,
                      const T k,
                      T* a) {
    for (int i = 0; i < len; i++) {
      a[i] /= k;
    }
}

__global__ void ClassifyKernel(const size_t n_faces,
                               const size_t n_dims,
                               const float* vertices,
                               const size_t* n_face_vertices,
                               const size_t* face_offset,
                               const size_t* n_face_vertex_idxes,
                               float* workspace,
                               int* labels ) {
  const size_t face_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (face_idx < n_faces) {
    // Summing.
    ArrayAdd<float>(n_dims,
                          workspace + face_idx * n_dims,
                          vertices + n_face_vertex_idxes[face_offset[face_idx]] * n_dims,
                          workspace + face_idx * n_dims);
    for (size_t vertex_idx = 1; vertex_idx < n_face_vertices[face_idx]; vertex_idx++) {
      ArrayAdd<float>(n_dims,
                            workspace + face_idx * n_dims,
                            vertices + n_face_vertex_idxes[face_offset[face_idx] + vertex_idx] * n_dims,
                            workspace + face_idx * n_dims);
    }

    // Averaging.
    ArrayDivByScalar<float>(n_dims,
                                  static_cast<float>(n_face_vertices[face_idx]),
                                  workspace + face_idx * n_dims);

    // Sequential ArgMax.
    float max = workspace[face_idx * n_dims];
    labels[face_idx] = 0;
    for (int dim = 1; dim < n_dims; dim++) {
      if (workspace[face_idx * n_dims + dim] > max) {
        max = workspace[face_idx * n_dims + dim];
        labels[face_idx] = dim;
      }
    }
  }
}

// void UPolytope::Classify() {
//   labels_.resize(n_polytopes());
//   int index = -1;
//   for (size_t p_idx = 0; p_idx < n_polytopes(); p_idx++) {
//     labels_(p_idx) = vertices_(polytopes_[p_idx], Eigen::all).colwise().mean().maxCoeff(&index);
//   }
//   classified_ = true;
//   return;
// }

void UPolytope::Classify() {
  if (device_ != SYRENN_DEVICE_CPU && vertices_.size() != vertices_device_.size()) {
    std::cerr << "Copying vertices to host to classify...\n";
    vertices_ = vertices_device_.eigen();
  }

  labels_.resize(n_polytopes());
  int index = -1;
  for (size_t p_idx = 0; p_idx < n_polytopes(); p_idx++) {
    vertices_(polytopes_[p_idx], Eigen::all).colwise().mean().maxCoeff(&index);
    labels_(p_idx) = index;
  }
  classified_ = true;
  return;
}

/*
void UPolytope::Classify() {

  const size_t n_classified_faces = n_polytopes();
  const size_t n_dims = vertices_device_.cols();

  Timer t; t.Reset();

  thrust::device_vector<size_t> n_face_vertices(n_classified_faces);
  thrust::device_vector<size_t> face_offset(n_classified_faces);
  std::cerr << t.Ticks() << " -- allocate 1\n"; t.Reset();

  size_t offset = 0;
  for (size_t face_idx = 0; face_idx < n_classified_faces; face_idx++) {
    n_face_vertices[face_idx] = face(face_idx).n_vertices();
    face_offset[face_idx] = offset;
    offset += n_face_vertices[face_idx];
  }
  std::cerr << t.Ticks() << " -- prepare 1\n"; t.Reset();

  thrust::device_vector<size_t> n_face_vertex_idxes(offset);
  std::cerr << t.Ticks() << " -- allocate 2\n"; t.Reset();

  for (size_t face_idx = 0; face_idx < n_classified_faces; face_idx++) {
    for (size_t idx2 = 0; idx2 < n_face_vertices[face_idx]; idx2++) {
      n_face_vertex_idxes[face_offset[face_idx]+idx2] = face(face_idx).vertex_idx(idx2);
    }
  }
  std::cerr << t.Ticks() << " -- prepare2\n"; t.Reset();

  thrust::device_vector<float> workspace(n_classified_faces * n_dims);
  RMMatrixXiDevice output(n_classified_faces, 1);
  std::cerr << t.Ticks() << " -- allocate workspace and output\n"; t.Reset();

  const dim3 threadsPerBlock(1024);
  const dim3 nBlocks((n_classified_faces / threadsPerBlock.x) + 1);
  ClassifyKernel<<<nBlocks, threadsPerBlock>>>(
    n_classified_faces,
    n_dims,
    vertices_device_.data(),
    raw_pointer_cast(n_face_vertices.data()),
    raw_pointer_cast(face_offset.data()),
    raw_pointer_cast(n_face_vertex_idxes.data()),
    raw_pointer_cast(workspace.data()),
    output.data()
  );

  std::cerr << t.Ticks() << " -- classify kernel\n"; t.Reset();

  // std::cerr << output.rows() << std::endl;
  // std::cerr << output.cols() << std::endl;

  // labels_ = output.eigen();
  // std::cerr << labels_.rows() << std::endl;
  // std::cerr << labels_.cols() << std::endl;

  labels_ = output.host();
  std::cerr << t.Ticks() << " -- copy to host\n"; t.Reset();

  classified_ = true;
  return;
}
*/
