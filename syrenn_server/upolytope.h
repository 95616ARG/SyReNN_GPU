#ifndef SYRENN_SYRENN_SERVER_UPOLYTOPE_H_
#define SYRENN_SYRENN_SERVER_UPOLYTOPE_H_

#include <memory>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "tbb/tbb.h"
#include "tbb/mutex.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_proto/syrenn.grpc.pb.h"

struct EdgeMapCompare {
  static size_t hash(const std::pair<size_t, size_t>& x) {
    size_t ret = 0;
    hash_combine(&ret, x.first, x.second);
    return ret;
  }
  static bool equal(const std::pair<size_t, size_t>& x,
                    const std::pair<size_t, size_t>& y) {
    return x == y;
  }
};

typedef tbb::concurrent_hash_map<std::pair<size_t, size_t>, size_t,
                                 EdgeMapCompare> EdgeMap;

class Edge;
class Face;

// Represents a union of V-Polytopes. Each vertex has a "post" and "pre"
// representation; the "post" representation is referred to as the "vertex" and
// the "pre" representation is referred to as the "combination" because it
// represents a convex combination of the pre-transformation vertices.
//
// OPTIMIZATION NOTES: Particularly optimized for the case where many of the
// polytopes share vertices. This implementation stores all vertices
// contiguously in memory, and each polytope is simply a list of indices into
// that contiguous block. This may not be the best choice for all scenarios,
// especially if the block of vertices does not fit in memory.
//
// MULTI-THREADING: Designed for multi-threaded transformer algorithms. In
// particular, new vertices are transparently added to a ``pending'' vector,
// which is merged into the main block when the vertices() method is called.
// Refer to the per-method comments for more information on how to use the
// class in a multi-threaded environment. For the most part, as long as you
// only call "vertices" in a single-threaded environment and assign at most one
// thread per polytope, it should be hard to break things.
class UPolytope {
 public:
  UPolytope(RMMatrixXf *vertices, size_t subspace_dimensions,
            std::vector<std::vector<size_t>> polytopes,
            const int device = -1);
  // Stores the upolytope on device if @vertices is RMMatrixXfDevice*.
  // TODO(zhetao): Revise this API.
  // UPolytope(RMMatrixXfDevice *vertices, size_t subspace_dimensions,
  //           std::vector<std::vector<size_t>> polytopes);

  // Stores the upolytope on device iff @on_device
  // TODO(zhetao): Revise this API.
  static UPolytope Deserialize(const syrenn_server::UPolytope &upolytope,
                               const int device = -1);
  syrenn_server::UPolytope Serialize();

  // Returns a mutable reference to a Matrix containing all vertices.
  // MULTI-THREADING: This method should *NOT* be called when multiple threads
  // may be simultaneously accessing the UPolytope instance; instead, use the
  // vertex and AppendVertex methods. It will serially collapse pending_* into
  // vertices and combinations before returning.
  RMMatrixXf &vertices();
  RMMatrixXfDevice &vertices_device();
  RMMatrixXf &combinations();
  RMMatrixXfDevice &combinations_device();

  // Returns a mutable reference to the vector of vertices for a particular
  // polytope.
  // MULTI-THREADING: This method *MAY* be called when multiple threads are
  // simultaneously accessing the UPolytope instance, _as long as_ no two
  // threads are accessing the vertex_indices for the same polytope.
  std::vector<size_t> &vertex_indices(size_t polytope);
  // Returns true iff the UPolytope lives in a two-dimensional subspace and its
  // vertices are in counter-clockwise orientation. (NOTE that in this
  // implementation, it is assumed that all 2D UPolytopes are in CCW
  // orientation).
  // MULTI-THREADING: Always safe.
  bool is_counter_clockwise() const;
  // Returns the number of dimensions of the space that the polytope lives in.
  // This is the same as the number of components of each vertex.
  // MULTI-THREADING: Always safe.
  size_t space_dimensions() const;
  // Returns the number of convex polytopes in this UPolytope.
  // MULTI-THREADING: Always safe.
  size_t n_polytopes() const;
  // Returns the number of vertices in a particular polytope.
  // MULTI-THREADING: Always safe.
  size_t n_vertices(size_t polytope) const;
  size_t n_vertices() const;
  // Returns the raw index of the @vertexth vertex in the @polytopeth polytope.
  // MULTI-THREADING: Always safe.
  size_t vertex_index(size_t polytope, size_t vertex) const;
  // Returns an (immutable) reference to the vertex indexed by @raw_index.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> vertex(size_t raw_index) const;
  // Returns an (immutable) reference to the @vertexth vertex in the
  // @polytopeth polytope.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> vertex(size_t polytope, size_t vertex) const;
  // Returns an (immutable) reference to the combination indexed by @raw_index.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> combination(size_t raw_index) const;
  // Returns an (immutable) reference to the @vertexth combination in the
  // @polytopeth polytope.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> combination(size_t polytope,
                                           size_t vertex) const;
  // Appends @vertex and @combination to the list of vertices/combinations,
  // returning their indices.
  // NOTE: This function is *DESTRUCTIVE* to @vertex and @combination.
  // MULTI-THREADING: Safe to call in a multi-threaded evironment. Acquires a
  // write lock on @pending_vertices_ and @pending_combinations_.
  size_t AppendVertex(RMVectorXf *vertex, RMVectorXf *combination);
  // Appends @vertex_indices to the list of polytopes, returning the new
  // polytope's index.
  // NOTE: This function is *DESTRUCTIVE* to @vertex_indices.
  // MULTI-THREADING: Safe to call in a multi-threaded environment. Acquires a
  // write lock on @polytopes_.
  size_t AppendPolytope(std::vector<size_t> *vertex_indices);

  // APIs relate to edges

  // Returns the amount of edges.
  size_t n_edges() const;
  // Returns the edge index of given endpoints (regardless order).
  size_t edge_idx(size_t v1_idx, size_t v2_idx);
  // Returns a reference to the edge.
  Edge & edge(size_t edge_idx);
  Edge & edge(size_t v1_idx, size_t v2_idx);
  // Returns if the edge of given endpoints exists (regardless order).
  bool HasEdge(size_t v1_idx, size_t v2_idx);
  // Appends an edge of given endpoints (regardless order) and returns its index.
  // NOTE: requires that HasEdge(v1, v2) returns false.
  size_t AppendEdge(size_t v1, size_t v2);
  // Reuses the @edge_idx, updates it with @new_v1_idx and @new_v2_idx.
  void UpdateEdge(size_t edge_idx, size_t new_v1_idx, size_t new_v2_idx);
  // Splits the edge of @edge_idx with the vertex of @vertex_idx.
  // NOTE(zhetao): Does not update the faces containing edge_idx.
  size_t SplitEdge(size_t edge_idx, size_t vertex_idx);
  // Returns a reference to a contigious vector of edge endpoints on device.
  thrust::device_vector<size_t> & endpoints();
  std::vector<size_t> & endpoints_host();
  // void UpdateEndpoints(size_t edge_idx);
  // Pre-allocate device memory.
  // TODO(zhetao): Revise this API.
  void reserve_n_edges(size_t n_extra);

  int GetAttachedFace(size_t edge_idx, size_t idx2);
  int & GetAttachedFaceRef(size_t edge_idx, size_t idx2);
  void AttachEdgeToFace(size_t edge_idx, size_t face_idx);
  void AttachEdgeToFaces(size_t edge_idx, size_t face1_idx, size_t face2_idx);
  void DetachEdgeFromFace(size_t edge_idx, size_t face_idx);
  void UpdateEdgeToFace(size_t edge_idx, size_t old_face_idx, size_t new_face_idx);

  RMVectorXi & labels();

  void ComputePreImage();
  void Classify();

  bool include_post_ = false;
  bool computed_preimage_ = false;
  bool classified_ = false;

  // APIs related to faces.

  // Returns the amount of faces.
  // size_t n_faces() const;
  // Returns the index of a new face.
  size_t NewFace();
  // Returns the face of given @idx.
  Face face(size_t idx);

  RMMatrixXf input_plane_;
  RMMatrixXfDevice input_plane_device_;

  int device_ = -1;
  void to(const int device);

 private:
  void FlushPending();

  RMMatrixXf vertices_;
  RMMatrixXfDevice vertices_device_;
  RMMatrixXf combinations_;
  RMMatrixXfDevice combinations_device_;
  // RMMatrixXfDevice preimage_;
  RMVectorXi labels_;
  // thrust::host_vector<int> labels_;

  // tbb::concurrent_vector<Face> faces_;
  tbb::concurrent_vector<Edge> edges_;
  // std::vector<Edge> edges_;
  EdgeMap edge_map_;

  std::vector<int> faces_of_edges_;

  size_t n_finished_edges_ = 0;
  thrust::device_vector<size_t> endpoints_;
  std::vector<size_t> endpoints_host_;
  thrust::host_vector<size_t> new_endpoints_;

  size_t subspace_dimensions_;
  tbb::concurrent_vector<std::vector<size_t>> polytopes_;

  struct PendingVertex {
    PendingVertex(RMVectorXf *vertex, RMVectorXf *combination)
        : vertex(0), combination(0) {
      this->vertex.swap(*vertex);
      this->combination.swap(*combination);
    }
    RMVectorXf vertex;
    RMVectorXf combination;
  };
  tbb::concurrent_vector<PendingVertex> pending_;
};

class Edge {
public:
  Edge(size_t v1_idx, size_t v2_idx)
  : v1_idx(v1_idx), v2_idx(v2_idx) {}

  friend std::ostream& operator<<(std::ostream& os, const Edge& edge);

  size_t v1_idx, v2_idx;
};

// NOTE(zhetao): In current case an Face is the same as the original polytope.
class Face{
public:
  Face(std::vector<size_t> & vertex_idxes) : vertex_idxes(vertex_idxes) {};

  // Returns the amount of vertices on this face.
  size_t n_vertices() const;
  // Returns the vertex_idx of @idx2.
  size_t vertex_idx(size_t idx2) const;
  // Returns the last vertex_idx of @idx2.
  size_t last_vertex_idx(size_t idx2) const;
  // Returns if there is no vertex on this face.
  bool empty() const;
  // Prints the vertices on this face.
  friend std::ostream& operator<<(std::ostream& os, const Face& face);

  // Stores the vertices on this face in counter-clockwise order.
  std::vector<size_t> & vertex_idxes;

};

#endif  // SYRENN_SYRENN_SERVER_UPOLYTOPE_H_
