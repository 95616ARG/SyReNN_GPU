#include <utility>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "syrenn_server/upolytope.h"

RMVectorXi & UPolytope::labels() {
  return labels_;
}

UPolytope UPolytope::Deserialize(const syrenn_server::UPolytope &upolytope,
                                 const int device) {
  // TODO(masotoud): assume that combinations is empty.
  int n_vertices = 0;
  for (int i = 0; i < upolytope.flatten().polytopes_size(); i++) {
    auto &polytope = upolytope.flatten().polytopes()[i];
    n_vertices += polytope.num_vertices();
  }
  RMMatrixXf vertices(n_vertices, upolytope.flatten().space_dimensions());
  std::vector<std::vector<size_t>> polytopes(upolytope.flatten().polytopes_size());
  int vertices_index = 0;
  for (int i = 0; i < upolytope.flatten().polytopes_size(); i++) {
    auto &polytope = upolytope.flatten().polytopes()[i];
    vertices.block(vertices_index, 0,
                   polytope.num_vertices(),
                   upolytope.flatten().space_dimensions()) =
        Eigen::Map<const RMMatrixXf>(polytope.vertices().data(),
                                     polytope.num_vertices(),
                                     upolytope.flatten().space_dimensions());
    for (size_t j = 0; j < polytope.num_vertices(); j++) {
      polytopes[i].push_back(vertices_index);
      vertices_index++;
    }
  }

  return UPolytope(&vertices, upolytope.flatten().subspace_dimensions(), polytopes, device);
}

syrenn_server::UPolytope UPolytope::Serialize() {
  FlushPending();
  auto serialized = syrenn_server::UPolytope();
  auto serialized_upolytope = serialized.mutable_compressed();

  serialized_upolytope->set_space_dimensions(space_dimensions());
  serialized_upolytope->set_subspace_dimensions(subspace_dimensions_);
  serialized_upolytope->set_num_vertices(n_vertices());

  if (include_post_) {
    if (device_ != SYRENN_DEVICE_CPU) {
      if (!(classified_ && vertices_.size() == vertices_device_.size())) {
        vertices_ = vertices_device_.eigen();
      }
    }
    for (size_t i = 0; i < vertices_.rows(); i++) {
      for (size_t dim = 0; dim < vertices_.cols(); dim++) {
        serialized_upolytope->add_vertices(vertices_(i,dim));
      }
    }
  }

  if (device_ != SYRENN_DEVICE_CPU) {
    combinations_ = combinations_device_.eigen();
  }

  for (size_t i = 0; i < combinations_.rows(); i++) {
    for (size_t dim = 0; dim < combinations_.cols(); dim++) {
      serialized_upolytope->add_combinations(combinations_(i,dim));
    }
  }

  for (size_t face_idx = 0; face_idx < n_polytopes(); face_idx++) {
    if (classified_) {
      serialized_upolytope->add_labels(labels_(face_idx));
    }

    auto serialized_vpolytope = serialized_upolytope->add_polytopes();
    for (size_t v_idx: polytopes_[face_idx]) {
      serialized_vpolytope->add_vertex_idxes(v_idx);
    }

  }

  return serialized;
}

void UPolytope::FlushPending() {
  if (!pending_.empty()) {
    size_t n_old = vertices_.rows();
    size_t n_pending = pending_.size();
    vertices_.conservativeResize(n_old + n_pending, vertices_.cols());
    combinations_.conservativeResize(n_old + n_pending, combinations_.cols());
    for (size_t i = 0; i < n_pending; i++) {
      vertices_.row(n_old + i) = pending_[i].vertex;
      combinations_.row(n_old + i) = pending_[i].combination;
    }
    pending_.clear();
  }
}

RMMatrixXf &UPolytope::vertices() {
  FlushPending();
  return vertices_;
}

RMMatrixXfDevice &UPolytope::vertices_device() {
  return vertices_device_;
}

RMMatrixXf &UPolytope::combinations() {
  FlushPending();
  return combinations_;
}

RMMatrixXfDevice &UPolytope::combinations_device() {
  return combinations_device_;
}

std::vector<size_t> &UPolytope::vertex_indices(size_t polytope) {
  return polytopes_[polytope];
}

bool UPolytope::is_counter_clockwise() const {
  return subspace_dimensions_ == 2;
}

size_t UPolytope::space_dimensions() const {
  if (device_ == SYRENN_DEVICE_CPU) {
    return vertices_.cols();
  } else {
    return vertices_device_.cols();
  }
}

size_t UPolytope::n_polytopes() const {
  return polytopes_.size();
}

size_t UPolytope::n_vertices(size_t polytope) const {
  return polytopes_[polytope].size();
}

size_t UPolytope::n_vertices() const {
  if (device_ == SYRENN_DEVICE_CPU) {
    return vertices_.rows() + pending_.size();
  } else {
    return vertices_device_.rows();
  }
}

size_t UPolytope::vertex_index(size_t polytope, size_t vertex) const {
  return polytopes_[polytope][vertex];
}

Eigen::Ref<const RMVectorXf> UPolytope::vertex(size_t raw_index) const {
  if (raw_index >= static_cast<size_t>(vertices_.rows())) {
    return pending_[raw_index - vertices_.rows()].vertex;
  }
  return vertices_.row(raw_index);
}

Eigen::Ref<const RMVectorXf> UPolytope::vertex(size_t polytope,
                                               size_t vertex) const {
  return this->vertex(vertex_index(polytope, vertex));
}

Eigen::Ref<const RMVectorXf> UPolytope::combination(size_t raw_index) const {
  if (raw_index >= static_cast<size_t>(combinations_.rows())) {
    return pending_[raw_index - combinations_.rows()].combination;
  }
  return combinations_.row(raw_index);
}

Eigen::Ref<const RMVectorXf> UPolytope::combination(size_t polytope,
                                                    size_t vertex) const {
  return combination(vertex_index(polytope, vertex));
}

size_t UPolytope::AppendVertex(RMVectorXf *vertex, RMVectorXf *combination) {
  auto iterator = pending_.emplace_back(vertex, combination);
  return vertices_.rows() + std::distance(pending_.begin(), iterator);
}

size_t UPolytope::AppendPolytope(std::vector<size_t> *vertex_indices) {
  auto iterator = polytopes_.emplace_back(std::move(*vertex_indices));
  // See AppendVertex above about this hack.
  size_t index = polytopes_.size() - 1;
  auto iterator2 = polytopes_.begin() + index;
  for (; iterator2 != iterator; index--, iterator2--) {}
  return index;
}

size_t UPolytope::n_edges() const {
  return edges_.size();
}

// size_t UPolytope::n_faces() const {
//   return faces_.size();
// }

bool UPolytope::HasEdge(size_t v1_idx, size_t v2_idx) {
  if (v1_idx > v2_idx) { std::swap(v1_idx, v2_idx); }
  EdgeMap::accessor a;
  if (edge_map_.find(a, std::make_pair(v1_idx, v2_idx))) {
    return true;
  }
  return false;
}

size_t UPolytope::edge_idx(size_t v1_idx, size_t v2_idx) {
  if (v1_idx > v2_idx) { std::swap(v1_idx, v2_idx); }
  EdgeMap::accessor a;
  if (edge_map_.find(a, std::make_pair(v1_idx, v2_idx))) {
    return a->second;
  }
  // TODO(zhetao): Revise the error-handling here.
  std::ostringstream buf;
  buf << "error: no edge (" << v1_idx << "," << v2_idx << ")\n";
  std::cerr << buf.str();
  assert (false);
}

Edge & UPolytope::edge(size_t edge_idx) {
  return edges_[edge_idx];
}

Edge & UPolytope::edge(size_t v1_idx, size_t v2_idx) {
  return edges_[edge_idx(v1_idx, v2_idx)];
}

int UPolytope::GetAttachedFace(size_t edge_idx, size_t idx2) {
#ifdef DEBUG_TRANSFORM
  assert(idx2 < 2);
#endif
  return faces_of_edges_[edge_idx * 2 + idx2];
}

int & UPolytope::GetAttachedFaceRef(size_t edge_idx, size_t idx2) {
#ifdef DEBUG_TRANSFORM
  assert(idx2 < 2);
#endif
  return faces_of_edges_[edge_idx * 2 + idx2];
}

void UPolytope::AttachEdgeToFace(size_t edge_idx, size_t face_idx) {
  for (size_t i = 0; i < 2; i++) {
    if (faces_of_edges_[edge_idx * 2 + i] == -1) {
      faces_of_edges_[edge_idx * 2 + i] = face_idx;
      return;
    }
  }
  assert (false);
}
void UPolytope::AttachEdgeToFaces(size_t edge_idx, size_t face1_idx, size_t face2_idx) {
  faces_of_edges_[edge_idx * 2] = face1_idx;
  faces_of_edges_[edge_idx * 2 + 1] = face2_idx;
  return;
}

void UPolytope::DetachEdgeFromFace(size_t edge_idx, size_t face_idx) {
  for (size_t i = 0; i < 2; i++) {
    if (faces_of_edges_[edge_idx * 2 + i] == face_idx) {
      faces_of_edges_[edge_idx * 2 + i] = -1;
      return;
    }
  }
  assert (false);
}

void UPolytope::UpdateEdgeToFace(size_t edge_idx, size_t old_face_idx, size_t new_face_idx) {
  for (size_t i = 0; i < 2; i++) {
    if (faces_of_edges_[edge_idx * 2 + i] == old_face_idx) {
      faces_of_edges_[edge_idx * 2 + i] = new_face_idx;
      return;
    }
  }
  assert (false);
}

size_t UPolytope::AppendEdge(size_t v1_idx, size_t v2_idx) {
  // assert (v1 != v2);
  if (v1_idx > v2_idx) { std::swap(v1_idx, v2_idx); }

  EdgeMap::accessor a;
  if (edge_map_.find(a, std::make_pair(v1_idx, v2_idx))) {
    return a->second;
  }

  // assert (!(edge_map_.find(a, std::make_pair(v1, v2))));
  if (edge_map_.insert(a, std::make_pair(v1_idx, v2_idx))) {
    auto edge_it = edges_.emplace_back(Edge(v1_idx, v2_idx));
    a->second = std::distance(edges_.begin(), edge_it);
    if (faces_of_edges_.size() < n_edges() * 2) {
      faces_of_edges_.resize(n_edges() * 2);
    }
  }


#if defined(DEBUG_TRANSFORM) && defined(debug_v1_idx) && defined(debug_v2_idx)
  std::ostringstream buf;
  if (v1_idx == debug_v1_idx &&
      v2_idx == debug_v2_idx) {
    buf << "+++ appended edge (vert " << debug_v1_idx << ",vert " << debug_v2_idx << ") as " << a->second << std::endl;
  }
  std::cerr << buf.str();
#endif

  return a->second;
}


void UPolytope::UpdateEdge(size_t edge_idx, size_t new_v1_idx, size_t new_v2_idx) {
  if (new_v1_idx > new_v2_idx) { std::swap(new_v1_idx, new_v2_idx); }

  Edge & old_edge =  edge(edge_idx);
  size_t old_v1_idx = old_edge.v1_idx;
  size_t old_v2_idx = old_edge.v2_idx;
  if (old_v1_idx > old_v2_idx) { std::swap(old_v1_idx, old_v2_idx); }

  if (new_v1_idx == old_v1_idx && new_v2_idx == old_v2_idx) {
    return;
  }

#if defined(DEBUG_TRANSFORM) && defined(debug_v1_idx) && defined(debug_v2_idx)
  std::ostringstream buf;
  if (old_v1_idx == debug_v1_idx &&
      old_v2_idx == debug_v2_idx) {
    buf << "--- splitted edge (vert " << debug_v1_idx << ",vert " << debug_v2_idx << ")\n";
  }
  std::cerr << buf.str();
#endif

  EdgeMap::accessor a;
  if (edge_map_.insert(a, std::make_pair(new_v1_idx, new_v2_idx))) {
    edge_map_.erase(std::make_pair(old_v1_idx, old_v2_idx));
    old_edge.v1_idx = new_v1_idx;
    old_edge.v2_idx = new_v2_idx;
    a->second = edge_idx;
  }
  // NOTE(zhetao): handling them while computing intersection as an optimization
  // UpdateEndpoints(a->second); // of old edge_idx
}

size_t UPolytope::SplitEdge(size_t edge_idx, size_t vertex_idx) {
  size_t v1_idx = edge(edge_idx).v1_idx;
  size_t v2_idx = edge(edge_idx).v2_idx;
  if (v1_idx > v2_idx) { std::swap(v1_idx, v2_idx); }

  UpdateEdge(edge_idx, v1_idx, vertex_idx);

#if defined(DEBUG_TRANSFORM) && defined(debug_v1_idx) && defined(debug_v2_idx)
  std::ostringstream buf;
  if (vertex_idx == debug_v1_idx && v2_idx == debug_v2_idx ||
      vertex_idx == debug_v2_idx && v2_idx == debug_v1_idx) {
    buf << "+++ appended edge (vert " << debug_v1_idx << ",vert " << debug_v2_idx << ") by split\n";
  }
  std::cerr << buf.str();
#endif

  const size_t new_edge_idx = AppendEdge(vertex_idx, v2_idx);
  AttachEdgeToFaces(new_edge_idx, GetAttachedFace(edge_idx, 0), GetAttachedFace(edge_idx, 1));

  return new_edge_idx;
}

size_t UPolytope::NewFace() {
  auto it = polytopes_.emplace_back(std::vector<size_t>());
  return std::distance(polytopes_.begin(), it);
  // auto it = faces_.emplace_back(Face());
  // return std::distance(faces_.begin(), it);
}

Face UPolytope::face(size_t idx) {
  return Face(polytopes_[idx]);
  // return faces_[idx];
}

size_t Face::n_vertices() const {
  return vertex_idxes.size();
}

size_t Face::last_vertex_idx(size_t idx2) const {
  // assert (idx2 < n_vertices());
  return vertex_idxes[(idx2 + vertex_idxes.size() - 1) % vertex_idxes.size()];
}

size_t Face::vertex_idx(size_t idx2) const {
  // assert (idx2 < n_vertices());
  return vertex_idxes[idx2];
}

bool Face::empty() const { return n_vertices() == 0; }

std::ostream& operator<<(std::ostream& os, const Edge& edge) {
  os << "(" << edge.v1_idx << "," << edge.v2_idx << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Face& face) {
  os << "{";
  for (size_t v_idx: face.vertex_idxes) { os << v_idx << ","; }
  os << "}";
  return os;
}

std::vector<size_t> & UPolytope::endpoints_host() {

  const size_t n_endpoints = n_edges() * 2;
  const size_t n_new_edges = n_edges() - n_finished_edges_;
  const size_t n_new_endpoints = n_new_edges * 2;

#ifdef TIME_INTERSECTION
  Timer t;
#endif

  if (endpoints_host_.size() < n_endpoints) {
    endpoints_host_.resize(n_endpoints);
  }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time reserve device space: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  // if (new_endpoints_.size() < n_new_endpoints) {
  //   new_endpoints_.resize(n_new_endpoints);
  // }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time reserve host space: " << t.Ticks() << std::endl;
  t.Reset();
  std::cerr << "working on n new edges: " << n_new_edges << std::endl;
#endif

  const size_t new_endpoints_base = n_finished_edges_ * 2;
  for (size_t i = 0; i < n_new_edges; i++) {
    endpoints_host_[new_endpoints_base + i * 2] = edges_[n_finished_edges_ + i].v1_idx;
    endpoints_host_[new_endpoints_base + i * 2 + 1] = edges_[n_finished_edges_ + i].v2_idx;
  }

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time setup: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  // thrust::copy(new_endpoints_.begin(),
  //              new_endpoints_.begin() + n_new_endpoints,
  //              endpoints_host_.begin() + n_finished_edges_ * 2);
  n_finished_edges_ = n_edges();

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>> time copy: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  return endpoints_host_;
}
