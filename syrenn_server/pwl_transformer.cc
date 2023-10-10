#include "syrenn_server/pwl_transformer.h"
#include <assert.h>
#include <algorithm>
#include <utility>
#include <stack>
#include <vector>
#include <memory>
#include "tbb/tbb.h"
#include "eigen3/Eigen/Dense"
#include<thread>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/partition.h>
// #include "mkldnn.hpp"

namespace {
void Interpolate(const Eigen::Ref<const RMVectorXf> from_point,
                 const Eigen::Ref<const RMVectorXf> to_point,
                 const double ratio, RMVectorXf *out) {
  out->noalias() = ((1.0 - ratio) * from_point) + (ratio * to_point);
}
}  // namespace

class PWLTransformer::ParallelPlaneTransformer {
 public:
  ParallelPlaneTransformer(const PWLTransformer &layer, UPolytope *inout)
      : layer_(layer), inout_(inout), inserted_points_(new NewPointsMemo()) {}

  size_t MaybeIntersectEdge(IntersectionPointMetadata key) const {
    NewPointsMemo::accessor a;
    if (inserted_points_->insert(a, key)) {
      // Returns true if item is new.
      double crossing_ratio =
        layer_.CrossingRatio(inout_->vertex(key.min_index),
                             inout_->vertex(key.max_index),
                             key.face);
      RMVectorXf vertex;
      RMVectorXf combination;

      Interpolate(inout_->vertex(key.min_index),
                  inout_->vertex(key.max_index),
                  crossing_ratio, &vertex);
      Interpolate(inout_->combination(key.min_index),
                  inout_->combination(key.max_index),
                  crossing_ratio, &combination);
      a->second = inout_->AppendVertex(&vertex, &combination);
    }
    return a->second;
  }

  void operator()(PolytopeMetadata &current_split,
                  tbb::parallel_do_feeder<PolytopeMetadata> &feeder) const {
    int polytope = current_split.polytope_index;
    std::vector<size_t> &possible_faces = current_split.remaining_faces;
    while (!possible_faces.empty()) {
      int n_vertices = inout_->n_vertices(polytope);
      int sign = 0, i = -1, j = -1, split_face = -1;
      for (size_t face_i = 0; face_i < possible_faces.size(); face_i++) {
        bool active = false;
        int face = possible_faces[face_i];
        sign = 0;
        for (i = 0; i < n_vertices; i++) {
          int i_sign = layer_.PointSign(inout_->vertex(polytope, i), face);
          if (i_sign == 0) {
            continue;
          }
          if (sign == 0) {
            sign = i_sign;
            continue;
          }
          if (sign != i_sign) {
            sign = i_sign;
            break;
          }
        }
        if (sign == 0 || i == n_vertices) {
          continue;
        }
        active = layer_.IsFaceActive(inout_->vertex(polytope, i - 1),
                                     inout_->vertex(polytope, i),
                                     face);
        for (j = i + 1; j < n_vertices; j++) {
          int j_sign = layer_.PointSign(inout_->vertex(polytope, j), face);
          if (sign < 0 && j_sign > 0) {
            break;
          } else if (sign > 0 && j_sign < 0) {
            break;
          }
        }
        active = active ||
          layer_.IsFaceActive(inout_->vertex(polytope, j - 1),
                              inout_->vertex(polytope, j % n_vertices),
                              face);
        if (!active) {
          continue;
        }
        split_face = face;
        possible_faces.erase(possible_faces.begin() + face_i);
        break;
      }

      if (split_face == -1) {
        // It's all done!
        return;
      }

      // Now, i is the first vertex with a sign of 'sign,' while j is the first
      // vertex with a sign of '\not sign'.
      IntersectionPointMetadata i_metadata =
        IntersectionPointMetadata(inout_->vertex_index(polytope, i - 1),
                                  inout_->vertex_index(polytope, i),
                                  split_face);
      size_t cross_i_index = MaybeIntersectEdge(i_metadata);
      IntersectionPointMetadata j_metadata =
        IntersectionPointMetadata(inout_->vertex_index(polytope, j - 1),
                                  inout_->vertex_index(polytope,
                                                       j % n_vertices),
                                  split_face);
      size_t cross_j_index = MaybeIntersectEdge(j_metadata);

      // First, we add the intersection points to the existing polytope (which
      // will end up being the 'top' one).
      std::vector<size_t> &top_vertices = inout_->vertex_indices(polytope);
      // NOTE: The order of these insertions is important.
      top_vertices.insert(top_vertices.begin() + i, cross_i_index);
      j++;
      top_vertices.insert(top_vertices.begin() + j, cross_j_index);

      // Now we ``steal'' vertices from the top to put in the bottom.
      std::vector<size_t> bottom_vertices;
      bottom_vertices.reserve(3 + (n_vertices - (j - i)));
      for (size_t v = 0, o = 0; v < top_vertices.size(); o++) {
        if (o <= static_cast<size_t>(i) || o >= static_cast<size_t>(j)) {
          bottom_vertices.push_back(top_vertices[v]);
        }
        if (o < static_cast<size_t>(i) || o > static_cast<size_t>(j)) {
          // These should not be in top_vertices.
          top_vertices.erase(top_vertices.begin() + v);
        } else {
          v++;
        }
      }

      size_t new_polytope = inout_->AppendPolytope(&bottom_vertices);
      feeder.add(PolytopeMetadata(new_polytope, possible_faces));
    }
  }

 private:
  const PWLTransformer &layer_;
  UPolytope *inout_;
  // We use a pointer here because the operator() must be const to work with
  // TBB parallel_do, so we cannot directly modify any member variables.
  std::unique_ptr<NewPointsMemo> inserted_points_;
};

bool PWLTransformer::IsFaceActive(Eigen::Ref<const RMVectorXf> from,
                                  Eigen::Ref<const RMVectorXf> to,
                                  const size_t face) const {
  // This is a safe default.
  return true;
}

bool PWLTransformer::IsFaceActive(const RMMatrixXfDevice &vertices,
                                  const size_t from_idx,
                                  const size_t to_idx,
                                  const size_t face) const {
  // This is a safe default.
  return true;
}

void PWLTransformer::EnumerateLineIntersections(
        Eigen::Ref<const RMVectorXf> from_point,
        Eigen::Ref<const RMVectorXf> to_point,
        double from_distance, double to_distance,
        std::vector<double> *new_endpoints) const {
  double delta = to_distance - from_distance;

  std::vector<double> crossing_distances;
  for (size_t i = 0; i < n_piece_faces(to_point.size()); i++) {
    if ((PointSign(to_point, i) * PointSign(from_point, i)) < 0 &&
        IsFaceActive(from_point, to_point, i)) {
      // The points lie in different linear regions, so we need to add an
      // endpoint where they cross this face separating the linear regions.
      // This is the distance between from_distance and to_distance
      double crossing_distance = CrossingRatio(from_point, to_point, i);
      new_endpoints->emplace_back(
          from_distance + (crossing_distance * delta));
    }
  }
}

std::vector<double> PWLTransformer::ProposeLineEndpoints(
    const SegmentedLine &line) const {
  const RMMatrixXf &points = line.points();
  size_t n_segments = line.Size() - 1;

  // NOTE(masotoud): we could use a tbb::concurrent_set here to avoid the merge
  // overhead, or std::vector<std::set<>> to avoid the sort overhead, but my
  // guess is that this those will probably come with too much overhead of
  // their own.
  std::vector<std::vector<double>> segment_endpoints(n_segments);

  tbb::parallel_for(size_t(0), n_segments, [&](size_t i) {
    EnumerateLineIntersections(
            points.row(i), points.row(i + 1),
            line.endpoint_ratio(i), line.endpoint_ratio(i + 1),
            &(segment_endpoints[i]));
    std::sort(segment_endpoints[i].begin(), segment_endpoints[i].end());
  });

  // TODO(masotoud): Perhaps we shouldn't flatten, and just let
  // AddEndpointsThresholded take in the multi-dimensional segment_endpoints?
  std::vector<double> endpoints;
  for (auto &single_segment_endpoints : segment_endpoints) {
    endpoints.insert(endpoints.end(), single_segment_endpoints.begin(),
                     single_segment_endpoints.end());
  }
  segment_endpoints.clear();

  return endpoints;
}

class PWLTransformer::ParallelSplitPlane {
 public:
  ParallelSplitPlane(
    const PWLTransformer &layer,
    UPolytope *inout,
    const size_t plane_idx,
    const std::vector<int> & point_sign,
    const std::map<std::pair<size_t, size_t>, size_t> & intersection_map,
    const std::vector<int> & intersected_faces)
    : layer(layer), inout(inout), plane_idx(plane_idx),
      point_sign(point_sign), intersection_map(intersection_map),
      intersected_faces(intersected_faces) {}

  void FindSplitIndexes(int split_idxes[],
                        const size_t face_idx,
                        const Face& face) const {
    const size_t n_vertices = face.n_vertices();
    size_t i, j;
    int i_sign, j_sign, sign = 0;
    for (i = 0; i < n_vertices; i++) {
      i_sign = point_sign[face.vertex_idx(i)];
      if (i_sign == 0) {
        continue;
      } if (sign == 0) {
        sign = i_sign;
        continue;
      }
      if (sign != i_sign) {
        sign = i_sign;
        split_idxes[0] = i;
        break;
      }
    }
    if (sign == 0 || i == n_vertices) {
      return;
    }
    for(j = i + 1; j < n_vertices; j++) {
      j_sign = point_sign[face.vertex_idx(j)];
      if (sign < 0 && j_sign > 0) {
        break;
      } else if (sign > 0 && j_sign < 0) {
        break;
      }
    }
    split_idxes[1] = j % n_vertices;
  }

  void FindSplitVertices(size_t split_vertices[],
                         const int split_idxes[],
                         bool split_vertices_new[],
                         const size_t face_idx,
                         const Face& face) const {
    for (size_t i = 0; i < 2; i++) {
      size_t v1_idx = face.last_vertex_idx(split_idxes[i]);
      size_t v2_idx = face.vertex_idx(split_idxes[i]);

      if (point_sign[v1_idx] == 0) {
        split_vertices[i] = v1_idx;
        split_vertices_new[i] = false;
      } else if(point_sign[v2_idx] == 0) {
        split_vertices[i] = v2_idx;
        split_vertices_new[i] = false;
      } else {
        if (v1_idx > v2_idx) { std::swap(v1_idx, v2_idx); }
#ifdef ENABLE_ASSERTIONS
        if (intersection_map.find(std::make_pair(v1_idx, v2_idx)) == intersection_map.end()) {
          std::ostringstream buf;
          buf << "error: no intersection on edge (vert " << v1_idx << ",vert " << v2_idx << ")\n";
          buf << "error: has this edge: " << (inout->HasEdge(v1_idx, v2_idx) ? "true" : "false") << "\n";

          for(int i=0;i<2;i++)buf<<"@"<<i<<":idx"<<split_idxes[i]<<":vert"<<split_vertices[i]<<":"<<split_vertices_new[i]<<std::endl;
          buf << "face " << face_idx << std::endl;
          buf << "1st endpoint: idx " << (split_idxes[i] + face.n_vertices() - 1) % face.n_vertices() << std::endl;
          buf << "2nd endpoint: idx " << split_idxes[i] << std::endl;
          for (size_t i = 0; i < face.n_vertices(); i++) {
            buf
            << ((face.vertex_idx(i) == v1_idx || face.vertex_idx(i) == v2_idx) ? "-> vert " : "  vert ")
            << "  vert "
            << face.vertex_idx(i) << ":" << point_sign[face.vertex_idx(i)] << "\n";
          }
          std::cerr << buf.str();
          assert(false);
        }
#endif
        split_vertices[i] = intersection_map.find(std::make_pair(v1_idx, v2_idx))->second;
        split_vertices_new[i] = true;
      }
    }
  }

  void SplitPlane(const size_t split_vertices[],
                  const int split_idxes[],
                  const bool split_vertices_new[],
                  const size_t face_idx,
                  Face& face) const {

    const size_t new_face_idx = inout->NewFace();
    Face new_face = inout->face(new_face_idx);

    const size_t split_edge_idx = inout->AppendEdge(split_vertices[0], split_vertices[1]);
    // NOTE(zhetao): One face_idx will later be updated to new_face_idx.
    inout->AttachEdgeToFaces(split_edge_idx, face_idx, face_idx);

#ifdef DEBUG_TRANSFORM
#if defined(debug_v1_idx) && defined(debug_v2_idx)
    if (split_vertices[0] == debug_v1_idx && split_vertices[1] == debug_v2_idx ||
        split_vertices[0] == debug_v2_idx && split_vertices[1] == debug_v1_idx) {
      debug_this = true;
    }
#endif
#if defined(debug_face_idx)
    if (face_idx == debug_face_idx) {
      std::cerr << "+++ splitting debugging face " << debug_face_idx << "\n";
      debug_this = true;
    }
    if (new_face_idx == debug_face_idx) {
      std::cerr << "+++ added debugging face " << debug_face_idx << "\n";
      debug_this = true;
    }
#endif
#if defined(debug_v1_idx) && defined(debug_v2_idx) && defined(debug_face_idx)

    // if (point_sign[face.vertex_idx(split_idxes[0])] == 0 ||
    //     point_sign[face.vertex_idx(split_idxes[1])] == 0 ||
    //     point_sign[face.last_vertex_idx(split_idxes[0])] == 0 ||
    //     point_sign[face.last_vertex_idx(split_idxes[1])] == 0) {
    //   debug_this = true;
    // }

    if (debug_this) {
      std::cerr << ( face_idx == debug_face_idx ? "* face " : "  face ") << face_idx << "\n";
      for (size_t i = 0; i < face.n_vertices(); i++) {
        std::cerr
          << ((face.vertex_idx(i) == debug_v1_idx || face.vertex_idx(i) == debug_v2_idx) ? "*" : " ")
          << ((i == split_idxes[0] || i == split_idxes[1]) ? (
            (i == split_idxes[0]) ? "@0 " : "@1 "
          ) : "   ")
          << "vert " << face.vertex_idx(i)
          << ":" << point_sign[face.vertex_idx(i)]
          << "\n";
      }
      for (int i=0;i<2;i++) std::cerr<<"@"<<i<<":"<<split_idxes[i]<<":"<<split_vertices[i]<<":"<<split_vertices_new[i]<<std::endl;
    }
#endif
#endif

    // new, new: hi::lo::1::2::hi, mi::2::1
    // old, new: hi::lo::1::2::hi, mi::2
    // new, old: hi::lo::1::hi   , mi::2::1
    // old, old: hi::lo::1::hi   , mi::2
    std::vector<size_t>&vs = face.vertex_idxes;

// for(auto x:vs) std::cerr << x << ":" << point_sign[x] << "\n"; std::cerr << " --> original\n";

    // size_t mi_extra_space = (split_vertices_new[0] == 0) ? 1 : 2;
    std::vector<size_t> mi(vs.begin() + split_idxes[0],
                           vs.begin() + split_idxes[1]);
    // if (debug_this) for(auto x:mi) std::cerr << x << ","; std::cerr << " --> mi\n";
    // mi.reserve(mi_extra_space);
    mi.reserve(2);
    if (split_vertices_new[1] || point_sign[face.vertex_idx(split_idxes[1])] == 0) { mi.emplace_back(split_vertices[1]); }
    if (split_vertices_new[0] || point_sign[face.vertex_idx(split_idxes[0])] != 0) { mi.emplace_back(split_vertices[0]); }
    new_face.vertex_idxes.swap(mi);

    // size_t hi_lo_extra_space = (split_vertices_new[1] == 0) ? 1 : 2;
    std::vector<size_t> hi_lo;
    // hi_lo.reserve(vs.size() - (split_idxes[1] - split_idxes[0]) + hi_lo_extra_space);
    hi_lo.reserve(2);
    hi_lo.resize(vs.size() - (split_idxes[1] - split_idxes[0]));
    std::copy(vs.begin() + split_idxes[1],
              vs.end(),
              hi_lo.begin());
    std::copy(vs.begin(),
              vs.begin() + split_idxes[0],
              hi_lo.begin() + (vs.size() - split_idxes[1]) );
    // if (debug_this) for(auto x:hi_lo) std::cerr << x << ","; std::cerr << " --> hi_lo\n";
    if (split_vertices_new[0] || point_sign[face.vertex_idx(split_idxes[0])] == 0) { hi_lo.emplace_back(split_vertices[0]); }
    if (split_vertices_new[1] || point_sign[face.vertex_idx(split_idxes[1])] != 0) { hi_lo.emplace_back(split_vertices[1]); }
    face.vertex_idxes.swap(hi_lo);

    for (size_t i = 0; i < new_face.n_vertices(); i++) {
#ifdef ENBALE_ASSERTION
      assert(inout->HasEdge(new_face.last_vertex_idx(i), new_face.vertex_idx(i)));
#endif
      inout->UpdateEdgeToFace(inout->edge_idx(new_face.last_vertex_idx(i), new_face.vertex_idx(i)), face_idx, new_face_idx);
    }

#ifdef DEBUG_TRANSFORM
    if (debug_this) {
      std::cerr << ( face_idx == debug_face_idx ? "* now face " : "  now face ") << face_idx << "\n";
      for (size_t i = 0; i < face.n_vertices(); i++) {
        std::cerr << ((face.vertex_idx(i) == debug_v1_idx || face.vertex_idx(i) == debug_v2_idx) ? "* vert " : "  vert ")
          << face.vertex_idx(i) << ":" << point_sign[face.vertex_idx(i)] << "\n";
      }
      std::cerr << ( new_face_idx == debug_face_idx ? "* now new face " : "  now new face ") << new_face_idx << "\n";
      for (size_t i = 0; i < new_face.n_vertices(); i++) {
        std::cerr << ((new_face.vertex_idx(i) == debug_v1_idx || new_face.vertex_idx(i) == debug_v2_idx) ? "* vert " : "  vert ")
          << new_face.vertex_idx(i) << ":" << point_sign[new_face.vertex_idx(i)] << "\n";
      }
    }
#endif

  }

  void IsConvex(const size_t face_idx) const {
    assert (false);
    const Face face = inout->face(face_idx);
    std::vector<int> signs(face.n_vertices());
    for (size_t i = 0; i < face.n_vertices(); i++) {
      signs[i] = point_sign[face.vertex_idx(i)];
    }
    auto it = std::unique(signs.begin(), signs.end());
    size_t size = std::distance(signs.begin(), it);
    // signs.resize(size);
    if (signs[0] == signs[size-1]) { size--; }
    if (size > 3) {
      std::cerr << "Error: invalid non-convex face " << face_idx << "\n";
      for(auto v:face.vertex_idxes) {
        std::cerr
          << "vert " << v
          << ":"     << point_sign[v]
          << ":"     << inout->vertices_device().at(v, plane_idx)
          << ":"     << layer.PointSign(inout->vertices_device(), v, plane_idx)
          << "\n";
      }
      assert (false);
    }
  }

  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t face_idx2 = r.begin(); face_idx2 < r.end(); face_idx2++) {
      (*this)(face_idx2);
    }
  }

  void operator()(const size_t face_idx2) const {
    const size_t face_idx = intersected_faces[face_idx2];
    Face face = inout->face(face_idx);
    // assert (!face.empty());

#ifdef DEBUG_TRANSFORM
    IsConvex(face_idx);
#endif

    int split_idxes[2] = { -1, -1 };
    bool split_vertices_new[2] = { false, false };
    size_t split_vertices[2] = {0, 0};

    FindSplitIndexes(split_idxes, face_idx, face);

    if (split_idxes[1] == -1) {
      // std::ostringstream buf; buf << "face " << face_idx << std::endl; for
      // (size_t i = 0; i < face.n_vertices(); i++) {buf << "  vert " <<
      // face.vertex_idx(i) << ":" << point_sign[face.vertex_idx(i)] << "\n";
      // }
      // std::cerr << buf.str(); assert (false);

      // NOTE(zhetao): Only possible if one/few consecutive vertices are on the
      // plane and others are on the same side.
      // assert (false);
      return;
    }

    if (split_idxes[1] == 0) {
      std::swap(split_idxes[0], split_idxes[1]);
    }

    FindSplitVertices(split_vertices, split_idxes, split_vertices_new, face_idx, face);

    SplitPlane(split_vertices, split_idxes, split_vertices_new, face_idx, face);

    return;
  }

 private:
  const PWLTransformer &layer;
  UPolytope *inout;
  const size_t plane_idx;
  const std::vector<int> & point_sign;
  const std::map<std::pair<size_t, size_t>, size_t> & intersection_map;
  const std::vector<int> & intersected_faces;
#ifdef DEBUG_TRANSFORM
  mutable bool debug_this = false;
#endif
};

class PWLTransformer::ParallelSplitEdge {
public:
  ParallelSplitEdge(const PWLTransformer &layer,
                    UPolytope *inout,
                    const std::vector<size_t> intersected_edge_idxes,
                    const std::vector<size_t> intersect_vertex_idxes,
                    const size_t n_original_vertices)
  : layer(layer), inout(inout),
    intersected_edge_idxes(intersected_edge_idxes),
    intersect_vertex_idxes(intersect_vertex_idxes),
    n_original_vertices(n_original_vertices) {}

  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i < r.end(); i++) {
      (*this)(i);
    }
  }

  void operator()(const size_t idx2) const {
    if (intersect_vertex_idxes[idx2] >= n_original_vertices) {
      inout->SplitEdge(intersected_edge_idxes[idx2], intersect_vertex_idxes[idx2]);
    }
  }

private:
  const PWLTransformer &layer;
  UPolytope *inout;
  const std::vector<size_t> intersected_edge_idxes;
  const std::vector<size_t> intersect_vertex_idxes;
  const size_t n_original_vertices;
};

void PWLTransformer::TransformUPolytopePlaneDevice(UPolytope *inout) const {
  assert(inout->is_counter_clockwise());
  last_split_scale = 0;

#ifdef TIME_TRANSFORM
  Timer t;
#endif

  for (size_t plane_idx = 0; plane_idx < n_piece_faces(inout->space_dimensions()); plane_idx++) {
    last_split_scale += inout->n_polytopes();
#ifdef DEBUG_TRANSFORM
    std::cerr << "-------- plane " << plane_idx << " ----------\n";
    std::cerr << "n vertices: " << inout->n_vertices() << "\n";
    std::cerr << "n faces   : " << inout->n_polytopes() << "\n";
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "-------- plane " << plane_idx << " ----------\n";
#endif

    size_t n_original_vertices = inout->n_vertices();
    size_t n_original_edges = inout->n_edges();
    size_t n_original_faces = inout->n_polytopes();
    size_t n_new_vertices = 0;
    size_t n_intersected_edges = 0;
    Intersect(inout, plane_idx, point_sign, intersected_edge_idxes, intersect_vertex_idxes, n_intersected_edges, n_new_vertices);

    // std::cerr << inout->vertices_device().eigen().block(0,0,3,3) << std::endl;

    // for (int i = 0; i < n_original_vertices; i++) {
    //   if (point_sign[i] != PointSign(inout->vertices_device(), i, plane_idx)) {
    //     std::cerr << "Error: wrong point sign for vertex " << i
    //       << "(" << inout->vertices_device().at(i, plane_idx)
    //       << "), got " << point_sign[i] << ", should be "
    //       << PointSign(inout->vertices_device(), i, plane_idx) << std::endl;
    //     assert(false);
    //   }
    // }

#ifdef DEBUG_TRANSFORM
    for (int i = 0; i < n_original_vertices; i++) {
      if (point_sign[i] != PointSign(inout->vertices_device(), i, plane_idx)) {
        std::cerr << "Error: wrong point sign for vertex " << i
          << "(" << inout->vertices_device().at(i, plane_idx)
          << "), got " << point_sign[i] << ", should be "
          << PointSign(inout->vertices_device(), i, plane_idx) << std::endl;
      }
    }
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "time interpolation for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif

    // Read-only after sequential initialization
    std::map<std::pair<size_t, size_t>, size_t> intersection_map;
    for (size_t i = 0; i < n_new_vertices; i++) {
      Edge & edge = inout->edge(intersected_edge_idxes[i]);
      size_t v1_idx = edge.v1_idx;
      size_t v2_idx = edge.v2_idx;
      if (v1_idx > v2_idx) { std::swap(v1_idx, v2_idx); }
      intersection_map[std::make_pair(v1_idx, v2_idx)] = intersect_vertex_idxes[i];

#ifdef DEBUG_TRANSFORM
      if (plane_idx == debug_plane_idx) {
        if (v1_idx == debug_v1_idx && v2_idx == debug_v2_idx) {
          std::cerr << "has intersection\n";
        }
      }
#endif

    }

#ifdef TIME_TRANSFORM
    std::cerr << "time prepare map   for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif

#ifdef PARALLEL_TRANSFORM
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n_new_vertices, TBB_GRAINSIZE_SPLIT),
      ParallelSplitEdge(*this, inout, intersected_edge_idxes, intersect_vertex_idxes, n_original_vertices));
#else
    for (int i = 0; i < n_new_vertices; i++) {
      if (intersect_vertex_idxes[i] >= n_original_vertices) {
        inout->SplitEdge(intersected_edge_idxes[i], intersect_vertex_idxes[i]);
      }
    }
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "time split edges   for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif

#ifdef DEBUG_TRANSFORM
  if (debug_face_idx < n_original_faces) {
    Face& face = inout->face(debug_face_idx);
    std::cerr << "* face " << debug_face_idx << "\n";
    for (size_t i = 0; i < face.n_vertices(); i++) {
      std::cerr << ((face.vertex_idx(i) == debug_v1_idx || face.vertex_idx(i) == debug_v2_idx) ? "* vert " : "  vert ")
        << face.vertex_idx(i) << ":" << point_sign[face.vertex_idx(i)] << "\n";
    }
  }
#endif


    // FIXME(zhetao): Suppose Interset returns all intersected edges.
    const size_t n_possible_faces = n_intersected_edges * 2;
    if (intersected_faces.size() < n_possible_faces) {
        intersected_faces.resize(n_possible_faces);
    }
    for (size_t i = 0; i < n_intersected_edges; i++) {
      const size_t edge_idx = intersected_edge_idxes[i];
      for (size_t j = 0; j < 2; j++) {
        intersected_faces[i * 2 + j] = inout->GetAttachedFace(edge_idx, j);
      }
    }
    std::sort(intersected_faces.begin(), intersected_faces.begin() + n_possible_faces);
    auto last = std::unique(intersected_faces.begin(), intersected_faces.begin() + n_possible_faces);
    const size_t n_intersected_faces = std::distance(intersected_faces.begin(), last);
    const size_t start_face_idx2 = intersected_faces[0] == -1 ? 1 : 0;

#ifdef PARALLEL_TRANSFORM
    tbb::parallel_for(
      tbb::blocked_range<size_t>(start_face_idx2, n_intersected_faces, TBB_GRAINSIZE_SPLIT),
      // tbb::blocked_range<size_t>(0, n_original_faces, 1),
      ParallelSplitPlane(*this, inout, plane_idx, point_sign, intersection_map, intersected_faces)
    );
#else
    for (size_t face_idx = start_face_idx2; face_idx < n_intersected_faces; face_idx++) {
      ParallelSplitPlane(*this, inout, plane_idx, point_sign, intersection_map, intersected_faces)(face_idx);
    }
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "time split planes  for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif
  }

#ifdef TIME_TRANSFORM
  std::cerr << "time compute: " << t.Ticks() << std::endl;
  t.Reset();
#endif

}

void PWLTransformer::TransformUPolytopePlane(UPolytope *inout) const {
  TransformUPolytopePlaneNewCPU(inout);
  // assert(inout->is_counter_clockwise());

  // std::vector<PolytopeMetadata> initial_polytopes;
  // std::vector<size_t> all_faces;
  // for (size_t i = 0; i < n_piece_faces(inout->space_dimensions()); i++) {
  //   all_faces.push_back(i);
  // }
  // for (size_t i = 0; i < inout->n_polytopes(); i++) {
  //   initial_polytopes.emplace_back(i, all_faces);
  // }
  // ParallelPlaneTransformer parallel_transformer(*this, inout);
  // tbb::parallel_do(initial_polytopes.begin(), initial_polytopes.end(),
  //                  parallel_transformer);
}

void PWLTransformer::TransformUPolytopePlaneNewCPU(UPolytope *inout) const {
  assert(inout->is_counter_clockwise());
  last_split_scale = 0;

#ifdef TIME_TRANSFORM
  Timer t;
#endif

  for (size_t plane_idx = 0; plane_idx < n_piece_faces(inout->space_dimensions()); plane_idx++) {
    last_split_scale += inout->n_polytopes();
#ifdef DEBUG_TRANSFORM
    assert(false);
    std::cerr << "-------- plane " << plane_idx << " ----------\n";
    std::cerr << "n vertices: " << inout->n_vertices() << "\n";
    std::cerr << "n faces   : " << inout->n_polytopes() << "\n";
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "-------- plane " << plane_idx << " ----------\n";
#endif

    // std::cerr << inout->vertices().block(0,0,3,3) << std::endl;

    size_t n_original_vertices = inout->n_vertices();
    size_t n_original_edges = inout->n_edges();
    size_t n_original_faces = inout->n_polytopes();
    size_t n_new_vertices = 0;
    size_t n_intersected_edges = 0;
    IntersectNewCPU(
      inout,
      plane_idx,
      point_sign,
      intersected_edge_idxes,
      intersect_vertex_idxes,
      n_intersected_edges,
      n_new_vertices
    );

    // std::cerr << inout->vertices().block(0,0,3,3) << std::endl;

#ifdef DEBUG_TRANSFORM
    assert(false);
    for (int i = 0; i < n_original_vertices; i++) {
      if (point_sign[i] != PointSign(inout->vertices_device(), i, plane_idx)) {
        std::cerr << "Error: wrong point sign for vertex " << i
          << "(" << inout->vertices_device().at(i, plane_idx)
          << "), got " << point_sign[i] << ", should be "
          << PointSign(inout->vertices_device(), i, plane_idx) << std::endl;
      }
    }
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "time interpolation for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif

    // std::cerr << "new vertices: " << n_new_vertices << std::endl;
    // Read-only after sequential initialization
    std::map<std::pair<size_t, size_t>, size_t> intersection_map;
    for (size_t i = 0; i < n_new_vertices; i++) {
      Edge & edge = inout->edge(intersected_edge_idxes[i]);
      size_t v1_idx = edge.v1_idx;
      size_t v2_idx = edge.v2_idx;
      if (v1_idx > v2_idx) { std::swap(v1_idx, v2_idx); }
      intersection_map[std::make_pair(v1_idx, v2_idx)] = intersect_vertex_idxes[i];

#ifdef DEBUG_TRANSFORM
      if (plane_idx == debug_plane_idx) {
        if (v1_idx == debug_v1_idx && v2_idx == debug_v2_idx) {
          std::cerr << "has intersection\n";
        }
      }
#endif

    }

#ifdef TIME_TRANSFORM
    std::cerr << "time prepare map   for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif

#ifdef PARALLEL_TRANSFORM
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n_new_vertices, TBB_GRAINSIZE_SPLIT),
      ParallelSplitEdge(*this, inout, intersected_edge_idxes, intersect_vertex_idxes, n_original_vertices));
#else
    assert(false);
    for (int i = 0; i < n_new_vertices; i++) {
      if (intersect_vertex_idxes[i] >= n_original_vertices) {
        inout->SplitEdge(intersected_edge_idxes[i], intersect_vertex_idxes[i]);
      }
    }
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "time split edges   for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif

#ifdef DEBUG_TRANSFORM
  if (debug_face_idx < n_original_faces) {
    Face& face = inout->face(debug_face_idx);
    std::cerr << "* face " << debug_face_idx << "\n";
    for (size_t i = 0; i < face.n_vertices(); i++) {
      std::cerr << ((face.vertex_idx(i) == debug_v1_idx || face.vertex_idx(i) == debug_v2_idx) ? "* vert " : "  vert ")
        << face.vertex_idx(i) << ":" << point_sign[face.vertex_idx(i)] << "\n";
    }
  }
#endif


    // FIXME(zhetao): Suppose Interset returns all intersected edges.
    const size_t n_possible_faces = n_intersected_edges * 2;
    if (intersected_faces.size() < n_possible_faces) {
        intersected_faces.resize(n_possible_faces);
    }
    for (size_t i = 0; i < n_intersected_edges; i++) {
      const size_t edge_idx = intersected_edge_idxes[i];
      for (size_t j = 0; j < 2; j++) {
        intersected_faces[i * 2 + j] = inout->GetAttachedFace(edge_idx, j);
      }
    }
    std::sort(intersected_faces.begin(), intersected_faces.begin() + n_possible_faces);
    auto last = std::unique(intersected_faces.begin(), intersected_faces.begin() + n_possible_faces);
    const size_t n_intersected_faces = std::distance(intersected_faces.begin(), last);
    const size_t start_face_idx2 = intersected_faces[0] == -1 ? 1 : 0;

#ifdef PARALLEL_TRANSFORM
    tbb::parallel_for(
      tbb::blocked_range<size_t>(start_face_idx2, n_intersected_faces, TBB_GRAINSIZE_SPLIT),
      // tbb::blocked_range<size_t>(0, n_original_faces, 1),
      ParallelSplitPlane(*this, inout, plane_idx, point_sign, intersection_map, intersected_faces)
    );
#else
    for (size_t face_idx = start_face_idx2; face_idx < n_intersected_faces; face_idx++) {
      ParallelSplitPlane(*this, inout, plane_idx, point_sign, intersection_map, intersected_faces)(face_idx);
    }
#endif

#ifdef TIME_TRANSFORM
    std::cerr << "time split planes  for plane" << plane_idx << ": " << t.Ticks() << std::endl;
    t.Reset();
#endif
  }

#ifdef TIME_TRANSFORM
  std::cerr << "time compute: " << t.Ticks() << std::endl;
  t.Reset();
#endif

}


void PWLTransformer::TransformUPolytopeWithoutCompute(UPolytope *inout) const {
  assert (layer_type() == "ArgMax");
  if (inout->is_counter_clockwise()) {
    int original_device = inout->device_;
    inout->to(SYRENN_DEVICE_CPU);
    TransformUPolytopePlane(inout);
    inout->to(original_device);
    return;
  }
  throw "No general-dimension transformer yet.";
}

void PWLTransformer::TransformUPolytope(UPolytope *inout) const {
  if (inout->is_counter_clockwise()) {
    if (device_ == SYRENN_DEVICE_CPU) {
      TransformUPolytopePlane(inout);
      Compute(&(inout->vertices()));
      return;
    } else {
      TransformUPolytopePlaneDevice(inout);
      ComputeDevice(&(inout->vertices_device()));
      return;
    }
  }
  throw "No general-dimension transformer yet.";
}

// Intersect CPU implementation for the new algorithm
void PWLTransformer::IntersectNewCPU(
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

  std::vector<size_t> & edge_endpoints = inout->endpoints_host();

#ifdef TIME_INTERSECTION
  std::cerr << ">>> time get endpoints: " << t.Ticks() << std::endl;
  t.Reset();
#endif

  std::vector<int> & point_sign = PointSignBatch(inout->vertices(), plane_idx);

#ifdef TIME_INTERSECTION
  std::cerr << ">>> time calc signs   : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  IntersectNewCPU(
    inout->vertices(),
    inout->combinations(),
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

class UpdateEndpointsNewCPUOp {
public:
  UpdateEndpointsNewCPUOp(
    const size_t n_todos,
    const size_t n_original_vertices,
    const size_t* todo_edge_idxes,
    const size_t* result_vertex_idxes,
    size_t* edge_endpoints
  ) : n_todos(n_todos),
      n_original_vertices(n_original_vertices),
      todo_edge_idxes(todo_edge_idxes),
      result_vertex_idxes(result_vertex_idxes),
      edge_endpoints(edge_endpoints) {}

  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i < r.end(); i++) {
      (*this)(i);
    }
  }

  void operator()(const size_t i) const {
    if (result_vertex_idxes[i] >= n_original_vertices) {
      const size_t edge_idx = todo_edge_idxes[i];
      edge_endpoints[edge_idx*2 + 1] = result_vertex_idxes[i];
    }
  }

private:
  const size_t n_todos;
  const size_t n_original_vertices;
  const size_t* todo_edge_idxes;
  const size_t* result_vertex_idxes;
  size_t* edge_endpoints;
};


class UpdateEndpointsOp {
public:
  UpdateEndpointsOp(
    const size_t n_todos,
    const size_t n_original_vertices,
    const size_t* todo_edge_idxes,
    const size_t* result_vertex_idxes,
    size_t* edge_endpoints)
  : n_todos(n_todos),
    n_original_vertices(n_original_vertices),
    todo_edge_idxes(todo_edge_idxes),
    result_vertex_idxes(result_vertex_idxes),
    edge_endpoints(edge_endpoints)
  {}

  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i < r.end(); i++) {
      (*this)(i);
    }
  }

  void operator()(const size_t i) const {
    if (result_vertex_idxes[i] >= n_original_vertices) {
      const size_t edge_idx = todo_edge_idxes[i];
      edge_endpoints[edge_idx*2 + 1] = result_vertex_idxes[i];
    }
  }

private:
  const size_t n_todos;
  const size_t n_original_vertices;
  const size_t* todo_edge_idxes;
  const size_t* result_vertex_idxes;
  size_t* edge_endpoints;
};

void UpdateEndpointsNewCPU(
  const size_t n_todos,
  const size_t n_original_vertices,
  const size_t* todo_edge_idxes,
  const size_t* result_vertex_idxes,
  size_t* edge_endpoints
) {
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, n_todos, TBB_GRAINSIZE_CPU_INTERSECT),
    UpdateEndpointsOp(n_todos, n_original_vertices, todo_edge_idxes, result_vertex_idxes, edge_endpoints)
  );

  // for (size_t i = 0; i < n_todos; i++) {
  //   if (result_vertex_idxes[i] >= n_original_vertices) {
  //     const size_t edge_idx = todo_edge_idxes[i];
  //     edge_endpoints[edge_idx*2 + 1] = result_vertex_idxes[i];
  //   }
  // }
}

class InterpolateVertexNewCPUOp {
public:
  InterpolateVertexNewCPUOp(
    const size_t n_todos,
    const size_t* todo_edge_idxes,
    size_t* result_vertex_idxes,
    float* vertices, const size_t n_dims,
    float* combinations, const size_t n_combination_dims,
    const size_t n_original_vertices,
    const size_t* edge_endpoints,
    const int* point_sign,
    const size_t plane_idx
  ) : n_todos(n_todos),
      todo_edge_idxes(todo_edge_idxes),
      result_vertex_idxes(result_vertex_idxes),
      vertices(vertices), n_dims(n_dims),
      combinations(combinations), n_combination_dims(n_combination_dims),
      n_original_vertices(n_original_vertices), edge_endpoints(edge_endpoints),
      point_sign(point_sign),
      plane_idx(plane_idx) {}

  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i < r.end(); i++) {
      (*this)(i);
    }
  }

  void operator()(const size_t todo_idx) const {
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
private:
  const size_t n_todos;
  const size_t* todo_edge_idxes;
  size_t* result_vertex_idxes;
  float* vertices;
  const size_t n_dims;
  float* combinations;
  const size_t n_combination_dims;
  const size_t n_original_vertices;
  const size_t* edge_endpoints;
  const int* point_sign;
  const size_t plane_idx;
};

void InterpolateVertexNewCPU(
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
  for (size_t todo_idx = 0; todo_idx < n_todos; todo_idx++) {
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

void PWLTransformer::IntersectNewCPU(
  RMMatrixXf& vertices,
  RMMatrixXf& combinations,
  const size_t n_edges,
  std::vector<size_t> & edge_endpoints,
  const std::vector<int> & point_sign,
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

  // std::cerr << n_edges << std::endl;
  // for (size_t i=0; i < n_edges; i++) {
  //   std::cerr << edge_endpoints[i];
  // }
  // std::cerr << std::endl;
  // assert (false);

  const size_t n_original_vertices = vertices.rows();
  const size_t n_dims = vertices.cols();

  auto& todo_edge_idxes = result_intersected_edge_idxes;
  auto& result_vertex_idxes = result_intersection_vertex_idxes;

  if (todo_edge_idxes.capacity() < n_edges) {
    todo_edge_idxes.resize(n_edges);
    std::cerr << "resize " << n_edges << " for todo_edge_idxes\n";
  }

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
    has_intersection(edge_endpoints.data(),
                     point_sign.data()));
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
    need_interpolation((edge_endpoints.data()),
                       (point_sign.data())));
  n_new_vertices = todo_edge_idxes_end - todo_edge_idxes.begin();

  // std::cerr << n_intersected_edges << "," << n_new_vertices << std::endl;

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time create todo   : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  if (result_vertex_idxes.capacity() < n_new_vertices) {
    result_vertex_idxes.resize(n_new_vertices);
    std::cerr << "resize " << n_new_vertices << " for result_vertex_idxes\n";
  }

  // std::cerr << "before resize\n" << vertices.block(0,0,3,3) << std::endl;
  // std::cerr << n_original_vertices << "," << n_new_vertices << "\n";
  const size_t n_preallocate_vertices = n_original_vertices + n_new_vertices;
  // vertices.resize_rows(n_preallocate_vertices);
  // combinations.resize_rows(n_preallocate_vertices);
  vertices.conservativeResize(n_preallocate_vertices, vertices.cols());
  combinations.conservativeResize(n_preallocate_vertices, combinations.cols());

  // std::cerr << "after resize\n" << vertices.block(0,0,3,3) << std::endl;

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time reserve space : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  // std::cerr << "start parallel interpolation\n";
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, n_new_vertices, TBB_GRAINSIZE_CPU_INTERSECT),
    InterpolateVertexNewCPUOp(
      n_new_vertices,
      (todo_edge_idxes.data()),
      (result_vertex_idxes.data()),
      vertices.data(), n_dims,
      combinations.data(), combinations.cols(),
      n_original_vertices,
      (edge_endpoints.data()),
      (point_sign.data()),
      plane_idx
    )
  );
  // std::cerr << "end parallel interpolation\n";
  // InterpolateVertexNewCPU(
  //   n_new_vertices,
  //   (todo_edge_idxes.data()),
  //   (result_vertex_idxes.data()),
  //   vertices.data(), n_dims,
  //   combinations.data(), combinations.cols(),
  //   n_original_vertices,
  //   (edge_endpoints.data()),
  //   (point_sign.data()),
  //   plane_idx
  // );

  // std::cerr << "after InterpolateVertexNewCPU\n" << vertices.block(0,0,3,3) << std::endl;

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time interpolation : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  // std::cerr << "start parallel UpdateEndpointsNewCPUOp\n";
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, n_new_vertices, TBB_GRAINSIZE_CPU_INTERSECT),
    UpdateEndpointsNewCPUOp(
      n_new_vertices,
      n_original_vertices,
      todo_edge_idxes.data(),
      result_vertex_idxes.data(),
      edge_endpoints.data()
    )
  );
  // std::cerr << "end parallel UpdateEndpointsNewCPUOp\n";

  // UpdateEndpointsNewCPU(
  //   n_new_vertices,
  //   n_original_vertices,
  //   todo_edge_idxes.data(),
  //   result_vertex_idxes.data(),
  //   edge_endpoints.data()
  // );

  // std::cerr << "after UpdateEndpointsNewCPU\n" << vertices.block(0,0,3,3) << std::endl;

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time upd endpoints : " << t.Ticks() << std::endl;
  t.Reset();
#endif

#ifdef TIME_INTERSECTION
  std::cerr << ">>> >>>  time copy results  : " << t.Ticks() << std::endl;
  t.Reset();
#endif

  thrust::copy(point_sign.begin(),
               point_sign.begin() + n_original_vertices,
               result_point_sign.begin());

  return;
}
