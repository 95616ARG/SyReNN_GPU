#ifndef SYRENN_SYRENN_SERVER_ARGMAX_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_ARGMAX_TRANSFORMER_H_

#include <memory>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/pwl_transformer.h"
#include "syrenn_server/transformer.h"

// Transformer for an ArgMax layer. This is mostly used as a helper method in
// the front-end to determine classification of lines/planes (see
// helpers/classify_{lines, planes}.py).
//
// *NOTE:* using this in a call to the line/plane transformer with
// include_post=True may not do what you expect, because the argmax at each of
// the endpoints will be ill-defined (as the endpoints are where the argmax
// change). See the Helpers for examples of how to use this correctly ---
// namely, with include_post=False. This is the only non-continuous function
// that has a transformer for it in this repository.
class ArgMaxTransformer : public PWLTransformer {
 public:
  ArgMaxTransformer(): PWLTransformer() {};
  ArgMaxTransformer(const int device): PWLTransformer(device) {};
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer,
      const int device);
  void Compute(RMMatrixXf *inout) const override;
  void ComputeDevice(RMMatrixXfDevice *inout) const override;
  std::string layer_type() const override { return "ArgMax"; };
  size_t out_size(size_t in_size) const { return 1; }
 protected:
  size_t n_piece_faces(size_t dims) const override;
  double CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                       Eigen::Ref<const RMVectorXf> to,
                       const size_t face) const override;
  bool IsFaceActive(Eigen::Ref<const RMVectorXf> from,
                    Eigen::Ref<const RMVectorXf> to,
                    const size_t face) const override;
  int PointSign(Eigen::Ref<const RMVectorXf> point,
                const size_t face) const override;

  // GPU versions
  double CrossingRatio(const RMMatrixXfDevice &vertices,
                       size_t from_idx,
                       size_t to_idx,
                       const size_t face) const override;
  std::vector<float> CrossingRatio(const RMMatrixXfDevice &vertices,
                               const std::vector<size_t> &from_idxes,
                               const std::vector<size_t> &to_idxes,
                               const size_t face) const override;
//   __device__ float CrossingRatio(const size_t n_dims,
//                                 const float* vertices,
//                                 const size_t from_idx,
//                                 const size_t to_idx,
//                                 const size_t plane_idx) const override;
  int PointSign(const RMMatrixXfDevice &vertices,
                size_t vertex_idx,
                const size_t face) const override;
  std::vector<int> & PointSignBatch(const RMMatrixXf &vertices,
                               const size_t face) const override;
  thrust::device_vector<int> & PointSignDevice(
                                const RMMatrixXfDevice &vertices,
                                const size_t face) const override;
  void Intersect(UPolytope* inout,
                const size_t plane_idx,
                std::vector<int> & result_point_sign,
                std::vector<size_t> & result_intersected_edge_idxes,
                std::vector<size_t> & result_intersection_vertex_idxes,
                size_t & n_intersected_edges,
                size_t & n_new_vertices) const override;

  void Intersect(
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
  ) const override;

 private:
  void ComputeSquareIndexFromTriangular(const size_t n_dims) const;
  static RMMatrixXiDevice square_index;
  static int computed_square_index_for;
};

#endif  // SYRENN_SYRENN_SERVER_ARGMAX_TRANSFORMER_H_
