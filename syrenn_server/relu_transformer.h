#ifndef SYRENN_SYRENN_SERVER_RELU_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_RELU_TRANSFORMER_H_

#include <memory>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/pwl_transformer.h"
#include "syrenn_server/transformer.h"

// Transformer for a ReLU layer, relu(x) = max(x, 0.0).
class ReLUTransformer : public PWLTransformer {
 public:
  ReLUTransformer(const int device = -1);
  ~ReLUTransformer();
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer,
      const int device = -1);
  void Compute(RMMatrixXf *inout) const override;
  void ComputeDevice(RMMatrixXfDevice *inout) const override;
  std::string layer_type() const override { return "ReLU"; }
  size_t out_size(size_t in_size) const override { return in_size; }

  // TODO(zhetao): move back under protected after debug
//  protected:
  size_t n_piece_faces(size_t dims) const override;
  double CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                       Eigen::Ref<const RMVectorXf> to,
                       const size_t face) const override;
  double CrossingRatio(const RMMatrixXfDevice &vertices,
                       const size_t from_idx,
                       const size_t to_idx,
                       const size_t face) const override;
  std::vector<float> CrossingRatio(const RMMatrixXfDevice &vertices,
                               const std::vector<size_t> &from_idxes,
                               const std::vector<size_t> &to_idxes,
                               const size_t face) const override;
  // __device__ float CrossingRatio(const size_t n_dims,
  //                               const float* vertices,
  //                               const size_t from_idx,
  //                               const size_t to_idx,
  //                               const size_t plane_idx) const override;
  int PointSign(Eigen::Ref<const RMVectorXf> point,
                const size_t face) const override;
  int PointSign(const RMMatrixXfDevice &vertices,
                size_t vertex_idx,
                const size_t face) const override;
  std::vector<int> & PointSignBatch(const RMMatrixXf &vertices,
                               const size_t face) const override;
  thrust::device_vector<int> & PointSignDevice(const RMMatrixXfDevice &vertices,
                                               const size_t plane_idx) const override;
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
  cudnnTensorDescriptor_t inout_on_device_desc;
  cudnnActivationDescriptor_t  activation_descriptor;
};

#endif  // SYRENN_SYRENN_SERVER_RELU_TRANSFORMER_H_
