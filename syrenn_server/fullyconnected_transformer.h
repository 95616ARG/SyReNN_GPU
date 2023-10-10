#ifndef SYRENN_SYRENN_SERVER_FULLYCONNECTED_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_FULLYCONNECTED_TRANSFORMER_H_

#include <memory>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/affine_transformer.h"
#include "cublas_v2.h"

// Transformer for Fully-Connected layers.
class FullyConnectedTransformer : public AffineTransformer {
 public:
  FullyConnectedTransformer(const RMMatrixXf &weights,
                            const RMVectorXf &biases,
                            const int device = -1);
  // ~FullyConnectedTransformer();
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer,
      const int device = -1);
  void Compute(RMMatrixXf *inout) const;
  void ComputeDevice(RMMatrixXfDevice *inout) const;
  std::string layer_type() const override { return "Fully-Connected"; };
  size_t out_size(size_t in_size) const { return biases_.size(); }

  const RMMatrixXf weights_;
  const RMVectorXf biases_;
  RMMatrixXfDevice weights_device_;
  RMMatrixXfDevice biases_device_;
 private:
};

#endif  // SYRENN_SYRENN_SERVER_FULLYCONNECTED_TRANSFORMER_H_
