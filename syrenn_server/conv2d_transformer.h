#ifndef SYRENN_SYRENN_SERVER_CONV2D_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_CONV2D_TRANSFORMER_H_

#include <memory>
#include <string>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/segmented_line.h"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/affine_transformer.h"
#include "syrenn_server/strided_window_data.h"

// Transformer for 2D convolution layers.
class Conv2DTransformer : public AffineTransformer {
 public:
  // We expect filters.shape = (height*width, in_channels*out_channels)
  // We expect biases.shape = (out_channels,)
  Conv2DTransformer(const RMMatrixXf &filters,
                    const RMVectorXf &biases,
                    const StridedWindowData &window_data,
                    const int device = -1);
  ~Conv2DTransformer();
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer,
      const int device=-1);
  // input MUST be row-major
  // input.shape = (N, Hi*Wi*Ci)
  void Compute(RMMatrixXf *inout) const;
  void ComputeDevice(RMMatrixXfDevice *inout) const;
  unsigned int out_channels() const { return window_data_.out_channels(); }
  size_t out_size(size_t in_size) const override;
  std::string layer_type() const override { return "Conv2D"; }

 private:
  void FindOptimalConvolutionForwardAlgorithm() const;
  void SetDescriptors(size_t n_batches) const;
  void SetKernel(const RMMatrixXf &filters);

  const RMMatrixXf filters_;
  const RMMatrixXf biases_;
  const StridedWindowData window_data_;

  RMMatrixXfDevice filters_device_;
  RMMatrixXfDevice biases_device_;

  cudnnTensorDescriptor_t biases_on_device_descriptor;
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnConvolutionDescriptor_t conv2d_descriptor;

  // let all transformers share the same pre-allocated space on device.
  // TODO(zhetao): Maybe add lock to make `ComputeDevice()` thread-safe.
  static bool static_members_are_initialized_;
  static cudnnTensorDescriptor_t input_on_device_descriptor;
  static cudnnTensorDescriptor_t output_on_device_descriptor;
  static cudnnTensorDescriptor_t z_on_device_descriptor;
  static cudnnActivationDescriptor_t activation_id_descriptor;
  static cudnnConvolutionFwdAlgo_t conv2d_algorithm;
  static size_t conv2d_workspace_size_in_bytes;
  static float alpha1, alpha2;
  static int requestedAlgoCount;
  static std::vector<cudnnConvolutionFwdAlgoPerf_t> conv2d_algo_results;
};

#endif  // SYRENN_SYRENN_SERVER_CONV2D_TRANSFORMER_H_
