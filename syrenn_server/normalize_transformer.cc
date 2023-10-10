#include <iostream>
#include <memory>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/normalize_transformer.h"

std::unique_ptr<LayerTransformer> NormalizeTransformer::Deserialize(
    const syrenn_server::Layer &layer,
    const int device) {
  if (!layer.has_normalize_data()) {
    return nullptr;
  }
  Eigen::Map<const RMVectorXf> means(
                  layer.normalize_data().means().data(),
                  layer.normalize_data().means_size());
  Eigen::Map<const RMVectorXf> standard_deviations(
                  layer.normalize_data().standard_deviations().data(),
                  layer.normalize_data().standard_deviations_size());

  return std::unique_ptr<LayerTransformer>(
            new NormalizeTransformer(means, standard_deviations, device));
}

void NormalizeTransformer::Compute(RMMatrixXf *inout) const {
  // input will be (N, H*W*C)
  // We reshape to (N*H*W, C)
  unsigned int hw = inout->cols() / means_.size();
  size_t batch = inout->rows();
  inout->resize(batch * hw, means_.size());

  inout->array().rowwise() -= means_.array();
  inout->array().rowwise() /= standard_deviations_.array();

  inout->resize(batch, hw * means_.size());
}

// #define TIME_NORM 1
// #define DEBUG_NORM 1

void NormalizeTransformer::ComputeDevice(RMMatrixXfDevice *inout) const {

#ifdef TIME_NORM
  Timer t; t.Reset();
#endif

  static const float one = 1.f;
  static const float zero  = 0.f;

  const size_t N = inout->rows();
  const size_t C = means_device_.cols();
  const size_t HW = inout->cols() / C;

  checkCUDNN( cudnnSetTensor4dDescriptor(inout_descriptor,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         N, C, HW, 1) );

#ifdef TIME_NORM
  std::cerr << t.Ticks() << " ms -- norm prepare\n"; t.Reset();
#endif

  CUDAShared::output.resize(N, inout->cols());

#ifdef TIME_NORM
  std::cerr << t.Ticks() << " ms -- norm resize\n"; t.Reset();
#endif

  checkCUDNN( cudnnNormalizationForwardInference(
                      CUDAShared::cudnn_handle,
                      CUDNN_NORM_PER_CHANNEL,
                      CUDNN_NORM_OPS_NORM,
                    //   CUDNN_NORM_ALGO_STANDARD,
                      CUDNN_NORM_ALGO_PERSIST,
                      &one, &zero,
                      inout_descriptor, inout->data(),
                      norm_descriptor, gamma.data(), beta.data(),
                      norm_descriptor, means_device_.data(), vars_device_.data(),
                      NULL, NULL,
                      NULL,
                      inout_descriptor, CUDAShared::output.data(),
                      0., 1) );

#ifdef TIME_NORM
  std::cerr << t.Ticks() << " ms -- norm forward\n"; t.Reset();
#endif

  inout->swap(CUDAShared::output);

#ifdef TIME_NORM
  std::cerr << t.Ticks() << " ms -- norm swap\n"; t.Reset();
#endif

  return;

}
