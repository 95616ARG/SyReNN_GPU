#include <iostream>
#include <memory>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/normalize_transformer.h"

NormalizeTransformer::NormalizeTransformer(const RMVectorXf &means,
                                   const RMVectorXf &standard_deviations,
                                   const int device)
    : AffineTransformer(device),
      means_(means), standard_deviations_(standard_deviations),
      means_device_(), vars_device_(),
      gamma(), beta() {

  if (device_ != -1) {
    CUDAShared::init();

    const size_t C = means.cols();

    RMMatrixXf vars = standard_deviations.array() * standard_deviations.array();
    means_device_ = RMMatrixXfDevice(means);
    vars_device_ = RMMatrixXfDevice(vars);
    gamma = RMMatrixXfDevice(1, C);
    beta = RMMatrixXfDevice(1, C);
    gamma.fill(1.);
    beta.fill(0.);

    checkCUDNN( cudnnCreateTensorDescriptor(&norm_descriptor) );
    checkCUDNN( cudnnCreateTensorDescriptor(&inout_descriptor) );
    checkCUDNN( cudnnSetTensor4dDescriptor(norm_descriptor,
                                          CUDNN_TENSOR_NHWC,
                                          CUDNN_DATA_FLOAT,
                                          1, C, 1, 1) );

  }
}
