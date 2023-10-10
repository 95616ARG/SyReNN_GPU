#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/conv2d_transformer.h"
#include "syrenn_server/strided_window_data.h"
// #include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
// #include <NvInfer.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

bool Conv2DTransformer::static_members_are_initialized_ = false;
cudnnTensorDescriptor_t Conv2DTransformer::input_on_device_descriptor;
cudnnTensorDescriptor_t Conv2DTransformer::output_on_device_descriptor;
cudnnTensorDescriptor_t Conv2DTransformer::z_on_device_descriptor;
cudnnActivationDescriptor_t Conv2DTransformer::activation_id_descriptor;
cudnnConvolutionFwdAlgo_t Conv2DTransformer::conv2d_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
size_t Conv2DTransformer::conv2d_workspace_size_in_bytes = 0;
float Conv2DTransformer::alpha1 = 1.0f,
      Conv2DTransformer::alpha2 = 0.0f;

int Conv2DTransformer::requestedAlgoCount = 0;
std::vector<cudnnConvolutionFwdAlgoPerf_t> Conv2DTransformer::conv2d_algo_results(0);

Conv2DTransformer::Conv2DTransformer(const RMMatrixXf &filters,
                                     const RMVectorXf &biases,
                                     const StridedWindowData &window_data,
                                     const int device)
    : AffineTransformer(device),
      filters_(filters), biases_(biases),
      window_data_(window_data),
      filters_device_(), biases_device_() {

  if (device_ != -1) {
    CUDAShared::init();
    filters_device_ = RMMatrixXfDevice(1, window_data.out_channels()
                                          * window_data.in_channels()
                                          * window_data.window_height()
                                          * window_data.window_width());
    biases_device_ = RMMatrixXfDevice(biases);

    checkCUDNN( cudnnCreateTensorDescriptor(&biases_on_device_descriptor) );
    checkCUDNN( cudnnSetTensor4dDescriptor(biases_on_device_descriptor,
                                          CUDNN_TENSOR_NHWC,
                                          CUDNN_DATA_FLOAT,
                                          1, //  static_cast<int>(inout->rows()),
                                          static_cast<int>(window_data_.out_channels()),
                                          1, 1) );

    checkCUDNN( cudnnCreateConvolutionDescriptor(&conv2d_descriptor) );
    checkCUDNN( cudnnSetConvolution2dDescriptor(conv2d_descriptor,
                                              window_data_.pad_height(),
                                              window_data_.pad_width(),
                                              window_data_.stride_height(),
                                              window_data_.stride_width(),
                                              1, 1, // dilation
                                              CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_FLOAT));

    checkCUDNN( cudnnCreateFilterDescriptor(&kernel_descriptor) );
    checkCUDNN( cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NHWC,
                                          static_cast<int>(window_data_.out_channels()),
                                          static_cast<int>(window_data_.in_channels()),
                                          static_cast<int>(window_data_.window_height()),
                                          static_cast<int>(window_data_.window_width())) );

    if (filters.size() == filters_device_.size()) {
      SetKernel(filters);
    }

    if (!static_members_are_initialized_) {
      checkCUDNN( cudnnCnnInferVersionCheck() ); // preloads conv2d kernels
      checkCUDNN( cudnnCreateTensorDescriptor(&input_on_device_descriptor) );
      checkCUDNN( cudnnCreateTensorDescriptor(&output_on_device_descriptor) );
      checkCUDNN( cudnnCreateTensorDescriptor(&z_on_device_descriptor) );
      checkCUDNN( cudnnCreateActivationDescriptor(&activation_id_descriptor) );
      checkCUDNN( cudnnSetActivationDescriptor(activation_id_descriptor,
                                              CUDNN_ACTIVATION_IDENTITY,
                                              CUDNN_PROPAGATE_NAN,
                                              0.0f) );
      static_members_are_initialized_ = true;

      checkCUDNN( cudnnGetConvolutionForwardAlgorithmMaxCount(CUDAShared::cudnn_handle, &requestedAlgoCount) );
      conv2d_algo_results.resize(requestedAlgoCount);

      // dummy init to preload required kernels
      // SetDescriptors(100);
      // FindOptimalConvolutionForwardAlgorithm();
    }
  }
}

Conv2DTransformer::~Conv2DTransformer() {
  // std::cerr << "destroying Conv2DTransformer" << std::endl;
  // checkCUDNN( cudnnDestroy(CUDAShared::cudnn_handle) );
  // checkCUDNN( cudnnDestroyTensorDescriptor(input_on_device_descriptor) );
  // checkCUDNN( cudnnDestroyTensorDescriptor(output_on_device_descriptor) );
  // checkCUDNN( cudnnDestroyTensorDescriptor(biases_on_device_descriptor) );
  // checkCUDNN( cudnnDestroyTensorDescriptor(z_on_device_descriptor) );
  // checkCUDNN( cudnnDestroyFilterDescriptor(kernel_descriptor) );
  // checkCUDNN( cudnnDestroyConvolutionDescriptor(conv2d_descriptor) );
  // checkCUDNN( cudnnDestroyActivationDescriptor(activation_id_descriptor) );
  // checkCudaErrors( cudaFree(input_on_device) );
  // checkCudaErrors( cudaFree(output_on_device) );
  // checkCudaErrors( cudaFree(kernel_on_device) );
  // checkCudaErrors( cudaFree(biases_on_device) );
  // checkCudaErrors( cudaFree(z_on_device) );
  // checkCudaErrors( cudaFree(conv2d_workspace_on_device) );
}

void Conv2DTransformer::SetKernel(const RMMatrixXf &filters) {

  thrust::host_vector<float> kernel_on_host(filters_.size());

  const size_t K = window_data_.out_channels(),
               C = window_data_.in_channels(),
               H = window_data_.window_height(),
               W = window_data_.window_width();

  for (size_t k = 0; k < K; k++) {
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        for (size_t c = 0; c < C; c++) {
          kernel_on_host[k * H * W * C
                           + h * W * C
                               + w * C
                                   + c]
          = filters(h * W + w,
                    c * K + k);
        }
      }
    }
  }

  thrust::copy(
    kernel_on_host.begin(),
    kernel_on_host.end(),
    filters_device_.device().begin());

  return;
}

void Conv2DTransformer::SetDescriptors(size_t n_batches) const {
  checkCUDNN( cudnnSetTensor4dDescriptor(input_on_device_descriptor,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         static_cast<int>(n_batches),
                                         static_cast<int>(window_data_.in_channels()),
                                         static_cast<int>(window_data_.in_height()),
                                         static_cast<int>(window_data_.in_width())) );

  checkCUDNN( cudnnSetTensor4dDescriptor(output_on_device_descriptor,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         static_cast<int>(n_batches),
                                         static_cast<int>(window_data_.out_channels()),
                                         static_cast<int>(window_data_.out_height()),
                                         static_cast<int>(window_data_.out_width())) );
}

void Conv2DTransformer::FindOptimalConvolutionForwardAlgorithm() const {
#if CUDNN_MAJOR >= 8
  int returnedAlgoCount = 0;
  checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm_v7(CUDAShared::cudnn_handle,
                                             input_on_device_descriptor,
                                             kernel_descriptor,
                                             conv2d_descriptor,
                                             output_on_device_descriptor,
                                             requestedAlgoCount,
                                             &returnedAlgoCount,
                                             &conv2d_algo_results[0]) );
  size_t free_memory, total_memory;
  checkCudaErrors( cudaMemGetInfo(&free_memory, &total_memory) );

  bool found_conv_algorithm = false;
  for (int i = 0; i < returnedAlgoCount; i++)
  {
    if (conv2d_algo_results[i].status == CUDNN_STATUS_SUCCESS &&
        conv2d_algo_results[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED &&
        conv2d_algo_results[i].memory < free_memory)
    {
      found_conv_algorithm = true;
      conv2d_algorithm = conv2d_algo_results[i].algo;
      conv2d_workspace_size_in_bytes = conv2d_algo_results[i].memory;
      break;
    }
  }

  if (!found_conv_algorithm) {
    std::cerr << "cuDNN did not return a suitable algorithm for convolution."
              << std::endl;
  }
#else
  checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(CUDAShared::cudnn_handle,
                                        input_on_device_descriptor,
                                        kernel_descriptor,
                                        conv2d_descriptor,
                                        output_on_device_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        0, // memory limit
                                        &conv2d_algorithm) );

  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(CUDAShared::cudnn_handle,
                                                  input_on_device_descriptor,
                                                  kernel_descriptor,
                                                  conv2d_descriptor,
                                                  output_on_device_descriptor,
                                                  conv2d_algorithm,
                                                  &conv2d_workspace_size_in_bytes));
  std::cerr << "Workspace size: " << (conv2d_workspace_size_in_bytes / 1048576.0) << "MB"
            << std::endl;
#endif
}

void Conv2DTransformer::ComputeDevice(RMMatrixXfDevice *inout) const {

  SetDescriptors(inout->rows());

  CUDAShared::output.clear();
  CUDAShared::output.resize(inout->rows(),
                            window_data_.out_height()
                            * window_data_.out_width()
                            * window_data_.out_channels());

  FindOptimalConvolutionForwardAlgorithm();

  CUDAShared::workspace.resize(1, conv2d_workspace_size_in_bytes / sizeof(float));

  checkCUDNN( cudnnConvolutionBiasActivationForward(
                CUDAShared::cudnn_handle,
                &alpha1,
                input_on_device_descriptor,
                inout->data(),
                kernel_descriptor,
                filters_device_.data(),
                conv2d_descriptor,
                conv2d_algorithm,
                CUDAShared::workspace.data(),
                conv2d_workspace_size_in_bytes,
                &alpha2,
                output_on_device_descriptor, // z_on_device_descriptor,
                CUDAShared::output.data(),
                biases_on_device_descriptor,
                biases_device_.data(),
                activation_id_descriptor,
                output_on_device_descriptor,
                CUDAShared::output.data()) );

  inout->swap(CUDAShared::output);

  return;
}
