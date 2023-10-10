#include <memory>
#include <vector>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/averagepool_transformer.h"
#include "syrenn_server/strided_window_data.h"
// #include "mkldnn.hpp"
#include <cudnn.h>

AveragePoolTransformer::AveragePoolTransformer(
    const StridedWindowData &window_data)
    : window_data_(window_data) {}

size_t AveragePoolTransformer::out_size(size_t in_size) const {
  return window_data_.out_size();
}

std::unique_ptr<LayerTransformer> AveragePoolTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_averagepool_data()) {
    return nullptr;
  }
  const auto &averagepool_data = layer.averagepool_data();
  const auto window_data = StridedWindowData::Deserialize(
      averagepool_data.window_data());

  return std::unique_ptr<LayerTransformer>(
        new AveragePoolTransformer(window_data));
}

void AveragePoolTransformer::Compute(RMMatrixXf *inout) const {

  float* input_on_device;
  checkCudaErrors( cudaMalloc((void **)&input_on_device,
                              sizeof(float) * static_cast<size_t>(inout->size())) );
  checkCudaErrors( cudaMemcpy(input_on_device, inout->data(),
                              sizeof(float) * static_cast<size_t>(inout->size()),
                              cudaMemcpyKind::cudaMemcpyHostToDevice) );

  cudnnTensorDescriptor_t input_on_device_descriptor;
  checkCUDNN( cudnnCreateTensorDescriptor(&input_on_device_descriptor) );
  checkCUDNN( cudnnSetTensor4dDescriptor(input_on_device_descriptor,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         static_cast<int>(inout->rows()),
                                         static_cast<int>(window_data_.in_channels()),
                                         static_cast<int>(window_data_.in_height()),
                                         static_cast<int>(window_data_.in_width())) );

  cudnnPoolingDescriptor_t pooling_descriptor;
  checkCUDNN( cudnnCreatePoolingDescriptor(&pooling_descriptor) );

  int nbDims = 2;
  int windowDimA[] = {(int)window_data_.window_height(), (int)window_data_.window_width()};
  int padA[] = {(int)window_data_.pad_height(), (int)window_data_.pad_width()};
  int strideA[] = {(int)window_data_.stride_height(), (int)window_data_.stride_width()};

  // int windowDimA[] = {(int)window_data_.window_width(), (int)window_data_.window_height()};
  // int padA[] = {(int)window_data_.pad_width(), (int)window_data_.pad_height()};
  // int strideA[] = {(int)window_data_.stride_width(),(int)window_data_.stride_height()};

  checkCUDNN( cudnnSetPoolingNdDescriptor(pooling_descriptor,
                              CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                              CUDNN_PROPAGATE_NAN,
                              nbDims, windowDimA, padA, strideA) );

  int outputDimA[4];
  checkCUDNN( cudnnGetPoolingNdForwardOutputDim(
                pooling_descriptor,
                input_on_device_descriptor,
                4,
                outputDimA) );

  // Allocate and copy the output to device.
  size_t output_on_device_size = outputDimA[0]
                                 * outputDimA[1]
                                 * outputDimA[2]
                                 * outputDimA[3];
  float* output_on_device;
  checkCudaErrors( cudaMalloc((void **)&output_on_device,
                              sizeof(float) * output_on_device_size) );

  cudnnTensorDescriptor_t output_on_device_descriptor;
  checkCUDNN( cudnnCreateTensorDescriptor(&output_on_device_descriptor) );
  checkCUDNN( cudnnSetTensor4dDescriptor(output_on_device_descriptor,
                                         CUDNN_TENSOR_NHWC,
                                         CUDNN_DATA_FLOAT,
                                         outputDimA[0],
                                         outputDimA[1],
                                         outputDimA[2],
                                         outputDimA[3]) );

  cudnnHandle_t cudnnHandle;
  checkCUDNN( cudnnCreate(&cudnnHandle) );

  std::cerr << "before forward" << std::endl;
  const float alpha = 1.0f,
              beta  = 0.0f;
  checkCUDNN( cudnnPoolingForward(
                cudnnHandle,
                pooling_descriptor,
                &alpha,
                input_on_device_descriptor,
                input_on_device,
                &beta,
                output_on_device_descriptor,
                output_on_device) );

  inout->resize(inout->rows(), outputDimA[1]
                               * outputDimA[2]
                               * outputDimA[3]);

  checkCudaErrors( cudaMemcpy(inout->data(), output_on_device,
                              sizeof(float) * output_on_device_size,
                              cudaMemcpyKind::cudaMemcpyDeviceToHost) );

  checkCUDNN( cudnnDestroy(cudnnHandle) );
  checkCudaErrors( cudaFree(input_on_device) );
  checkCudaErrors( cudaFree(output_on_device) );

  // nvinfer1::IPoolingLayer* pooling = network->addPooling(
  //   *inputTensor,
  //   nvinfer1::PoolingType::AVERAGE,
  //   nvinfer1::DimsHW{kernelShape[0], kernelShape[1]}
  // );
}

/*
void AveragePoolTransformer::Compute(RMMatrixXf *inout) const {
  // Modified from
  // https://github.com/intel/mkl-dnn/blob/mnt-v0/examples/simple_net.cpp
  // See conv2d_transformer.cc for more.

  mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
  mkldnn::stream cpu_stream(cpu_engine);

  int batch = inout->rows();

  mkldnn::memory::dims input_dims = window_data_.mkl_input_dims(batch);
  mkldnn::memory::dims window_dims = window_data_.mkl_window_dims();
  mkldnn::memory::dims strides = window_data_.mkl_stride_dims();
  mkldnn::memory::dims output_dims = window_data_.mkl_output_dims(batch);
  mkldnn::memory::dims padding = window_data_.mkl_pad_dims();

  // Initialize an output buffer for MKL to use
  RMMatrixXf output_data(batch, window_data_.out_size());
  output_data.setZero();

  // MKL memory references to the above buffers
  auto input_memory =
      mkldnn::memory(
          {
              { input_dims },
              mkldnn::memory::data_type::f32,
              mkldnn::memory::format_tag::nhwc
          }, cpu_engine, inout->data());
  auto output_memory =
      mkldnn::memory(
          {
              { output_dims },
              mkldnn::memory::data_type::f32,
              mkldnn::memory::format_tag::nhwc
          }, cpu_engine, output_data.data());

  // create an averagepool:
  // https://intel.github.io/mkl-dnn/cpu_cnn_inference_f32_8c-example.html#a41
  auto pool_descriptor = mkldnn::pooling_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::pooling_avg_include_padding,
          input_memory.get_desc(), output_memory.get_desc(),
          strides, window_dims, padding, padding);
  auto pool_primitive = mkldnn::pooling_forward::primitive_desc(
          pool_descriptor, cpu_engine);

  auto pool = mkldnn::pooling_forward(pool_primitive);
  pool.execute(cpu_stream, {
    {MKLDNN_ARG_SRC, input_memory},
    {MKLDNN_ARG_DST, output_memory},
  });

  inout->swap(output_data);
}
*/
