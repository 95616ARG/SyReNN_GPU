#include "syrenn_server/maxpool_transformer.h"
#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <unordered_set>
#include <utility>
#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1
#include "tbb/concurrent_set.h"
#include "eigen3/Eigen/Dense"
// #include "mkldnn.hpp"
#include "openblas/cblas.h"
#include "syrenn_server/strided_window_data.h"
#include <cudnn.h>

MaxPoolTransformer::MaxPoolTransformer(const StridedWindowData &window_data)
    : window_data_(window_data) {}

std::unique_ptr<LayerTransformer> MaxPoolTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_maxpool_data()) {
    return nullptr;
  }

  const auto &maxpool_data = layer.maxpool_data();
  const auto window_data = StridedWindowData::Deserialize(
      maxpool_data.window_data());

  return std::unique_ptr<MaxPoolTransformer>(
          new MaxPoolTransformer(window_data));
}

size_t MaxPoolTransformer::out_size(size_t in_size) const {
  return window_data_.out_size();
}

void MaxPoolTransformer::process_window(
        const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &from_window,
        const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &to_window,
        double from_ratio, double to_ratio,
        tbb::concurrent_set<double> *endpoints) const {
  // Find the existing maxes of the windows.
  WindowPixel from_argmax, to_argmax;
  from_window.maxCoeff(&from_argmax.h, &from_argmax.w);
  to_window.maxCoeff(&to_argmax.h, &to_argmax.w);
  if (from_argmax == to_argmax) {
    // They're the same, so the MaxPool is linear on this window/segment.
    return;
  }

  // These are linear regions we've already hit. Here to ensure floating point
  // inaccuracies don't get us caught in an infinite loop.
  std::unordered_set<WindowPixel> old_argmaxes{from_argmax};

  // TODO(masotoud): Allocate space for these in the caller and just pass a
  // reference or map instead of re-allocating them every time here.
  RMMatrixXf delta = to_window - from_window;
  double last_ratio = 0.0;

  // Loop over linear regions.
  while (old_argmaxes.size() < static_cast<size_t>(from_window.size())) {
    // We look for the closest coefficient to "overtake" from_argmax (that
    // hasn't already been an argmax and that's before to_window).
    WindowPixel best_arg = to_argmax;
    double best_ratio = 1.0;
    for (Eigen::Index row = 0; row < from_window.rows(); row++) {
    for (Eigen::Index col = 0; col < from_window.cols(); col++) {
      WindowPixel pixel(row, col);
      if (old_argmaxes.count(pixel) == 1) {
        continue;
      }
      // "Local" ratio between from_window and to_window.
      // (S + tD)[n] = (S + tD)[i]
      // S[n] - S[i] = t(D[i] - D[n])
      // t = (S[n] - S[i]) / (D[i] - D[n])
      double crossing_ratio =
        (from_window(row, col) - from_window(from_argmax.h, from_argmax.w)) /
        (delta(from_argmax.h, from_argmax.w) - delta(row, col));
      if (crossing_ratio > last_ratio && crossing_ratio < best_ratio) {
        best_arg = pixel;
        best_ratio = crossing_ratio;
      }
    }
    }

    if (best_ratio == 1.0) {
      // Done!
      break;
    }

    from_argmax = best_arg;
    double global_ratio = from_ratio + (best_ratio * (to_ratio - from_ratio));
    // Finally, we insert the crossing distance.
    endpoints->insert(global_ratio);
    last_ratio = best_ratio;
    if (best_arg == to_argmax) {
      // We just added the endpoint for the last linear region, so we're done!
      break;
    }
  }  // Loop over linear regions.
}

void MaxPoolTransformer::process_images(
        Eigen::Ref<const RMMatrixXf> from_image,
        Eigen::Ref<const RMMatrixXf> to_image,
        double from_ratio, double to_ratio,
        tbb::concurrent_set<double> *endpoints) const {
  // Both we assume are NHWC
  auto from_image_shaped = Eigen::Map<const RMMatrixXf>(
          from_image.data(),
          window_data_.in_height() * window_data_.in_width(),
          window_data_.in_channels());
  auto to_image_shaped = Eigen::Map<const RMMatrixXf>(
          to_image.data(),
          window_data_.in_height() * window_data_.in_width(),
          window_data_.in_channels());
  // We transpose both to be (CH, W) so we can block out the windows
  RMMatrixXf from_image_transposed = from_image_shaped.transpose();
  RMMatrixXf from_image_transformed = Eigen::Map<const RMMatrixXf>(
          from_image_transposed.data(),
          window_data_.in_channels() * window_data_.in_height(),
          window_data_.in_width());
  RMMatrixXf to_image_transposed = to_image_shaped.transpose();
  RMMatrixXf to_image_transformed = Eigen::Map<const RMMatrixXf>(
          to_image_transposed.data(),
          window_data_.in_channels() * window_data_.in_height(),
          window_data_.in_width());

  int pad_height = window_data_.pad_height(),
      pad_width = window_data_.pad_width(),
      window_height = window_data_.window_height(),
      window_width = window_data_.window_width(),
      padded_height = window_data_.padded_in_height(),
      padded_width = window_data_.padded_in_width(),
      stride_height = window_data_.stride_height(),
      stride_width = window_data_.stride_width();

  tbb::parallel_for(size_t(0), window_data_.in_channels(), [&](size_t c) {
  tbb::parallel_for(
      // Padding is considered "negative indices."
      -pad_height,
      // We finish once the end of the window extends beyond the padded height,
      // i.e. continue as long as (i + window_height) <= padded_height.
      (padded_height - window_height) + 1,
      // Strides determine the step size.
      stride_height, [&](int i) {
  tbb::parallel_for(
      -pad_width, (padded_width - window_width) + 1, stride_width, [&](int j) {
    // Sizes of the window not including the padding.
    size_t real_i = static_cast<size_t>(std::max(i, 0)),
           real_j = static_cast<size_t>(std::max(j, 0)),
           // Don't be fooled by the addition signs --- this amounts to
           // subtracting off the padding because min(i, 0) <= 0.
           real_height = window_data_.window_height() + std::min(i, 0),
           real_width = window_data_.window_width() + std::min(j, 0);
    if ((i + real_height) >= window_data_.in_height()) {
      real_height = (window_data_.in_height() - i);
    }
    if ((j + real_width) >= window_data_.in_width()) {
      real_width = (window_data_.in_width() - j);
    }
    Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> from_window(
            (from_image_transformed.data() +
             (c * window_data_.in_height() * window_data_.in_width()) +
             (real_i * window_data_.in_width()) + real_j),
            real_height, real_width,
            Eigen::OuterStride<>(window_data_.in_width()));
    Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> to_window(
            (to_image_transformed.data() +
             (c * window_data_.in_height() * window_data_.in_width()) +
             (real_i * window_data_.in_width()) + real_j),
            real_height, real_width,
            Eigen::OuterStride<>(window_data_.in_width()));
    process_window(from_window, to_window, from_ratio, to_ratio, endpoints);
  });  // Width-windows.
  });  // Height-windows
  });  // Channels.
}

std::vector<double> MaxPoolTransformer::ProposeLineEndpoints(
    const SegmentedLine &line) const {
  tbb::concurrent_set<double> endpoints;

  const RMMatrixXf &points = line.points();

  for (Eigen::Index i = 0; i < points.rows() - 1; i++) {
    process_images(
        points.block(i, 0, 1, points.cols()),
        points.block(i + 1, 0, 1, points.cols()),
        line.endpoint_ratio(i), line.endpoint_ratio(i + 1), &endpoints);
  }

  std::vector<double> vector_endpoints(endpoints.begin(), endpoints.end());
  return vector_endpoints;
}

void MaxPoolTransformer::Compute(RMMatrixXf *inout) const {

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
                              CUDNN_POOLING_MAX,
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
void MaxPoolTransformer::Compute(RMMatrixXf *inout) const {
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
          mkldnn::algorithm::pooling_max,
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
