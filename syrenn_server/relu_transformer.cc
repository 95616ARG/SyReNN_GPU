#include "syrenn_server/relu_transformer.h"
#include <memory>
#include <vector>
#include "eigen3/Eigen/Dense"
#include "mkldnn.hpp"

ReLUTransformer::ReLUTransformer(const int device): PWLTransformer(device) {
  checkCUDNN( cudnnCreateTensorDescriptor(&inout_on_device_desc) );
  checkCUDNN( cudnnCreateActivationDescriptor(&activation_descriptor) );
  checkCUDNN( cudnnSetActivationDescriptor(activation_descriptor,
                                           CUDNN_ACTIVATION_RELU,
                                           CUDNN_PROPAGATE_NAN,
                                           0.0f) );
}

ReLUTransformer::~ReLUTransformer() {
  checkCUDNN( cudnnDestroyTensorDescriptor(inout_on_device_desc) );
  checkCUDNN( cudnnDestroyActivationDescriptor(activation_descriptor) );
}

std::unique_ptr<LayerTransformer> ReLUTransformer::Deserialize(
    const syrenn_server::Layer &layer,
    const int device) {
  if (!layer.has_relu_data()) {
    return nullptr;
  }
  return std::unique_ptr<ReLUTransformer>(new ReLUTransformer(device));
}

size_t ReLUTransformer::n_piece_faces(size_t dims) const {
  return dims;
}

double ReLUTransformer::CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                                      Eigen::Ref<const RMVectorXf> to,
                                      const size_t face) const {
  return -from(face) / (to(face) - from(face));
}

int ReLUTransformer::PointSign(Eigen::Ref<const RMVectorXf> point,
                               const size_t face) const {
  if (point(face) == 0) {
    return 0;
  }
  return point(face) > 0 ? +1 : -1;
}

class PointSignOp {
public:
  PointSignOp(const RMMatrixXf &vertices,
              const size_t plane_idx,
              std::vector<int> & point_sign)
  : vertices(vertices), plane_idx(plane_idx), point_sign(point_sign)
  {}

  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i < r.end(); i++) {
      (*this)(i);
    }
  }

  void operator()(const size_t vertex_idx) const {
    if (vertices(vertex_idx, plane_idx) == 0) {
      point_sign[vertex_idx] = 0;
    } else {
      point_sign[vertex_idx] = vertices(vertex_idx, plane_idx) > 0 ? +1 : -1;
    }
  }
private:
  const RMMatrixXf &vertices;
  const size_t plane_idx;
  std::vector<int> & point_sign;
};

std::vector<int> & ReLUTransformer::PointSignBatch(const RMMatrixXf &vertices,
                                              const size_t plane_idx) const {


  // std::cerr << "before PointSignBatch\n" << vertices.block(0,0,3,3) << std::endl;

  const int n_vertices = vertices.rows();
  const int n_dims = vertices.cols();

  if (PWLTransformer::point_sign.capacity() < vertices.rows()) {
    PWLTransformer::point_sign.resize(vertices.rows());
    std::cerr << "resize " << n_vertices << " for point_sign\n";
  }

  // std::cerr << "after resize\n" << vertices.block(0,0,3,3) << std::endl;

  // std::cerr << "start parallel PointSignBatch\n";
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, n_vertices, TBB_GRAINSIZE_CPU_INTERSECT),
    PointSignOp(vertices, plane_idx, PWLTransformer::point_sign)
  );
  // std::cerr << "end parallel PointSignBatch\n";
  // for (size_t vertex_idx = 0; vertex_idx < n_vertices; vertex_idx++) {
  //   if (vertices(vertex_idx, plane_idx) == 0) {
  //     PWLTransformer::point_sign[vertex_idx] = 0;
  //   } else {
  //     PWLTransformer::point_sign[vertex_idx] = vertices(vertex_idx, plane_idx) > 0 ? +1 : -1;
  //   }
  //   // std::cerr << "after vertex " << vertex_idx << "\n" << vertices.block(0,0,3,3) << std::endl;
  // }

  return PWLTransformer::point_sign;
}

void ReLUTransformer::Compute(RMMatrixXf *inout) const {
  // Non-MKLDNN version.
  // inout->noalias() = inout->cwiseMax(0.0);

  // Modified from
  // https://github.com/intel/mkl-dnn/blob/mnt-v0/examples/simple_net.cpp
  // See conv2d_transformer.cc for more.

  mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
  mkldnn::stream cpu_stream(cpu_engine);

  // NOTE: MKL reads the dimension sizes in NCHW even though the layout we
  // store it in is NHWC
  mkldnn::memory::dims input_dimensions{static_cast<int>(inout->size())};

  // MKL memory references to the above buffers
  auto inout_memory =
      mkldnn::memory(
              {
                  { input_dimensions },
                  mkldnn::memory::data_type::f32,
                  mkldnn::memory::format_tag::x
              },
              cpu_engine, inout->data());

  auto relu_descriptor = mkldnn::eltwise_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::eltwise_relu,
          inout_memory.get_desc(), 0.0f, 0.0f);
  auto relu_primitive = mkldnn::eltwise_forward::primitive_desc(
                  relu_descriptor, cpu_engine);

  auto relu = mkldnn::eltwise_forward(relu_primitive);
  relu.execute(cpu_stream, {
    {MKLDNN_ARG_SRC, inout_memory},
    {MKLDNN_ARG_DST, inout_memory},
  });
}
