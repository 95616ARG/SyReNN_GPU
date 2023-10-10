#include <assert.h>

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <stack>
#include <queue>
#include <tuple>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "eigen3/Eigen/Dense"

#include "syrenn_proto/syrenn.grpc.pb.h"

#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/fullyconnected_transformer.h"
#include "syrenn_server/relu_transformer.h"
#include "syrenn_server/conv2d_transformer.h"
// #include "syrenn_server/maxpool_transformer.h"
// #include "syrenn_server/relu_maxpool_transformer.h"
// #include "syrenn_server/averagepool_transformer.h"
#include "syrenn_server/normalize_transformer.h"
// #include "syrenn_server/concat_transformer.h"
// #include "syrenn_server/hard_tanh_transformer.h"
#include "syrenn_server/argmax_transformer.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::ServerReaderWriter;
using grpc::Status;
using syrenn_server::TransformRequest;
using syrenn_server::TransformResponse;

struct StubDescriptor {
  StubDescriptor(std::unique_ptr<SegmentedLineStub> *stub,
                 double start_ratio, double end_ratio)
    : stub(std::move(*stub)), start_ratio(start_ratio), end_ratio(end_ratio) {}

  std::unique_ptr<SegmentedLineStub> stub;
  double start_ratio;
  double end_ratio;
};

void split_line(SegmentedLine *line, std::stack<StubDescriptor> *stub_stack,
                double *global_start_ratio, double *global_end_ratio) {
  // This is the target memory usage (in bytes) to stay under.
  constexpr size_t memory_usage = (8ull * 1024ull * 1024ull * 1024ull);
  // A few operations temporarily double the amount of memory used, so the true
  // threshold should be half of the desired memory usage.
  constexpr size_t stubify_threshold = memory_usage / 2ull;
  double global_length = (*global_end_ratio) - (*global_start_ratio);

  size_t points_per_subline = stubify_threshold / (4 * line->point_dims());

  std::stack<StubDescriptor> to_stack;
  // First, we add all of the stubs.
  for (size_t subline_start = points_per_subline;
       subline_start < line->Size();
       subline_start += points_per_subline) {
    // We take one more to make sure everything "links up."
    size_t n_endpoints = points_per_subline + 1;
    if ((subline_start + n_endpoints) > line->Size()) {
      n_endpoints = line->Size() - subline_start;
    }
    if (n_endpoints == 1) {
      break;
    }
    size_t subline_end = subline_start + n_endpoints;

    double local_start_ratio = line->endpoint_ratio(subline_start);
    double local_end_ratio = line->endpoint_ratio(subline_end - 1);
    double stub_start_ratio =
        (*global_start_ratio) + (local_start_ratio * global_length);
    double stub_end_ratio =
        (*global_start_ratio) + (local_end_ratio * global_length);

    auto stub = line->ExtractStub(subline_start, subline_end);
    to_stack.emplace(&stub, stub_start_ratio, stub_end_ratio);
  }

  // Then we add them to the stack
  while (!to_stack.empty()) {
    stub_stack->push(std::move(to_stack.top()));
    to_stack.pop();
  }

  line->RemoveAfter(points_per_subline - 1);
}

std::pair<std::vector<double>, RMMatrixXf> transform_line(
    const SegmentedLine &global_line,
    const std::vector<std::unique_ptr<LayerTransformer>> &layers,
    const bool include_post) {
  assert(global_line.Size() == 2);
  assert(global_line.endpoint_ratio(0) == 0.0);
  assert(global_line.endpoint_ratio(1) == 1.0);

  std::unique_ptr<SegmentedLine> line(
      new SegmentedLine(global_line.InterpolatePreimage(0.0),
                        global_line.InterpolatePreimage(1.0)));
  double global_start_ratio = 0.0, global_end_ratio = 1.0, global_length = 1.0;
  std::stack<StubDescriptor> remaining_stubs;

  std::vector<double> final_endpoints;
  RMMatrixXf post_vertices(0, 0);

  while (line) {
    size_t start_layer = line->n_applied_layers();
    for (size_t layer = start_layer; layer < layers.size(); layer++) {
      layers[layer]->TransformLine(line.get());
      split_line(line.get(), &remaining_stubs,
                 &global_start_ratio, &global_end_ratio);
    }
    for (size_t i = 0; i < line->Size(); i++) {
      double local_ratio = line->endpoint_ratio(i);
      double global_ratio = global_start_ratio + (local_ratio * global_length);
      final_endpoints.push_back(global_ratio);
    }
    if (include_post) {
      line->PrecomputePoints();
      int n_old = post_vertices.rows();
      int n_new = line->Size();
      int dims = line->point_dims();
      post_vertices.conservativeResize(n_old + n_new, dims);
      post_vertices.block(n_old, 0, n_new, dims) = line->points();
    }
    line.reset();

    if (!remaining_stubs.empty()) {
      StubDescriptor &descriptor = remaining_stubs.top();
      RMVectorXf start =
          global_line.InterpolatePreimage(descriptor.start_ratio);
      RMVectorXf end =
          global_line.InterpolatePreimage(descriptor.end_ratio);
      line = std::unique_ptr<SegmentedLine>(
          new SegmentedLine(descriptor.stub.get(), start, end));
      remaining_stubs.pop();
    }
  }
  // NOTE: This assumes the final_endpoints are sorted, which they *should* be
  // as long as split_line does its job correctly...
  size_t i = 0, j = 0;
  for (i = 0, j = 0; i < final_endpoints.size(); i++) {
    if (i == 0 || final_endpoints[i] != final_endpoints[i - 1]) {
      final_endpoints[j] = final_endpoints[i];
      if (include_post) {
        post_vertices.row(j) = post_vertices.row(i);
      }
      j++;
    }
  }
  final_endpoints.resize(j);
  post_vertices.conservativeResize(j, post_vertices.cols());

  std::pair<std::vector<double>, RMMatrixXf> result;
  result.first = std::move(final_endpoints);
  result.second.swap(post_vertices);
  return result;
}

#define TIME_TRANSFORM_TOP 1

void transform_polytope(
    UPolytope *upolytope,
    const std::vector<std::unique_ptr<LayerTransformer>> &layers,
    syrenn_server::Timing& timing,
    uint64_t & split_scale) {

  double t_layer = 0.;
  double t_fc = 0.;
  double t_conv2d = 0.;
  double t_norm = 0.;
  double t_relu = 0.;
  double t_argmax = 0.;
  double t_affine = 0.;
  double t_pwl = 0.;
  double t_total = 0.;

#ifdef TIME_TRANSFORM_TOP
  std::cerr << "------ start transform -------\n";
  std::cerr << "n vertices: " << upolytope->n_vertices() << std::endl;
#endif

  Timer t;
  Timer t0; t0.Reset();
  for (auto &layer : layers) {

#ifdef TIME_TRANSFORM_TOP
    std::cerr << "------ " << layer->layer_type() << " -------\n";
#endif

    t.Reset();
    layer->TransformUPolytope(upolytope);
    t_layer = t.Ticks();

    if (layer->layer_type() == "Fully-Connected") {
      t_fc += t_layer;
      t_affine += t_layer;
    }
    if (layer->layer_type() == "Conv2D") {
      t_conv2d += t_layer;
      t_affine += t_layer;
      }
    if (layer->layer_type() == "Normalize") {
      t_norm += t_layer;
      t_affine += t_layer;
    }
    if (layer->layer_type() == "ReLU") {
      t_relu += t_layer;
      t_pwl += t_layer;
      split_scale += layer->last_split_scale;
    }
    if (layer->layer_type() == "ArgMax") {
      t_argmax += t_layer;
      t_pwl += t_layer;
      split_scale += layer->last_split_scale;
    }
    t_total += t_layer;

#ifdef TIME_TRANSFORM_TOP
    std::cerr << "time: " << t_layer << " ms" << std::endl;
    std::cerr << "n polytopes: " << upolytope->n_polytopes() << std::endl;
    std::cerr << "n vertices: " << upolytope->n_vertices() << std::endl;
    std::cerr << "n dims: " << upolytope->space_dimensions() << std::endl;
    // std::cerr << "n combinations: " << upolytope->combinations_device().rows() << std::endl;
    // std::cerr << "n edges: " << upolytope->n_edges() << std::endl;
#endif
  }

// #ifdef TIME_TRANSFORM_TOP
  std::cerr << "------ end transform -------\n";
  std::cerr << "  FC    : " << t_fc << " ms" << std::endl;
  std::cerr << "  Conv2D: " << t_conv2d << " ms" << std::endl;
  std::cerr << "  Norm  : " << t_norm << " ms" << std::endl;
  std::cerr << "Affine  : " << t_affine << " ms" << std::endl;
  std::cerr << "---------------------\n";
  std::cerr << "  ReLU  : " << t_relu << " ms" << std::endl;
  std::cerr << "  ArgMax: " << t_argmax << " ms" << std::endl;
  std::cerr << "PWL     : " << t_pwl << " ms" << std::endl;
  std::cerr << "---------------------\n";
  std::cerr << "Total   : " << t_total << std::endl;
  std::cerr << "---------------------\n";
  std::cerr << "n polytopes: " << upolytope->n_polytopes() << std::endl;
  std::cerr << "split scale: " << split_scale << std::endl;
// #endif

  timing.set_fc(t_fc);
  timing.set_conv2d(t_conv2d);
  timing.set_norm(t_norm);
  timing.set_relu(t_relu);
  timing.set_argmax(t_argmax);
  timing.set_affine(t_affine);
  timing.set_pwl(t_pwl);
  timing.set_total(t_total);

}

void compute_pre(
    UPolytope *upolytope,
    syrenn_server::Timing& timing) {

  double t_preimage = 0.;
  Timer t;
  t.Reset();

  upolytope->ComputePreImage();

  t_preimage = t.Ticks();
  std::cerr << "pre-image: " << t_preimage << std::endl;
  timing.set_preimage(t_preimage);
}

void fuse_classify(
    UPolytope *upolytope,
    syrenn_server::Timing& timing) {

  double t_classify_argmax = 0.;
  double t_classify_extract =0.;
  Timer t; t.Reset();
  ArgMaxTransformer().TransformUPolytopeWithoutCompute(upolytope);
  t_classify_argmax = t.Ticks();
  std::cerr << "transform argmax: " << t_classify_argmax << std::endl;

  t.Reset();
  upolytope->Classify();
  t_classify_extract = t.Ticks();
  std::cerr << "extract labels: " << t_classify_extract << std::endl;

  timing.set_classify_argmax(t_classify_argmax);
  timing.set_classify_extract(t_classify_extract);
}

class SyReNNTransformerImpl final
    : public syrenn_server::SyReNNTransformer::Service {

 public:
  SyReNNTransformerImpl()
      : syrenn_server::SyReNNTransformer::Service() {}

  void OptimizeNetwork(std::vector<std::unique_ptr<LayerTransformer>> *layers) {
    // for (size_t i = 0; (i + 1) < layers->size(); i++) {
    //   // Optimization #1: (ReLU + MaxPool) \vee (MaxPool + ReLU) -> ReLUMaxPool
    //   ReLUTransformer *as_relu =
    //       dynamic_cast<ReLUTransformer *>(layers->at(i).get());
    //   MaxPoolTransformer *next_as_maxpool =
    //       dynamic_cast<MaxPoolTransformer *>(layers->at(i + 1).get());
    //   if (!(as_relu && next_as_maxpool)) {
    //     // Try the other order.
    //     as_relu = dynamic_cast<ReLUTransformer *>(layers->at(i + 1).get());
    //     next_as_maxpool =
    //         dynamic_cast<MaxPoolTransformer *>(layers->at(i).get());
    //   }
    //   if (as_relu && next_as_maxpool) {
    //     auto fused = std::unique_ptr<LayerTransformer>(
    //         new ReLUMaxPoolTransformer(*next_as_maxpool));
    //     std::swap(layers->at(i), fused);
    //     layers->erase(layers->begin() + i + 1);
    //   }
    // }
  }

  Status Transform(
      ServerContext *context,
      ServerReaderWriter<TransformResponse, TransformRequest> *stream) {
    std::vector<std::unique_ptr<LayerTransformer>> layer_transformers;
    Timer timer;

    TransformRequest request;
    while (stream->Read(&request)) {
      if (request.has_layer()) {
        // Add layers to the network.
        layer_transformers.push_back(
            LayerTransformer::Deserialize(request.layer(), request.device()));
        OptimizeNetwork(&layer_transformers);
      } else if (request.has_line()) {
        // Transform a segmented line.
        SegmentedLine pre_line = SegmentedLine::Deserialize(request.line());
        Timer timer;
        timer.Reset();
        auto transformed = transform_line(pre_line, layer_transformers,
                                          request.include_post());
        TransformResponse response;
        *(response.mutable_transformed_line()) =
            SegmentedLine::Serialize(transformed.first, transformed.second);
        stream->Write(response);
      } else if (request.has_upolytope()) {
        // Transform a UPolytope.
        UPolytope upolytope = UPolytope::Deserialize(request.upolytope(), request.device());
        upolytope.include_post_ = request.include_post();
        auto timing = syrenn_server::Timing();

        uint64_t split_scale = 0;
        // PWLTransformer::ReserveMemory();
        transform_polytope(&upolytope, layer_transformers, timing, split_scale);
        // PWLTransformer::ReleaseMemory();
        // for (auto & layer: layer_transformers) {
        //   layer.release();
        // }

        uint64_t fhat_size = upolytope.n_polytopes();

        // if (!request.include_post()) {
        //   upolytope.vertices_device().device().clear();
        //   upolytope.vertices_device().device().shrink_to_fit();
        // }
        if (request.fuse_classify()) {
          fuse_classify(&upolytope, timing);
        }
        if (request.compute_pre()) {
          compute_pre(&upolytope, timing);
        }

        TransformResponse response;
        if (request.dry_run()) {
          *(response.mutable_transformed_upolytope()) = syrenn_server::UPolytope();
          *(response.mutable_transformed_upolytope()->mutable_compressed()) = syrenn_server::UPolytopeCompressed();
        } else {
          *(response.mutable_transformed_upolytope()) = upolytope.Serialize();
        }
        response.mutable_transformed_upolytope()->set_fhat_size(fhat_size);
        response.mutable_transformed_upolytope()->set_split_scale(split_scale);
        *(response.mutable_timing()) = timing;
        stream->Write(response);
      }
    }
    return Status::OK;
  }
};

int main(int argc, char** argv) {
  std::string server_address("0.0.0.0:50051");
  SyReNNTransformerImpl service;

  LayerTransformer::RegisterDeserializer(
      &FullyConnectedTransformer::Deserialize);
  LayerTransformer::RegisterDeserializer(&Conv2DTransformer::Deserialize);
  LayerTransformer::RegisterDeserializer(&NormalizeTransformer::Deserialize);
  LayerTransformer::RegisterDeserializer(&ReLUTransformer::Deserialize);
  // LayerTransformer::RegisterDeserializer(&HardTanhTransformer::Deserialize);
  // LayerTransformer::RegisterDeserializer(&MaxPoolTransformer::Deserialize);
  // LayerTransformer::RegisterDeserializer(&AveragePoolTransformer::Deserialize);
  // LayerTransformer::RegisterDeserializer(&ConcatTransformer::Deserialize);
  LayerTransformer::RegisterDeserializer(&ArgMaxTransformer::Deserialize);

  CUDAShared::init();

  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(32 * 4194304);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();

  return 0;
}
