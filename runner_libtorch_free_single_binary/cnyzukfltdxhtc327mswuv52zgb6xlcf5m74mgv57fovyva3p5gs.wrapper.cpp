
#include <torch/csrc/inductor/aoti_include/cuda.h>
#include <torch/csrc/inductor/aoti_neutron/cuda/c_shim_cuda.h>
// Definition of AOTI runtime interface functions

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>
#ifdef AOTI_LIBTORCH_FREE
#include <torch/csrc/inductor/aoti_neutron/slim_tensor.h>
#endif

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_RUNTIME_FAILURE;                             \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_RUNTIME_FAILURE;                             \
  }                                                          \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() : prev_mode(aoti_torch_grad_mode_is_enabled()) {
    aoti_torch_grad_mode_set_enabled(false);
  }
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  bool prev_mode;
};

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0" << std::endl;
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerFlattenedRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    void** input_handles,
    size_t num_inputs,
    void** output_handles,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
#ifdef AOTI_LIBTORCH_FREE
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  using FlattenedTensor = std::tuple<
      void*, // data_ptr
      const int64_t*, // sizes
      const int64_t*, // strides
      int64_t, // dim
      int32_t, // dtype
      int32_t, // device_type
      int32_t, // device_index
      int64_t // storage_offset
      >;

  std::vector<torch::neutron::SlimTensor*> inputs;
  inputs.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    FlattenedTensor* tuple = reinterpret_cast<FlattenedTensor*>(input_handles[i]);
    IntArrayRef sizes(std::get<1>(*tuple), std::get<3>(*tuple));
    IntArrayRef strides(std::get<2>(*tuple), std::get<3>(*tuple));
    inputs.push_back(new torch::neutron::SlimTensor(
      torch::neutron::create_tensor_from_blob(
        std::get<0>(*tuple),
        sizes,
        strides,
        // dtype is 1-to-1 mapping for now
        static_cast<torch::neutron::ScalarType>(std::get<4>(*tuple)),
        // device_type is 1-to-1 mapping for now
        {static_cast<torch::neutron::DeviceType>(std::get<5>(*tuple)),
        static_cast<torch::neutron::DeviceIndex>(std::get<6>(*tuple))},
        std::get<7>(*tuple))));
    delete tuple;
  }

  auto stream =
    reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);

  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    std::vector<torch::neutron::SlimTensor*> outputs(num_outputs);

    container->run_single_threaded(
        inputs.data(), outputs.data(), stream, proxy_executor_handle);

    std::vector<torch::neutron::SlimTensor*> inputs;
    for (size_t i = 0; i < num_outputs; i++) {
      torch::neutron::SlimTensor* tensor = outputs[i];
      FlattenedTensor *tuple = new FlattenedTensor(
        tensor->data_ptr(),
        tensor->sizes().data(),
        tensor->strides().data(),
        tensor->dim(),
        static_cast<int32_t>(tensor->dtype()),
        static_cast<int32_t>(tensor->device_type()),
        static_cast<int32_t>(tensor->device_index()),
        tensor->storage_offset()
        );
      output_handles[i] = static_cast<void*>(tuple);
      // Set storage to non-owning and transfer storage ownership to at::Tensor
      tensor->storage()->unsafe_set_to_non_owning();
      delete tensor;
    }
  })
#else
  // This function is only provided in the libtorch-free mode
  return AOTI_RUNTIME_FAILURE;
#endif
}


AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *type = container->constant_type(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto constants_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { const auto ret = container->extract_constants_map(use_inactive);
      for (const auto& pair: ret) {
        constants_map->emplace(pair.first, pair.second);
      }
    })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
          constant_map_handle,
          /*use_inactive*/ true,
          /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->free_inactive_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

} // extern "C"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_neutron/cuda/c_shim_cuda.h>

namespace torch::neutron {
namespace {

void sgemm_cublas(
    SlimTensor& out,
    SlimTensor& A,
    SlimTensor& B,
    SlimTensor& C,
    float beta,
    float alpha) {
  // out = alpha* A @ B + beta * C
  // TODO: check contiguous and tranform if needed

  cublasHandle_t handle{};
  cublasCreate(&handle);

  int m = A.size(0);
  int k = A.size(1);
  int n = B.size(1);

  if (C.data_ptr() != out.data_ptr()) {
    // HACK
    for (int64_t i = 0; i < m; i++) {
      cudaMemcpy(
          static_cast<float*>(out.data_ptr()) + i * n,
          C.data_ptr(),
          n * sizeof(float),
          cudaMemcpyDeviceToDevice);
    }
  }

  cublasSgemm(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_T,
      n,
      m,
      k,
      &alpha,
      static_cast<const float*>(B.data_ptr()),
      n,
      static_cast<const float*>(A.data_ptr()),
      k,
      &beta, // Compute mm only; add bias in the next step
      static_cast<float*>(out.data_ptr()),
      n);

  /*
  // Compiling .cu files needs extra work to codecache.py and cpp_builder.py
  __global__ void add_bias_kernel(float* output, const float* bias, int
  batch_size, int out_features) {
      // Calculate global thread ID
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      // Check if within bounds
      if (idx < batch_size * out_features) {
          // Get row and column indices
          int row = idx / out_features;
          int col = idx % out_features;

          // Add bias[col] to output[row][col]
          output[idx] += bias[col];
      }
  }
  */

  cublasDestroy(handle);
}

} // namespace
} // namespace torch::neutron

/*
AOTITorchError aoti_torch_cuda_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  if (self->dim() == 2 && mat2->dim() == 2 &&
      self->dtype() == torch::neutron::ScalarType::_float32 &&
      mat2->dtype() == torch::neutron::ScalarType::_float32) {
    torch::neutron::sgemm_cublas(*out, *self, *mat2, *out, 0.0f, 1.0f);
  } else {
    throw std::runtime_error("matmul only supports float32 tensors");
  }
  return AOTI_TORCH_SUCCESS;
}
*/

AOTITorchError aoti_torch_cuda_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha) {
  if (out->dtype() == torch::neutron::ScalarType::_float32 &&
      self->dtype() == torch::neutron::ScalarType::_float32 &&
      mat1->dtype() == torch::neutron::ScalarType::_float32 &&
      mat2->dtype() == torch::neutron::ScalarType::_float32) {
    torch::neutron::sgemm_cublas(*out, *mat1, *mat2, *self, beta, alpha);
  } else {
    throw std::runtime_error("matmul only supports float32 tensors");
  }
  return AOTI_TORCH_SUCCESS;
}

#endif // USE_CUDA


#define CUDA_DRIVER_CHECK(EXPR)                    \
do {                                               \
    CUresult code = EXPR;                          \
    const char *msg;                               \
    CUresult code_get_error = cuGetErrorString(code, &msg); \
    if (code_get_error != CUDA_SUCCESS) {          \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string("invalid error code!"));   \
    }                                              \
    if (code != CUDA_SUCCESS) {                    \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string(msg));                     \
    }                                              \
} while (0);

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline CUfunction loadKernel(const void* start, const std::string &funcName, uint32_t sharedMemBytes) {
    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoadData(&mod, start));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DTYPE(bfloat16);
CACHE_TORCH_DTYPE(bool);
CACHE_TORCH_DEVICE(cuda);
CACHE_TORCH_LAYOUT(strided);
namespace torch::aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    CUfunction triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6{nullptr};
    CUfunction triton_per_fused_bmm_8{nullptr};
    CUfunction triton_poi_fused__to_copy_mul_4{nullptr};
    CUfunction triton_poi_fused_index_put_3{nullptr};
    CUfunction triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10{nullptr};
    CUfunction triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13{nullptr};
    CUfunction triton_red_fused__to_copy_add_mean_mul_pow_19{nullptr};
    CUfunction triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22{nullptr};
    CUfunction triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23{nullptr};
    CUfunction triton_red_fused__to_copy_embedding_mean_pow_0{nullptr};
    CUfunction triton_red_fused__to_copy_mean_pow_17{nullptr};
    CUfunction triton_red_fused_add_embedding_mm_mul_16{nullptr};
    CUfunction triton_red_fused_add_mm_mul_24{nullptr};
    CUfunction triton_red_fused_bmm_5{nullptr};
    CUfunction triton_red_fused_bmm_7{nullptr};
    CUfunction triton_red_fused_index_put_mm_15{nullptr};
    CUfunction triton_red_fused_index_put_mm_2{nullptr};
    CUfunction triton_red_fused_index_put_mm_21{nullptr};
    CUfunction triton_red_fused_mm_1{nullptr};
    CUfunction triton_red_fused_mm_11{nullptr};
    CUfunction triton_red_fused_mm_12{nullptr};
    CUfunction triton_red_fused_mm_14{nullptr};
    CUfunction triton_red_fused_mm_18{nullptr};
    CUfunction triton_red_fused_mm_20{nullptr};
    CUfunction triton_red_fused_mm_25{nullptr};
    CUfunction triton_red_fused_mm_9{nullptr};
};
}  // namespace


extern "C" {
    extern const unsigned char __triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6_start[];
    extern const unsigned char __triton_per_fused_bmm_8_start[];
    extern const unsigned char __triton_poi_fused__to_copy_mul_4_start[];
    extern const unsigned char __triton_poi_fused_index_put_3_start[];
    extern const unsigned char __triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10_start[];
    extern const unsigned char __triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13_start[];
    extern const unsigned char __triton_red_fused__to_copy_add_mean_mul_pow_19_start[];
    extern const unsigned char __triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22_start[];
    extern const unsigned char __triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23_start[];
    extern const unsigned char __triton_red_fused__to_copy_embedding_mean_pow_0_start[];
    extern const unsigned char __triton_red_fused__to_copy_mean_pow_17_start[];
    extern const unsigned char __triton_red_fused_add_embedding_mm_mul_16_start[];
    extern const unsigned char __triton_red_fused_add_mm_mul_24_start[];
    extern const unsigned char __triton_red_fused_bmm_5_start[];
    extern const unsigned char __triton_red_fused_bmm_7_start[];
    extern const unsigned char __triton_red_fused_index_put_mm_15_start[];
    extern const unsigned char __triton_red_fused_index_put_mm_2_start[];
    extern const unsigned char __triton_red_fused_index_put_mm_21_start[];
    extern const unsigned char __triton_red_fused_mm_1_start[];
    extern const unsigned char __triton_red_fused_mm_11_start[];
    extern const unsigned char __triton_red_fused_mm_12_start[];
    extern const unsigned char __triton_red_fused_mm_14_start[];
    extern const unsigned char __triton_red_fused_mm_18_start[];
    extern const unsigned char __triton_red_fused_mm_20_start[];
    extern const unsigned char __triton_red_fused_mm_25_start[];
    extern const unsigned char __triton_red_fused_mm_9_start[];
}

AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir,
                                   bool include_weights)
    : AOTInductorModelBase(2, 1, 357, device_str, cubin_dir, true) {
    inputs_info_[0].name = "arg357_1";
    inputs_info_[1].name = "arg358_1";
    constants_info_[0].name = "model_tok_embeddings_weight";
    constants_info_[0].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 1050673152;
    constants_info_[0].from_folded = false;
    constants_info_[0].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[0].shape = {128256, 4096};
    constants_info_[0].stride = {4096, 1};
    constants_info_[0].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[0].original_fqn = "model.tok_embeddings.weight";
    constants_info_[1].name = "model_layers_0_attention_wq_weight";
    constants_info_[1].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 33554432;
    constants_info_[1].from_folded = false;
    constants_info_[1].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[1].shape = {4096, 4096};
    constants_info_[1].stride = {4096, 1};
    constants_info_[1].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[1].original_fqn = "model.layers.0.attention.wq.weight";
    constants_info_[2].name = "model_layers_0_attention_wk_weight";
    constants_info_[2].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 8388608;
    constants_info_[2].from_folded = false;
    constants_info_[2].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[2].shape = {1024, 4096};
    constants_info_[2].stride = {4096, 1};
    constants_info_[2].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[2].original_fqn = "model.layers.0.attention.wk.weight";
    constants_info_[3].name = "model_layers_0_attention_wv_weight";
    constants_info_[3].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[3].offset = 0;
    constants_info_[3].data_size = 8388608;
    constants_info_[3].from_folded = false;
    constants_info_[3].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[3].shape = {1024, 4096};
    constants_info_[3].stride = {4096, 1};
    constants_info_[3].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[3].original_fqn = "model.layers.0.attention.wv.weight";
    constants_info_[4].name = "model_layers_0_attention_wo_weight";
    constants_info_[4].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[4].offset = 0;
    constants_info_[4].data_size = 33554432;
    constants_info_[4].from_folded = false;
    constants_info_[4].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[4].shape = {4096, 4096};
    constants_info_[4].stride = {4096, 1};
    constants_info_[4].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[4].original_fqn = "model.layers.0.attention.wo.weight";
    constants_info_[5].name = "model_layers_0_feed_forward_w1_weight";
    constants_info_[5].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[5].offset = 0;
    constants_info_[5].data_size = 117440512;
    constants_info_[5].from_folded = false;
    constants_info_[5].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[5].shape = {14336, 4096};
    constants_info_[5].stride = {4096, 1};
    constants_info_[5].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[5].original_fqn = "model.layers.0.feed_forward.w1.weight";
    constants_info_[6].name = "model_layers_0_feed_forward_w2_weight";
    constants_info_[6].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[6].offset = 0;
    constants_info_[6].data_size = 117440512;
    constants_info_[6].from_folded = false;
    constants_info_[6].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[6].shape = {4096, 14336};
    constants_info_[6].stride = {14336, 1};
    constants_info_[6].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[6].original_fqn = "model.layers.0.feed_forward.w2.weight";
    constants_info_[7].name = "model_layers_0_feed_forward_w3_weight";
    constants_info_[7].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[7].offset = 0;
    constants_info_[7].data_size = 117440512;
    constants_info_[7].from_folded = false;
    constants_info_[7].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[7].shape = {14336, 4096};
    constants_info_[7].stride = {4096, 1};
    constants_info_[7].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[7].original_fqn = "model.layers.0.feed_forward.w3.weight";
    constants_info_[8].name = "model_layers_0_ffn_norm_weight";
    constants_info_[8].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[8].offset = 0;
    constants_info_[8].data_size = 8192;
    constants_info_[8].from_folded = false;
    constants_info_[8].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[8].shape = {4096};
    constants_info_[8].stride = {1};
    constants_info_[8].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[8].original_fqn = "model.layers.0.ffn_norm.weight";
    constants_info_[9].name = "model_layers_0_attention_norm_weight";
    constants_info_[9].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[9].offset = 0;
    constants_info_[9].data_size = 8192;
    constants_info_[9].from_folded = false;
    constants_info_[9].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[9].shape = {4096};
    constants_info_[9].stride = {1};
    constants_info_[9].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[9].original_fqn = "model.layers.0.attention_norm.weight";
    constants_info_[10].name = "model_layers_1_attention_wq_weight";
    constants_info_[10].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[10].offset = 0;
    constants_info_[10].data_size = 33554432;
    constants_info_[10].from_folded = false;
    constants_info_[10].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[10].shape = {4096, 4096};
    constants_info_[10].stride = {4096, 1};
    constants_info_[10].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[10].original_fqn = "model.layers.1.attention.wq.weight";
    constants_info_[11].name = "model_layers_1_attention_wk_weight";
    constants_info_[11].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[11].offset = 0;
    constants_info_[11].data_size = 8388608;
    constants_info_[11].from_folded = false;
    constants_info_[11].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[11].shape = {1024, 4096};
    constants_info_[11].stride = {4096, 1};
    constants_info_[11].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[11].original_fqn = "model.layers.1.attention.wk.weight";
    constants_info_[12].name = "model_layers_1_attention_wv_weight";
    constants_info_[12].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[12].offset = 0;
    constants_info_[12].data_size = 8388608;
    constants_info_[12].from_folded = false;
    constants_info_[12].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[12].shape = {1024, 4096};
    constants_info_[12].stride = {4096, 1};
    constants_info_[12].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[12].original_fqn = "model.layers.1.attention.wv.weight";
    constants_info_[13].name = "model_layers_1_attention_wo_weight";
    constants_info_[13].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[13].offset = 0;
    constants_info_[13].data_size = 33554432;
    constants_info_[13].from_folded = false;
    constants_info_[13].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[13].shape = {4096, 4096};
    constants_info_[13].stride = {4096, 1};
    constants_info_[13].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[13].original_fqn = "model.layers.1.attention.wo.weight";
    constants_info_[14].name = "model_layers_1_feed_forward_w1_weight";
    constants_info_[14].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[14].offset = 0;
    constants_info_[14].data_size = 117440512;
    constants_info_[14].from_folded = false;
    constants_info_[14].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[14].shape = {14336, 4096};
    constants_info_[14].stride = {4096, 1};
    constants_info_[14].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[14].original_fqn = "model.layers.1.feed_forward.w1.weight";
    constants_info_[15].name = "model_layers_1_feed_forward_w2_weight";
    constants_info_[15].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[15].offset = 0;
    constants_info_[15].data_size = 117440512;
    constants_info_[15].from_folded = false;
    constants_info_[15].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[15].shape = {4096, 14336};
    constants_info_[15].stride = {14336, 1};
    constants_info_[15].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[15].original_fqn = "model.layers.1.feed_forward.w2.weight";
    constants_info_[16].name = "model_layers_1_feed_forward_w3_weight";
    constants_info_[16].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[16].offset = 0;
    constants_info_[16].data_size = 117440512;
    constants_info_[16].from_folded = false;
    constants_info_[16].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[16].shape = {14336, 4096};
    constants_info_[16].stride = {4096, 1};
    constants_info_[16].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[16].original_fqn = "model.layers.1.feed_forward.w3.weight";
    constants_info_[17].name = "model_layers_1_ffn_norm_weight";
    constants_info_[17].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[17].offset = 0;
    constants_info_[17].data_size = 8192;
    constants_info_[17].from_folded = false;
    constants_info_[17].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[17].shape = {4096};
    constants_info_[17].stride = {1};
    constants_info_[17].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[17].original_fqn = "model.layers.1.ffn_norm.weight";
    constants_info_[18].name = "model_layers_1_attention_norm_weight";
    constants_info_[18].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[18].offset = 0;
    constants_info_[18].data_size = 8192;
    constants_info_[18].from_folded = false;
    constants_info_[18].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[18].shape = {4096};
    constants_info_[18].stride = {1};
    constants_info_[18].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[18].original_fqn = "model.layers.1.attention_norm.weight";
    constants_info_[19].name = "model_layers_2_attention_wq_weight";
    constants_info_[19].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[19].offset = 0;
    constants_info_[19].data_size = 33554432;
    constants_info_[19].from_folded = false;
    constants_info_[19].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[19].shape = {4096, 4096};
    constants_info_[19].stride = {4096, 1};
    constants_info_[19].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[19].original_fqn = "model.layers.2.attention.wq.weight";
    constants_info_[20].name = "model_layers_2_attention_wk_weight";
    constants_info_[20].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[20].offset = 0;
    constants_info_[20].data_size = 8388608;
    constants_info_[20].from_folded = false;
    constants_info_[20].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[20].shape = {1024, 4096};
    constants_info_[20].stride = {4096, 1};
    constants_info_[20].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[20].original_fqn = "model.layers.2.attention.wk.weight";
    constants_info_[21].name = "model_layers_2_attention_wv_weight";
    constants_info_[21].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[21].offset = 0;
    constants_info_[21].data_size = 8388608;
    constants_info_[21].from_folded = false;
    constants_info_[21].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[21].shape = {1024, 4096};
    constants_info_[21].stride = {4096, 1};
    constants_info_[21].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[21].original_fqn = "model.layers.2.attention.wv.weight";
    constants_info_[22].name = "model_layers_2_attention_wo_weight";
    constants_info_[22].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[22].offset = 0;
    constants_info_[22].data_size = 33554432;
    constants_info_[22].from_folded = false;
    constants_info_[22].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[22].shape = {4096, 4096};
    constants_info_[22].stride = {4096, 1};
    constants_info_[22].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[22].original_fqn = "model.layers.2.attention.wo.weight";
    constants_info_[23].name = "model_layers_2_feed_forward_w1_weight";
    constants_info_[23].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[23].offset = 0;
    constants_info_[23].data_size = 117440512;
    constants_info_[23].from_folded = false;
    constants_info_[23].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[23].shape = {14336, 4096};
    constants_info_[23].stride = {4096, 1};
    constants_info_[23].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[23].original_fqn = "model.layers.2.feed_forward.w1.weight";
    constants_info_[24].name = "model_layers_2_feed_forward_w2_weight";
    constants_info_[24].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[24].offset = 0;
    constants_info_[24].data_size = 117440512;
    constants_info_[24].from_folded = false;
    constants_info_[24].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[24].shape = {4096, 14336};
    constants_info_[24].stride = {14336, 1};
    constants_info_[24].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[24].original_fqn = "model.layers.2.feed_forward.w2.weight";
    constants_info_[25].name = "model_layers_2_feed_forward_w3_weight";
    constants_info_[25].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[25].offset = 0;
    constants_info_[25].data_size = 117440512;
    constants_info_[25].from_folded = false;
    constants_info_[25].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[25].shape = {14336, 4096};
    constants_info_[25].stride = {4096, 1};
    constants_info_[25].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[25].original_fqn = "model.layers.2.feed_forward.w3.weight";
    constants_info_[26].name = "model_layers_2_ffn_norm_weight";
    constants_info_[26].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[26].offset = 0;
    constants_info_[26].data_size = 8192;
    constants_info_[26].from_folded = false;
    constants_info_[26].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[26].shape = {4096};
    constants_info_[26].stride = {1};
    constants_info_[26].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[26].original_fqn = "model.layers.2.ffn_norm.weight";
    constants_info_[27].name = "model_layers_2_attention_norm_weight";
    constants_info_[27].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[27].offset = 0;
    constants_info_[27].data_size = 8192;
    constants_info_[27].from_folded = false;
    constants_info_[27].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[27].shape = {4096};
    constants_info_[27].stride = {1};
    constants_info_[27].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[27].original_fqn = "model.layers.2.attention_norm.weight";
    constants_info_[28].name = "model_layers_3_attention_wq_weight";
    constants_info_[28].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[28].offset = 0;
    constants_info_[28].data_size = 33554432;
    constants_info_[28].from_folded = false;
    constants_info_[28].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[28].shape = {4096, 4096};
    constants_info_[28].stride = {4096, 1};
    constants_info_[28].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[28].original_fqn = "model.layers.3.attention.wq.weight";
    constants_info_[29].name = "model_layers_3_attention_wk_weight";
    constants_info_[29].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[29].offset = 0;
    constants_info_[29].data_size = 8388608;
    constants_info_[29].from_folded = false;
    constants_info_[29].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[29].shape = {1024, 4096};
    constants_info_[29].stride = {4096, 1};
    constants_info_[29].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[29].original_fqn = "model.layers.3.attention.wk.weight";
    constants_info_[30].name = "model_layers_3_attention_wv_weight";
    constants_info_[30].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[30].offset = 0;
    constants_info_[30].data_size = 8388608;
    constants_info_[30].from_folded = false;
    constants_info_[30].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[30].shape = {1024, 4096};
    constants_info_[30].stride = {4096, 1};
    constants_info_[30].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[30].original_fqn = "model.layers.3.attention.wv.weight";
    constants_info_[31].name = "model_layers_3_attention_wo_weight";
    constants_info_[31].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[31].offset = 0;
    constants_info_[31].data_size = 33554432;
    constants_info_[31].from_folded = false;
    constants_info_[31].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[31].shape = {4096, 4096};
    constants_info_[31].stride = {4096, 1};
    constants_info_[31].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[31].original_fqn = "model.layers.3.attention.wo.weight";
    constants_info_[32].name = "model_layers_3_feed_forward_w1_weight";
    constants_info_[32].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[32].offset = 0;
    constants_info_[32].data_size = 117440512;
    constants_info_[32].from_folded = false;
    constants_info_[32].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[32].shape = {14336, 4096};
    constants_info_[32].stride = {4096, 1};
    constants_info_[32].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[32].original_fqn = "model.layers.3.feed_forward.w1.weight";
    constants_info_[33].name = "model_layers_3_feed_forward_w2_weight";
    constants_info_[33].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[33].offset = 0;
    constants_info_[33].data_size = 117440512;
    constants_info_[33].from_folded = false;
    constants_info_[33].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[33].shape = {4096, 14336};
    constants_info_[33].stride = {14336, 1};
    constants_info_[33].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[33].original_fqn = "model.layers.3.feed_forward.w2.weight";
    constants_info_[34].name = "model_layers_3_feed_forward_w3_weight";
    constants_info_[34].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[34].offset = 0;
    constants_info_[34].data_size = 117440512;
    constants_info_[34].from_folded = false;
    constants_info_[34].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[34].shape = {14336, 4096};
    constants_info_[34].stride = {4096, 1};
    constants_info_[34].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[34].original_fqn = "model.layers.3.feed_forward.w3.weight";
    constants_info_[35].name = "model_layers_3_ffn_norm_weight";
    constants_info_[35].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[35].offset = 0;
    constants_info_[35].data_size = 8192;
    constants_info_[35].from_folded = false;
    constants_info_[35].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[35].shape = {4096};
    constants_info_[35].stride = {1};
    constants_info_[35].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[35].original_fqn = "model.layers.3.ffn_norm.weight";
    constants_info_[36].name = "model_layers_3_attention_norm_weight";
    constants_info_[36].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[36].offset = 0;
    constants_info_[36].data_size = 8192;
    constants_info_[36].from_folded = false;
    constants_info_[36].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[36].shape = {4096};
    constants_info_[36].stride = {1};
    constants_info_[36].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[36].original_fqn = "model.layers.3.attention_norm.weight";
    constants_info_[37].name = "model_layers_4_attention_wq_weight";
    constants_info_[37].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[37].offset = 0;
    constants_info_[37].data_size = 33554432;
    constants_info_[37].from_folded = false;
    constants_info_[37].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[37].shape = {4096, 4096};
    constants_info_[37].stride = {4096, 1};
    constants_info_[37].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[37].original_fqn = "model.layers.4.attention.wq.weight";
    constants_info_[38].name = "model_layers_4_attention_wk_weight";
    constants_info_[38].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[38].offset = 0;
    constants_info_[38].data_size = 8388608;
    constants_info_[38].from_folded = false;
    constants_info_[38].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[38].shape = {1024, 4096};
    constants_info_[38].stride = {4096, 1};
    constants_info_[38].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[38].original_fqn = "model.layers.4.attention.wk.weight";
    constants_info_[39].name = "model_layers_4_attention_wv_weight";
    constants_info_[39].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[39].offset = 0;
    constants_info_[39].data_size = 8388608;
    constants_info_[39].from_folded = false;
    constants_info_[39].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[39].shape = {1024, 4096};
    constants_info_[39].stride = {4096, 1};
    constants_info_[39].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[39].original_fqn = "model.layers.4.attention.wv.weight";
    constants_info_[40].name = "model_layers_4_attention_wo_weight";
    constants_info_[40].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[40].offset = 0;
    constants_info_[40].data_size = 33554432;
    constants_info_[40].from_folded = false;
    constants_info_[40].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[40].shape = {4096, 4096};
    constants_info_[40].stride = {4096, 1};
    constants_info_[40].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[40].original_fqn = "model.layers.4.attention.wo.weight";
    constants_info_[41].name = "model_layers_4_feed_forward_w1_weight";
    constants_info_[41].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[41].offset = 0;
    constants_info_[41].data_size = 117440512;
    constants_info_[41].from_folded = false;
    constants_info_[41].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[41].shape = {14336, 4096};
    constants_info_[41].stride = {4096, 1};
    constants_info_[41].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[41].original_fqn = "model.layers.4.feed_forward.w1.weight";
    constants_info_[42].name = "model_layers_4_feed_forward_w2_weight";
    constants_info_[42].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[42].offset = 0;
    constants_info_[42].data_size = 117440512;
    constants_info_[42].from_folded = false;
    constants_info_[42].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[42].shape = {4096, 14336};
    constants_info_[42].stride = {14336, 1};
    constants_info_[42].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[42].original_fqn = "model.layers.4.feed_forward.w2.weight";
    constants_info_[43].name = "model_layers_4_feed_forward_w3_weight";
    constants_info_[43].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[43].offset = 0;
    constants_info_[43].data_size = 117440512;
    constants_info_[43].from_folded = false;
    constants_info_[43].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[43].shape = {14336, 4096};
    constants_info_[43].stride = {4096, 1};
    constants_info_[43].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[43].original_fqn = "model.layers.4.feed_forward.w3.weight";
    constants_info_[44].name = "model_layers_4_ffn_norm_weight";
    constants_info_[44].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[44].offset = 0;
    constants_info_[44].data_size = 8192;
    constants_info_[44].from_folded = false;
    constants_info_[44].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[44].shape = {4096};
    constants_info_[44].stride = {1};
    constants_info_[44].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[44].original_fqn = "model.layers.4.ffn_norm.weight";
    constants_info_[45].name = "model_layers_4_attention_norm_weight";
    constants_info_[45].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[45].offset = 0;
    constants_info_[45].data_size = 8192;
    constants_info_[45].from_folded = false;
    constants_info_[45].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[45].shape = {4096};
    constants_info_[45].stride = {1};
    constants_info_[45].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[45].original_fqn = "model.layers.4.attention_norm.weight";
    constants_info_[46].name = "model_layers_5_attention_wq_weight";
    constants_info_[46].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[46].offset = 0;
    constants_info_[46].data_size = 33554432;
    constants_info_[46].from_folded = false;
    constants_info_[46].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[46].shape = {4096, 4096};
    constants_info_[46].stride = {4096, 1};
    constants_info_[46].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[46].original_fqn = "model.layers.5.attention.wq.weight";
    constants_info_[47].name = "model_layers_5_attention_wk_weight";
    constants_info_[47].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[47].offset = 0;
    constants_info_[47].data_size = 8388608;
    constants_info_[47].from_folded = false;
    constants_info_[47].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[47].shape = {1024, 4096};
    constants_info_[47].stride = {4096, 1};
    constants_info_[47].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[47].original_fqn = "model.layers.5.attention.wk.weight";
    constants_info_[48].name = "model_layers_5_attention_wv_weight";
    constants_info_[48].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[48].offset = 0;
    constants_info_[48].data_size = 8388608;
    constants_info_[48].from_folded = false;
    constants_info_[48].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[48].shape = {1024, 4096};
    constants_info_[48].stride = {4096, 1};
    constants_info_[48].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[48].original_fqn = "model.layers.5.attention.wv.weight";
    constants_info_[49].name = "model_layers_5_attention_wo_weight";
    constants_info_[49].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[49].offset = 0;
    constants_info_[49].data_size = 33554432;
    constants_info_[49].from_folded = false;
    constants_info_[49].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[49].shape = {4096, 4096};
    constants_info_[49].stride = {4096, 1};
    constants_info_[49].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[49].original_fqn = "model.layers.5.attention.wo.weight";
    constants_info_[50].name = "model_layers_5_feed_forward_w1_weight";
    constants_info_[50].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[50].offset = 0;
    constants_info_[50].data_size = 117440512;
    constants_info_[50].from_folded = false;
    constants_info_[50].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[50].shape = {14336, 4096};
    constants_info_[50].stride = {4096, 1};
    constants_info_[50].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[50].original_fqn = "model.layers.5.feed_forward.w1.weight";
    constants_info_[51].name = "model_layers_5_feed_forward_w2_weight";
    constants_info_[51].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[51].offset = 0;
    constants_info_[51].data_size = 117440512;
    constants_info_[51].from_folded = false;
    constants_info_[51].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[51].shape = {4096, 14336};
    constants_info_[51].stride = {14336, 1};
    constants_info_[51].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[51].original_fqn = "model.layers.5.feed_forward.w2.weight";
    constants_info_[52].name = "model_layers_5_feed_forward_w3_weight";
    constants_info_[52].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[52].offset = 0;
    constants_info_[52].data_size = 117440512;
    constants_info_[52].from_folded = false;
    constants_info_[52].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[52].shape = {14336, 4096};
    constants_info_[52].stride = {4096, 1};
    constants_info_[52].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[52].original_fqn = "model.layers.5.feed_forward.w3.weight";
    constants_info_[53].name = "model_layers_5_ffn_norm_weight";
    constants_info_[53].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[53].offset = 0;
    constants_info_[53].data_size = 8192;
    constants_info_[53].from_folded = false;
    constants_info_[53].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[53].shape = {4096};
    constants_info_[53].stride = {1};
    constants_info_[53].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[53].original_fqn = "model.layers.5.ffn_norm.weight";
    constants_info_[54].name = "model_layers_5_attention_norm_weight";
    constants_info_[54].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[54].offset = 0;
    constants_info_[54].data_size = 8192;
    constants_info_[54].from_folded = false;
    constants_info_[54].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[54].shape = {4096};
    constants_info_[54].stride = {1};
    constants_info_[54].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[54].original_fqn = "model.layers.5.attention_norm.weight";
    constants_info_[55].name = "model_layers_6_attention_wq_weight";
    constants_info_[55].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[55].offset = 0;
    constants_info_[55].data_size = 33554432;
    constants_info_[55].from_folded = false;
    constants_info_[55].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[55].shape = {4096, 4096};
    constants_info_[55].stride = {4096, 1};
    constants_info_[55].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[55].original_fqn = "model.layers.6.attention.wq.weight";
    constants_info_[56].name = "model_layers_6_attention_wk_weight";
    constants_info_[56].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[56].offset = 0;
    constants_info_[56].data_size = 8388608;
    constants_info_[56].from_folded = false;
    constants_info_[56].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[56].shape = {1024, 4096};
    constants_info_[56].stride = {4096, 1};
    constants_info_[56].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[56].original_fqn = "model.layers.6.attention.wk.weight";
    constants_info_[57].name = "model_layers_6_attention_wv_weight";
    constants_info_[57].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[57].offset = 0;
    constants_info_[57].data_size = 8388608;
    constants_info_[57].from_folded = false;
    constants_info_[57].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[57].shape = {1024, 4096};
    constants_info_[57].stride = {4096, 1};
    constants_info_[57].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[57].original_fqn = "model.layers.6.attention.wv.weight";
    constants_info_[58].name = "model_layers_6_attention_wo_weight";
    constants_info_[58].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[58].offset = 0;
    constants_info_[58].data_size = 33554432;
    constants_info_[58].from_folded = false;
    constants_info_[58].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[58].shape = {4096, 4096};
    constants_info_[58].stride = {4096, 1};
    constants_info_[58].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[58].original_fqn = "model.layers.6.attention.wo.weight";
    constants_info_[59].name = "model_layers_6_feed_forward_w1_weight";
    constants_info_[59].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[59].offset = 0;
    constants_info_[59].data_size = 117440512;
    constants_info_[59].from_folded = false;
    constants_info_[59].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[59].shape = {14336, 4096};
    constants_info_[59].stride = {4096, 1};
    constants_info_[59].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[59].original_fqn = "model.layers.6.feed_forward.w1.weight";
    constants_info_[60].name = "model_layers_6_feed_forward_w2_weight";
    constants_info_[60].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[60].offset = 0;
    constants_info_[60].data_size = 117440512;
    constants_info_[60].from_folded = false;
    constants_info_[60].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[60].shape = {4096, 14336};
    constants_info_[60].stride = {14336, 1};
    constants_info_[60].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[60].original_fqn = "model.layers.6.feed_forward.w2.weight";
    constants_info_[61].name = "model_layers_6_feed_forward_w3_weight";
    constants_info_[61].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[61].offset = 0;
    constants_info_[61].data_size = 117440512;
    constants_info_[61].from_folded = false;
    constants_info_[61].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[61].shape = {14336, 4096};
    constants_info_[61].stride = {4096, 1};
    constants_info_[61].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[61].original_fqn = "model.layers.6.feed_forward.w3.weight";
    constants_info_[62].name = "model_layers_6_ffn_norm_weight";
    constants_info_[62].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[62].offset = 0;
    constants_info_[62].data_size = 8192;
    constants_info_[62].from_folded = false;
    constants_info_[62].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[62].shape = {4096};
    constants_info_[62].stride = {1};
    constants_info_[62].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[62].original_fqn = "model.layers.6.ffn_norm.weight";
    constants_info_[63].name = "model_layers_6_attention_norm_weight";
    constants_info_[63].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[63].offset = 0;
    constants_info_[63].data_size = 8192;
    constants_info_[63].from_folded = false;
    constants_info_[63].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[63].shape = {4096};
    constants_info_[63].stride = {1};
    constants_info_[63].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[63].original_fqn = "model.layers.6.attention_norm.weight";
    constants_info_[64].name = "model_layers_7_attention_wq_weight";
    constants_info_[64].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[64].offset = 0;
    constants_info_[64].data_size = 33554432;
    constants_info_[64].from_folded = false;
    constants_info_[64].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[64].shape = {4096, 4096};
    constants_info_[64].stride = {4096, 1};
    constants_info_[64].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[64].original_fqn = "model.layers.7.attention.wq.weight";
    constants_info_[65].name = "model_layers_7_attention_wk_weight";
    constants_info_[65].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[65].offset = 0;
    constants_info_[65].data_size = 8388608;
    constants_info_[65].from_folded = false;
    constants_info_[65].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[65].shape = {1024, 4096};
    constants_info_[65].stride = {4096, 1};
    constants_info_[65].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[65].original_fqn = "model.layers.7.attention.wk.weight";
    constants_info_[66].name = "model_layers_7_attention_wv_weight";
    constants_info_[66].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[66].offset = 0;
    constants_info_[66].data_size = 8388608;
    constants_info_[66].from_folded = false;
    constants_info_[66].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[66].shape = {1024, 4096};
    constants_info_[66].stride = {4096, 1};
    constants_info_[66].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[66].original_fqn = "model.layers.7.attention.wv.weight";
    constants_info_[67].name = "model_layers_7_attention_wo_weight";
    constants_info_[67].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[67].offset = 0;
    constants_info_[67].data_size = 33554432;
    constants_info_[67].from_folded = false;
    constants_info_[67].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[67].shape = {4096, 4096};
    constants_info_[67].stride = {4096, 1};
    constants_info_[67].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[67].original_fqn = "model.layers.7.attention.wo.weight";
    constants_info_[68].name = "model_layers_7_feed_forward_w1_weight";
    constants_info_[68].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[68].offset = 0;
    constants_info_[68].data_size = 117440512;
    constants_info_[68].from_folded = false;
    constants_info_[68].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[68].shape = {14336, 4096};
    constants_info_[68].stride = {4096, 1};
    constants_info_[68].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[68].original_fqn = "model.layers.7.feed_forward.w1.weight";
    constants_info_[69].name = "model_layers_7_feed_forward_w2_weight";
    constants_info_[69].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[69].offset = 0;
    constants_info_[69].data_size = 117440512;
    constants_info_[69].from_folded = false;
    constants_info_[69].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[69].shape = {4096, 14336};
    constants_info_[69].stride = {14336, 1};
    constants_info_[69].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[69].original_fqn = "model.layers.7.feed_forward.w2.weight";
    constants_info_[70].name = "model_layers_7_feed_forward_w3_weight";
    constants_info_[70].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[70].offset = 0;
    constants_info_[70].data_size = 117440512;
    constants_info_[70].from_folded = false;
    constants_info_[70].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[70].shape = {14336, 4096};
    constants_info_[70].stride = {4096, 1};
    constants_info_[70].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[70].original_fqn = "model.layers.7.feed_forward.w3.weight";
    constants_info_[71].name = "model_layers_7_ffn_norm_weight";
    constants_info_[71].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[71].offset = 0;
    constants_info_[71].data_size = 8192;
    constants_info_[71].from_folded = false;
    constants_info_[71].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[71].shape = {4096};
    constants_info_[71].stride = {1};
    constants_info_[71].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[71].original_fqn = "model.layers.7.ffn_norm.weight";
    constants_info_[72].name = "model_layers_7_attention_norm_weight";
    constants_info_[72].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[72].offset = 0;
    constants_info_[72].data_size = 8192;
    constants_info_[72].from_folded = false;
    constants_info_[72].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[72].shape = {4096};
    constants_info_[72].stride = {1};
    constants_info_[72].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[72].original_fqn = "model.layers.7.attention_norm.weight";
    constants_info_[73].name = "model_layers_8_attention_wq_weight";
    constants_info_[73].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[73].offset = 0;
    constants_info_[73].data_size = 33554432;
    constants_info_[73].from_folded = false;
    constants_info_[73].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[73].shape = {4096, 4096};
    constants_info_[73].stride = {4096, 1};
    constants_info_[73].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[73].original_fqn = "model.layers.8.attention.wq.weight";
    constants_info_[74].name = "model_layers_8_attention_wk_weight";
    constants_info_[74].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[74].offset = 0;
    constants_info_[74].data_size = 8388608;
    constants_info_[74].from_folded = false;
    constants_info_[74].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[74].shape = {1024, 4096};
    constants_info_[74].stride = {4096, 1};
    constants_info_[74].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[74].original_fqn = "model.layers.8.attention.wk.weight";
    constants_info_[75].name = "model_layers_8_attention_wv_weight";
    constants_info_[75].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[75].offset = 0;
    constants_info_[75].data_size = 8388608;
    constants_info_[75].from_folded = false;
    constants_info_[75].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[75].shape = {1024, 4096};
    constants_info_[75].stride = {4096, 1};
    constants_info_[75].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[75].original_fqn = "model.layers.8.attention.wv.weight";
    constants_info_[76].name = "model_layers_8_attention_wo_weight";
    constants_info_[76].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[76].offset = 0;
    constants_info_[76].data_size = 33554432;
    constants_info_[76].from_folded = false;
    constants_info_[76].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[76].shape = {4096, 4096};
    constants_info_[76].stride = {4096, 1};
    constants_info_[76].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[76].original_fqn = "model.layers.8.attention.wo.weight";
    constants_info_[77].name = "model_layers_8_feed_forward_w1_weight";
    constants_info_[77].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[77].offset = 0;
    constants_info_[77].data_size = 117440512;
    constants_info_[77].from_folded = false;
    constants_info_[77].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[77].shape = {14336, 4096};
    constants_info_[77].stride = {4096, 1};
    constants_info_[77].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[77].original_fqn = "model.layers.8.feed_forward.w1.weight";
    constants_info_[78].name = "model_layers_8_feed_forward_w2_weight";
    constants_info_[78].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[78].offset = 0;
    constants_info_[78].data_size = 117440512;
    constants_info_[78].from_folded = false;
    constants_info_[78].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[78].shape = {4096, 14336};
    constants_info_[78].stride = {14336, 1};
    constants_info_[78].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[78].original_fqn = "model.layers.8.feed_forward.w2.weight";
    constants_info_[79].name = "model_layers_8_feed_forward_w3_weight";
    constants_info_[79].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[79].offset = 0;
    constants_info_[79].data_size = 117440512;
    constants_info_[79].from_folded = false;
    constants_info_[79].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[79].shape = {14336, 4096};
    constants_info_[79].stride = {4096, 1};
    constants_info_[79].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[79].original_fqn = "model.layers.8.feed_forward.w3.weight";
    constants_info_[80].name = "model_layers_8_ffn_norm_weight";
    constants_info_[80].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[80].offset = 0;
    constants_info_[80].data_size = 8192;
    constants_info_[80].from_folded = false;
    constants_info_[80].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[80].shape = {4096};
    constants_info_[80].stride = {1};
    constants_info_[80].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[80].original_fqn = "model.layers.8.ffn_norm.weight";
    constants_info_[81].name = "model_layers_8_attention_norm_weight";
    constants_info_[81].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[81].offset = 0;
    constants_info_[81].data_size = 8192;
    constants_info_[81].from_folded = false;
    constants_info_[81].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[81].shape = {4096};
    constants_info_[81].stride = {1};
    constants_info_[81].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[81].original_fqn = "model.layers.8.attention_norm.weight";
    constants_info_[82].name = "model_layers_9_attention_wq_weight";
    constants_info_[82].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[82].offset = 0;
    constants_info_[82].data_size = 33554432;
    constants_info_[82].from_folded = false;
    constants_info_[82].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[82].shape = {4096, 4096};
    constants_info_[82].stride = {4096, 1};
    constants_info_[82].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[82].original_fqn = "model.layers.9.attention.wq.weight";
    constants_info_[83].name = "model_layers_9_attention_wk_weight";
    constants_info_[83].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[83].offset = 0;
    constants_info_[83].data_size = 8388608;
    constants_info_[83].from_folded = false;
    constants_info_[83].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[83].shape = {1024, 4096};
    constants_info_[83].stride = {4096, 1};
    constants_info_[83].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[83].original_fqn = "model.layers.9.attention.wk.weight";
    constants_info_[84].name = "model_layers_9_attention_wv_weight";
    constants_info_[84].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[84].offset = 0;
    constants_info_[84].data_size = 8388608;
    constants_info_[84].from_folded = false;
    constants_info_[84].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[84].shape = {1024, 4096};
    constants_info_[84].stride = {4096, 1};
    constants_info_[84].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[84].original_fqn = "model.layers.9.attention.wv.weight";
    constants_info_[85].name = "model_layers_9_attention_wo_weight";
    constants_info_[85].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[85].offset = 0;
    constants_info_[85].data_size = 33554432;
    constants_info_[85].from_folded = false;
    constants_info_[85].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[85].shape = {4096, 4096};
    constants_info_[85].stride = {4096, 1};
    constants_info_[85].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[85].original_fqn = "model.layers.9.attention.wo.weight";
    constants_info_[86].name = "model_layers_9_feed_forward_w1_weight";
    constants_info_[86].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[86].offset = 0;
    constants_info_[86].data_size = 117440512;
    constants_info_[86].from_folded = false;
    constants_info_[86].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[86].shape = {14336, 4096};
    constants_info_[86].stride = {4096, 1};
    constants_info_[86].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[86].original_fqn = "model.layers.9.feed_forward.w1.weight";
    constants_info_[87].name = "model_layers_9_feed_forward_w2_weight";
    constants_info_[87].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[87].offset = 0;
    constants_info_[87].data_size = 117440512;
    constants_info_[87].from_folded = false;
    constants_info_[87].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[87].shape = {4096, 14336};
    constants_info_[87].stride = {14336, 1};
    constants_info_[87].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[87].original_fqn = "model.layers.9.feed_forward.w2.weight";
    constants_info_[88].name = "model_layers_9_feed_forward_w3_weight";
    constants_info_[88].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[88].offset = 0;
    constants_info_[88].data_size = 117440512;
    constants_info_[88].from_folded = false;
    constants_info_[88].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[88].shape = {14336, 4096};
    constants_info_[88].stride = {4096, 1};
    constants_info_[88].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[88].original_fqn = "model.layers.9.feed_forward.w3.weight";
    constants_info_[89].name = "model_layers_9_ffn_norm_weight";
    constants_info_[89].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[89].offset = 0;
    constants_info_[89].data_size = 8192;
    constants_info_[89].from_folded = false;
    constants_info_[89].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[89].shape = {4096};
    constants_info_[89].stride = {1};
    constants_info_[89].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[89].original_fqn = "model.layers.9.ffn_norm.weight";
    constants_info_[90].name = "model_layers_9_attention_norm_weight";
    constants_info_[90].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[90].offset = 0;
    constants_info_[90].data_size = 8192;
    constants_info_[90].from_folded = false;
    constants_info_[90].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[90].shape = {4096};
    constants_info_[90].stride = {1};
    constants_info_[90].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[90].original_fqn = "model.layers.9.attention_norm.weight";
    constants_info_[91].name = "model_layers_10_attention_wq_weight";
    constants_info_[91].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[91].offset = 0;
    constants_info_[91].data_size = 33554432;
    constants_info_[91].from_folded = false;
    constants_info_[91].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[91].shape = {4096, 4096};
    constants_info_[91].stride = {4096, 1};
    constants_info_[91].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[91].original_fqn = "model.layers.10.attention.wq.weight";
    constants_info_[92].name = "model_layers_10_attention_wk_weight";
    constants_info_[92].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[92].offset = 0;
    constants_info_[92].data_size = 8388608;
    constants_info_[92].from_folded = false;
    constants_info_[92].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[92].shape = {1024, 4096};
    constants_info_[92].stride = {4096, 1};
    constants_info_[92].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[92].original_fqn = "model.layers.10.attention.wk.weight";
    constants_info_[93].name = "model_layers_10_attention_wv_weight";
    constants_info_[93].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[93].offset = 0;
    constants_info_[93].data_size = 8388608;
    constants_info_[93].from_folded = false;
    constants_info_[93].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[93].shape = {1024, 4096};
    constants_info_[93].stride = {4096, 1};
    constants_info_[93].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[93].original_fqn = "model.layers.10.attention.wv.weight";
    constants_info_[94].name = "model_layers_10_attention_wo_weight";
    constants_info_[94].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[94].offset = 0;
    constants_info_[94].data_size = 33554432;
    constants_info_[94].from_folded = false;
    constants_info_[94].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[94].shape = {4096, 4096};
    constants_info_[94].stride = {4096, 1};
    constants_info_[94].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[94].original_fqn = "model.layers.10.attention.wo.weight";
    constants_info_[95].name = "model_layers_10_feed_forward_w1_weight";
    constants_info_[95].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[95].offset = 0;
    constants_info_[95].data_size = 117440512;
    constants_info_[95].from_folded = false;
    constants_info_[95].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[95].shape = {14336, 4096};
    constants_info_[95].stride = {4096, 1};
    constants_info_[95].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[95].original_fqn = "model.layers.10.feed_forward.w1.weight";
    constants_info_[96].name = "model_layers_10_feed_forward_w2_weight";
    constants_info_[96].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[96].offset = 0;
    constants_info_[96].data_size = 117440512;
    constants_info_[96].from_folded = false;
    constants_info_[96].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[96].shape = {4096, 14336};
    constants_info_[96].stride = {14336, 1};
    constants_info_[96].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[96].original_fqn = "model.layers.10.feed_forward.w2.weight";
    constants_info_[97].name = "model_layers_10_feed_forward_w3_weight";
    constants_info_[97].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[97].offset = 0;
    constants_info_[97].data_size = 117440512;
    constants_info_[97].from_folded = false;
    constants_info_[97].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[97].shape = {14336, 4096};
    constants_info_[97].stride = {4096, 1};
    constants_info_[97].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[97].original_fqn = "model.layers.10.feed_forward.w3.weight";
    constants_info_[98].name = "model_layers_10_ffn_norm_weight";
    constants_info_[98].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[98].offset = 0;
    constants_info_[98].data_size = 8192;
    constants_info_[98].from_folded = false;
    constants_info_[98].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[98].shape = {4096};
    constants_info_[98].stride = {1};
    constants_info_[98].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[98].original_fqn = "model.layers.10.ffn_norm.weight";
    constants_info_[99].name = "model_layers_10_attention_norm_weight";
    constants_info_[99].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[99].offset = 0;
    constants_info_[99].data_size = 8192;
    constants_info_[99].from_folded = false;
    constants_info_[99].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[99].shape = {4096};
    constants_info_[99].stride = {1};
    constants_info_[99].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[99].original_fqn = "model.layers.10.attention_norm.weight";
    constants_info_[100].name = "model_layers_11_attention_wq_weight";
    constants_info_[100].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[100].offset = 0;
    constants_info_[100].data_size = 33554432;
    constants_info_[100].from_folded = false;
    constants_info_[100].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[100].shape = {4096, 4096};
    constants_info_[100].stride = {4096, 1};
    constants_info_[100].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[100].original_fqn = "model.layers.11.attention.wq.weight";
    constants_info_[101].name = "model_layers_11_attention_wk_weight";
    constants_info_[101].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[101].offset = 0;
    constants_info_[101].data_size = 8388608;
    constants_info_[101].from_folded = false;
    constants_info_[101].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[101].shape = {1024, 4096};
    constants_info_[101].stride = {4096, 1};
    constants_info_[101].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[101].original_fqn = "model.layers.11.attention.wk.weight";
    constants_info_[102].name = "model_layers_11_attention_wv_weight";
    constants_info_[102].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[102].offset = 0;
    constants_info_[102].data_size = 8388608;
    constants_info_[102].from_folded = false;
    constants_info_[102].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[102].shape = {1024, 4096};
    constants_info_[102].stride = {4096, 1};
    constants_info_[102].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[102].original_fqn = "model.layers.11.attention.wv.weight";
    constants_info_[103].name = "model_layers_11_attention_wo_weight";
    constants_info_[103].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[103].offset = 0;
    constants_info_[103].data_size = 33554432;
    constants_info_[103].from_folded = false;
    constants_info_[103].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[103].shape = {4096, 4096};
    constants_info_[103].stride = {4096, 1};
    constants_info_[103].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[103].original_fqn = "model.layers.11.attention.wo.weight";
    constants_info_[104].name = "model_layers_11_feed_forward_w1_weight";
    constants_info_[104].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[104].offset = 0;
    constants_info_[104].data_size = 117440512;
    constants_info_[104].from_folded = false;
    constants_info_[104].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[104].shape = {14336, 4096};
    constants_info_[104].stride = {4096, 1};
    constants_info_[104].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[104].original_fqn = "model.layers.11.feed_forward.w1.weight";
    constants_info_[105].name = "model_layers_11_feed_forward_w2_weight";
    constants_info_[105].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[105].offset = 0;
    constants_info_[105].data_size = 117440512;
    constants_info_[105].from_folded = false;
    constants_info_[105].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[105].shape = {4096, 14336};
    constants_info_[105].stride = {14336, 1};
    constants_info_[105].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[105].original_fqn = "model.layers.11.feed_forward.w2.weight";
    constants_info_[106].name = "model_layers_11_feed_forward_w3_weight";
    constants_info_[106].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[106].offset = 0;
    constants_info_[106].data_size = 117440512;
    constants_info_[106].from_folded = false;
    constants_info_[106].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[106].shape = {14336, 4096};
    constants_info_[106].stride = {4096, 1};
    constants_info_[106].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[106].original_fqn = "model.layers.11.feed_forward.w3.weight";
    constants_info_[107].name = "model_layers_11_ffn_norm_weight";
    constants_info_[107].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[107].offset = 0;
    constants_info_[107].data_size = 8192;
    constants_info_[107].from_folded = false;
    constants_info_[107].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[107].shape = {4096};
    constants_info_[107].stride = {1};
    constants_info_[107].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[107].original_fqn = "model.layers.11.ffn_norm.weight";
    constants_info_[108].name = "model_layers_11_attention_norm_weight";
    constants_info_[108].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[108].offset = 0;
    constants_info_[108].data_size = 8192;
    constants_info_[108].from_folded = false;
    constants_info_[108].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[108].shape = {4096};
    constants_info_[108].stride = {1};
    constants_info_[108].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[108].original_fqn = "model.layers.11.attention_norm.weight";
    constants_info_[109].name = "model_layers_12_attention_wq_weight";
    constants_info_[109].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[109].offset = 0;
    constants_info_[109].data_size = 33554432;
    constants_info_[109].from_folded = false;
    constants_info_[109].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[109].shape = {4096, 4096};
    constants_info_[109].stride = {4096, 1};
    constants_info_[109].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[109].original_fqn = "model.layers.12.attention.wq.weight";
    constants_info_[110].name = "model_layers_12_attention_wk_weight";
    constants_info_[110].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[110].offset = 0;
    constants_info_[110].data_size = 8388608;
    constants_info_[110].from_folded = false;
    constants_info_[110].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[110].shape = {1024, 4096};
    constants_info_[110].stride = {4096, 1};
    constants_info_[110].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[110].original_fqn = "model.layers.12.attention.wk.weight";
    constants_info_[111].name = "model_layers_12_attention_wv_weight";
    constants_info_[111].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[111].offset = 0;
    constants_info_[111].data_size = 8388608;
    constants_info_[111].from_folded = false;
    constants_info_[111].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[111].shape = {1024, 4096};
    constants_info_[111].stride = {4096, 1};
    constants_info_[111].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[111].original_fqn = "model.layers.12.attention.wv.weight";
    constants_info_[112].name = "model_layers_12_attention_wo_weight";
    constants_info_[112].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[112].offset = 0;
    constants_info_[112].data_size = 33554432;
    constants_info_[112].from_folded = false;
    constants_info_[112].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[112].shape = {4096, 4096};
    constants_info_[112].stride = {4096, 1};
    constants_info_[112].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[112].original_fqn = "model.layers.12.attention.wo.weight";
    constants_info_[113].name = "model_layers_12_feed_forward_w1_weight";
    constants_info_[113].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[113].offset = 0;
    constants_info_[113].data_size = 117440512;
    constants_info_[113].from_folded = false;
    constants_info_[113].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[113].shape = {14336, 4096};
    constants_info_[113].stride = {4096, 1};
    constants_info_[113].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[113].original_fqn = "model.layers.12.feed_forward.w1.weight";
    constants_info_[114].name = "model_layers_12_feed_forward_w2_weight";
    constants_info_[114].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[114].offset = 0;
    constants_info_[114].data_size = 117440512;
    constants_info_[114].from_folded = false;
    constants_info_[114].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[114].shape = {4096, 14336};
    constants_info_[114].stride = {14336, 1};
    constants_info_[114].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[114].original_fqn = "model.layers.12.feed_forward.w2.weight";
    constants_info_[115].name = "model_layers_12_feed_forward_w3_weight";
    constants_info_[115].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[115].offset = 0;
    constants_info_[115].data_size = 117440512;
    constants_info_[115].from_folded = false;
    constants_info_[115].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[115].shape = {14336, 4096};
    constants_info_[115].stride = {4096, 1};
    constants_info_[115].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[115].original_fqn = "model.layers.12.feed_forward.w3.weight";
    constants_info_[116].name = "model_layers_12_ffn_norm_weight";
    constants_info_[116].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[116].offset = 0;
    constants_info_[116].data_size = 8192;
    constants_info_[116].from_folded = false;
    constants_info_[116].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[116].shape = {4096};
    constants_info_[116].stride = {1};
    constants_info_[116].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[116].original_fqn = "model.layers.12.ffn_norm.weight";
    constants_info_[117].name = "model_layers_12_attention_norm_weight";
    constants_info_[117].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[117].offset = 0;
    constants_info_[117].data_size = 8192;
    constants_info_[117].from_folded = false;
    constants_info_[117].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[117].shape = {4096};
    constants_info_[117].stride = {1};
    constants_info_[117].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[117].original_fqn = "model.layers.12.attention_norm.weight";
    constants_info_[118].name = "model_layers_13_attention_wq_weight";
    constants_info_[118].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[118].offset = 0;
    constants_info_[118].data_size = 33554432;
    constants_info_[118].from_folded = false;
    constants_info_[118].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[118].shape = {4096, 4096};
    constants_info_[118].stride = {4096, 1};
    constants_info_[118].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[118].original_fqn = "model.layers.13.attention.wq.weight";
    constants_info_[119].name = "model_layers_13_attention_wk_weight";
    constants_info_[119].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[119].offset = 0;
    constants_info_[119].data_size = 8388608;
    constants_info_[119].from_folded = false;
    constants_info_[119].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[119].shape = {1024, 4096};
    constants_info_[119].stride = {4096, 1};
    constants_info_[119].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[119].original_fqn = "model.layers.13.attention.wk.weight";
    constants_info_[120].name = "model_layers_13_attention_wv_weight";
    constants_info_[120].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[120].offset = 0;
    constants_info_[120].data_size = 8388608;
    constants_info_[120].from_folded = false;
    constants_info_[120].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[120].shape = {1024, 4096};
    constants_info_[120].stride = {4096, 1};
    constants_info_[120].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[120].original_fqn = "model.layers.13.attention.wv.weight";
    constants_info_[121].name = "model_layers_13_attention_wo_weight";
    constants_info_[121].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[121].offset = 0;
    constants_info_[121].data_size = 33554432;
    constants_info_[121].from_folded = false;
    constants_info_[121].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[121].shape = {4096, 4096};
    constants_info_[121].stride = {4096, 1};
    constants_info_[121].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[121].original_fqn = "model.layers.13.attention.wo.weight";
    constants_info_[122].name = "model_layers_13_feed_forward_w1_weight";
    constants_info_[122].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[122].offset = 0;
    constants_info_[122].data_size = 117440512;
    constants_info_[122].from_folded = false;
    constants_info_[122].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[122].shape = {14336, 4096};
    constants_info_[122].stride = {4096, 1};
    constants_info_[122].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[122].original_fqn = "model.layers.13.feed_forward.w1.weight";
    constants_info_[123].name = "model_layers_13_feed_forward_w2_weight";
    constants_info_[123].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[123].offset = 0;
    constants_info_[123].data_size = 117440512;
    constants_info_[123].from_folded = false;
    constants_info_[123].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[123].shape = {4096, 14336};
    constants_info_[123].stride = {14336, 1};
    constants_info_[123].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[123].original_fqn = "model.layers.13.feed_forward.w2.weight";
    constants_info_[124].name = "model_layers_13_feed_forward_w3_weight";
    constants_info_[124].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[124].offset = 0;
    constants_info_[124].data_size = 117440512;
    constants_info_[124].from_folded = false;
    constants_info_[124].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[124].shape = {14336, 4096};
    constants_info_[124].stride = {4096, 1};
    constants_info_[124].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[124].original_fqn = "model.layers.13.feed_forward.w3.weight";
    constants_info_[125].name = "model_layers_13_ffn_norm_weight";
    constants_info_[125].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[125].offset = 0;
    constants_info_[125].data_size = 8192;
    constants_info_[125].from_folded = false;
    constants_info_[125].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[125].shape = {4096};
    constants_info_[125].stride = {1};
    constants_info_[125].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[125].original_fqn = "model.layers.13.ffn_norm.weight";
    constants_info_[126].name = "model_layers_13_attention_norm_weight";
    constants_info_[126].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[126].offset = 0;
    constants_info_[126].data_size = 8192;
    constants_info_[126].from_folded = false;
    constants_info_[126].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[126].shape = {4096};
    constants_info_[126].stride = {1};
    constants_info_[126].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[126].original_fqn = "model.layers.13.attention_norm.weight";
    constants_info_[127].name = "model_layers_14_attention_wq_weight";
    constants_info_[127].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[127].offset = 0;
    constants_info_[127].data_size = 33554432;
    constants_info_[127].from_folded = false;
    constants_info_[127].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[127].shape = {4096, 4096};
    constants_info_[127].stride = {4096, 1};
    constants_info_[127].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[127].original_fqn = "model.layers.14.attention.wq.weight";
    constants_info_[128].name = "model_layers_14_attention_wk_weight";
    constants_info_[128].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[128].offset = 0;
    constants_info_[128].data_size = 8388608;
    constants_info_[128].from_folded = false;
    constants_info_[128].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[128].shape = {1024, 4096};
    constants_info_[128].stride = {4096, 1};
    constants_info_[128].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[128].original_fqn = "model.layers.14.attention.wk.weight";
    constants_info_[129].name = "model_layers_14_attention_wv_weight";
    constants_info_[129].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[129].offset = 0;
    constants_info_[129].data_size = 8388608;
    constants_info_[129].from_folded = false;
    constants_info_[129].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[129].shape = {1024, 4096};
    constants_info_[129].stride = {4096, 1};
    constants_info_[129].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[129].original_fqn = "model.layers.14.attention.wv.weight";
    constants_info_[130].name = "model_layers_14_attention_wo_weight";
    constants_info_[130].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[130].offset = 0;
    constants_info_[130].data_size = 33554432;
    constants_info_[130].from_folded = false;
    constants_info_[130].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[130].shape = {4096, 4096};
    constants_info_[130].stride = {4096, 1};
    constants_info_[130].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[130].original_fqn = "model.layers.14.attention.wo.weight";
    constants_info_[131].name = "model_layers_14_feed_forward_w1_weight";
    constants_info_[131].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[131].offset = 0;
    constants_info_[131].data_size = 117440512;
    constants_info_[131].from_folded = false;
    constants_info_[131].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[131].shape = {14336, 4096};
    constants_info_[131].stride = {4096, 1};
    constants_info_[131].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[131].original_fqn = "model.layers.14.feed_forward.w1.weight";
    constants_info_[132].name = "model_layers_14_feed_forward_w2_weight";
    constants_info_[132].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[132].offset = 0;
    constants_info_[132].data_size = 117440512;
    constants_info_[132].from_folded = false;
    constants_info_[132].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[132].shape = {4096, 14336};
    constants_info_[132].stride = {14336, 1};
    constants_info_[132].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[132].original_fqn = "model.layers.14.feed_forward.w2.weight";
    constants_info_[133].name = "model_layers_14_feed_forward_w3_weight";
    constants_info_[133].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[133].offset = 0;
    constants_info_[133].data_size = 117440512;
    constants_info_[133].from_folded = false;
    constants_info_[133].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[133].shape = {14336, 4096};
    constants_info_[133].stride = {4096, 1};
    constants_info_[133].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[133].original_fqn = "model.layers.14.feed_forward.w3.weight";
    constants_info_[134].name = "model_layers_14_ffn_norm_weight";
    constants_info_[134].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[134].offset = 0;
    constants_info_[134].data_size = 8192;
    constants_info_[134].from_folded = false;
    constants_info_[134].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[134].shape = {4096};
    constants_info_[134].stride = {1};
    constants_info_[134].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[134].original_fqn = "model.layers.14.ffn_norm.weight";
    constants_info_[135].name = "model_layers_14_attention_norm_weight";
    constants_info_[135].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[135].offset = 0;
    constants_info_[135].data_size = 8192;
    constants_info_[135].from_folded = false;
    constants_info_[135].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[135].shape = {4096};
    constants_info_[135].stride = {1};
    constants_info_[135].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[135].original_fqn = "model.layers.14.attention_norm.weight";
    constants_info_[136].name = "model_layers_15_attention_wq_weight";
    constants_info_[136].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[136].offset = 0;
    constants_info_[136].data_size = 33554432;
    constants_info_[136].from_folded = false;
    constants_info_[136].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[136].shape = {4096, 4096};
    constants_info_[136].stride = {4096, 1};
    constants_info_[136].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[136].original_fqn = "model.layers.15.attention.wq.weight";
    constants_info_[137].name = "model_layers_15_attention_wk_weight";
    constants_info_[137].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[137].offset = 0;
    constants_info_[137].data_size = 8388608;
    constants_info_[137].from_folded = false;
    constants_info_[137].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[137].shape = {1024, 4096};
    constants_info_[137].stride = {4096, 1};
    constants_info_[137].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[137].original_fqn = "model.layers.15.attention.wk.weight";
    constants_info_[138].name = "model_layers_15_attention_wv_weight";
    constants_info_[138].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[138].offset = 0;
    constants_info_[138].data_size = 8388608;
    constants_info_[138].from_folded = false;
    constants_info_[138].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[138].shape = {1024, 4096};
    constants_info_[138].stride = {4096, 1};
    constants_info_[138].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[138].original_fqn = "model.layers.15.attention.wv.weight";
    constants_info_[139].name = "model_layers_15_attention_wo_weight";
    constants_info_[139].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[139].offset = 0;
    constants_info_[139].data_size = 33554432;
    constants_info_[139].from_folded = false;
    constants_info_[139].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[139].shape = {4096, 4096};
    constants_info_[139].stride = {4096, 1};
    constants_info_[139].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[139].original_fqn = "model.layers.15.attention.wo.weight";
    constants_info_[140].name = "model_layers_15_feed_forward_w1_weight";
    constants_info_[140].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[140].offset = 0;
    constants_info_[140].data_size = 117440512;
    constants_info_[140].from_folded = false;
    constants_info_[140].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[140].shape = {14336, 4096};
    constants_info_[140].stride = {4096, 1};
    constants_info_[140].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[140].original_fqn = "model.layers.15.feed_forward.w1.weight";
    constants_info_[141].name = "model_layers_15_feed_forward_w2_weight";
    constants_info_[141].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[141].offset = 0;
    constants_info_[141].data_size = 117440512;
    constants_info_[141].from_folded = false;
    constants_info_[141].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[141].shape = {4096, 14336};
    constants_info_[141].stride = {14336, 1};
    constants_info_[141].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[141].original_fqn = "model.layers.15.feed_forward.w2.weight";
    constants_info_[142].name = "model_layers_15_feed_forward_w3_weight";
    constants_info_[142].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[142].offset = 0;
    constants_info_[142].data_size = 117440512;
    constants_info_[142].from_folded = false;
    constants_info_[142].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[142].shape = {14336, 4096};
    constants_info_[142].stride = {4096, 1};
    constants_info_[142].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[142].original_fqn = "model.layers.15.feed_forward.w3.weight";
    constants_info_[143].name = "model_layers_15_ffn_norm_weight";
    constants_info_[143].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[143].offset = 0;
    constants_info_[143].data_size = 8192;
    constants_info_[143].from_folded = false;
    constants_info_[143].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[143].shape = {4096};
    constants_info_[143].stride = {1};
    constants_info_[143].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[143].original_fqn = "model.layers.15.ffn_norm.weight";
    constants_info_[144].name = "model_layers_15_attention_norm_weight";
    constants_info_[144].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[144].offset = 0;
    constants_info_[144].data_size = 8192;
    constants_info_[144].from_folded = false;
    constants_info_[144].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[144].shape = {4096};
    constants_info_[144].stride = {1};
    constants_info_[144].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[144].original_fqn = "model.layers.15.attention_norm.weight";
    constants_info_[145].name = "model_layers_16_attention_wq_weight";
    constants_info_[145].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[145].offset = 0;
    constants_info_[145].data_size = 33554432;
    constants_info_[145].from_folded = false;
    constants_info_[145].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[145].shape = {4096, 4096};
    constants_info_[145].stride = {4096, 1};
    constants_info_[145].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[145].original_fqn = "model.layers.16.attention.wq.weight";
    constants_info_[146].name = "model_layers_16_attention_wk_weight";
    constants_info_[146].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[146].offset = 0;
    constants_info_[146].data_size = 8388608;
    constants_info_[146].from_folded = false;
    constants_info_[146].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[146].shape = {1024, 4096};
    constants_info_[146].stride = {4096, 1};
    constants_info_[146].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[146].original_fqn = "model.layers.16.attention.wk.weight";
    constants_info_[147].name = "model_layers_16_attention_wv_weight";
    constants_info_[147].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[147].offset = 0;
    constants_info_[147].data_size = 8388608;
    constants_info_[147].from_folded = false;
    constants_info_[147].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[147].shape = {1024, 4096};
    constants_info_[147].stride = {4096, 1};
    constants_info_[147].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[147].original_fqn = "model.layers.16.attention.wv.weight";
    constants_info_[148].name = "model_layers_16_attention_wo_weight";
    constants_info_[148].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[148].offset = 0;
    constants_info_[148].data_size = 33554432;
    constants_info_[148].from_folded = false;
    constants_info_[148].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[148].shape = {4096, 4096};
    constants_info_[148].stride = {4096, 1};
    constants_info_[148].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[148].original_fqn = "model.layers.16.attention.wo.weight";
    constants_info_[149].name = "model_layers_16_feed_forward_w1_weight";
    constants_info_[149].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[149].offset = 0;
    constants_info_[149].data_size = 117440512;
    constants_info_[149].from_folded = false;
    constants_info_[149].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[149].shape = {14336, 4096};
    constants_info_[149].stride = {4096, 1};
    constants_info_[149].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[149].original_fqn = "model.layers.16.feed_forward.w1.weight";
    constants_info_[150].name = "model_layers_16_feed_forward_w2_weight";
    constants_info_[150].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[150].offset = 0;
    constants_info_[150].data_size = 117440512;
    constants_info_[150].from_folded = false;
    constants_info_[150].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[150].shape = {4096, 14336};
    constants_info_[150].stride = {14336, 1};
    constants_info_[150].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[150].original_fqn = "model.layers.16.feed_forward.w2.weight";
    constants_info_[151].name = "model_layers_16_feed_forward_w3_weight";
    constants_info_[151].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[151].offset = 0;
    constants_info_[151].data_size = 117440512;
    constants_info_[151].from_folded = false;
    constants_info_[151].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[151].shape = {14336, 4096};
    constants_info_[151].stride = {4096, 1};
    constants_info_[151].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[151].original_fqn = "model.layers.16.feed_forward.w3.weight";
    constants_info_[152].name = "model_layers_16_ffn_norm_weight";
    constants_info_[152].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[152].offset = 0;
    constants_info_[152].data_size = 8192;
    constants_info_[152].from_folded = false;
    constants_info_[152].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[152].shape = {4096};
    constants_info_[152].stride = {1};
    constants_info_[152].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[152].original_fqn = "model.layers.16.ffn_norm.weight";
    constants_info_[153].name = "model_layers_16_attention_norm_weight";
    constants_info_[153].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[153].offset = 0;
    constants_info_[153].data_size = 8192;
    constants_info_[153].from_folded = false;
    constants_info_[153].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[153].shape = {4096};
    constants_info_[153].stride = {1};
    constants_info_[153].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[153].original_fqn = "model.layers.16.attention_norm.weight";
    constants_info_[154].name = "model_layers_17_attention_wq_weight";
    constants_info_[154].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[154].offset = 0;
    constants_info_[154].data_size = 33554432;
    constants_info_[154].from_folded = false;
    constants_info_[154].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[154].shape = {4096, 4096};
    constants_info_[154].stride = {4096, 1};
    constants_info_[154].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[154].original_fqn = "model.layers.17.attention.wq.weight";
    constants_info_[155].name = "model_layers_17_attention_wk_weight";
    constants_info_[155].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[155].offset = 0;
    constants_info_[155].data_size = 8388608;
    constants_info_[155].from_folded = false;
    constants_info_[155].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[155].shape = {1024, 4096};
    constants_info_[155].stride = {4096, 1};
    constants_info_[155].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[155].original_fqn = "model.layers.17.attention.wk.weight";
    constants_info_[156].name = "model_layers_17_attention_wv_weight";
    constants_info_[156].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[156].offset = 0;
    constants_info_[156].data_size = 8388608;
    constants_info_[156].from_folded = false;
    constants_info_[156].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[156].shape = {1024, 4096};
    constants_info_[156].stride = {4096, 1};
    constants_info_[156].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[156].original_fqn = "model.layers.17.attention.wv.weight";
    constants_info_[157].name = "model_layers_17_attention_wo_weight";
    constants_info_[157].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[157].offset = 0;
    constants_info_[157].data_size = 33554432;
    constants_info_[157].from_folded = false;
    constants_info_[157].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[157].shape = {4096, 4096};
    constants_info_[157].stride = {4096, 1};
    constants_info_[157].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[157].original_fqn = "model.layers.17.attention.wo.weight";
    constants_info_[158].name = "model_layers_17_feed_forward_w1_weight";
    constants_info_[158].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[158].offset = 0;
    constants_info_[158].data_size = 117440512;
    constants_info_[158].from_folded = false;
    constants_info_[158].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[158].shape = {14336, 4096};
    constants_info_[158].stride = {4096, 1};
    constants_info_[158].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[158].original_fqn = "model.layers.17.feed_forward.w1.weight";
    constants_info_[159].name = "model_layers_17_feed_forward_w2_weight";
    constants_info_[159].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[159].offset = 0;
    constants_info_[159].data_size = 117440512;
    constants_info_[159].from_folded = false;
    constants_info_[159].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[159].shape = {4096, 14336};
    constants_info_[159].stride = {14336, 1};
    constants_info_[159].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[159].original_fqn = "model.layers.17.feed_forward.w2.weight";
    constants_info_[160].name = "model_layers_17_feed_forward_w3_weight";
    constants_info_[160].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[160].offset = 0;
    constants_info_[160].data_size = 117440512;
    constants_info_[160].from_folded = false;
    constants_info_[160].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[160].shape = {14336, 4096};
    constants_info_[160].stride = {4096, 1};
    constants_info_[160].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[160].original_fqn = "model.layers.17.feed_forward.w3.weight";
    constants_info_[161].name = "model_layers_17_ffn_norm_weight";
    constants_info_[161].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[161].offset = 0;
    constants_info_[161].data_size = 8192;
    constants_info_[161].from_folded = false;
    constants_info_[161].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[161].shape = {4096};
    constants_info_[161].stride = {1};
    constants_info_[161].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[161].original_fqn = "model.layers.17.ffn_norm.weight";
    constants_info_[162].name = "model_layers_17_attention_norm_weight";
    constants_info_[162].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[162].offset = 0;
    constants_info_[162].data_size = 8192;
    constants_info_[162].from_folded = false;
    constants_info_[162].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[162].shape = {4096};
    constants_info_[162].stride = {1};
    constants_info_[162].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[162].original_fqn = "model.layers.17.attention_norm.weight";
    constants_info_[163].name = "model_layers_18_attention_wq_weight";
    constants_info_[163].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[163].offset = 0;
    constants_info_[163].data_size = 33554432;
    constants_info_[163].from_folded = false;
    constants_info_[163].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[163].shape = {4096, 4096};
    constants_info_[163].stride = {4096, 1};
    constants_info_[163].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[163].original_fqn = "model.layers.18.attention.wq.weight";
    constants_info_[164].name = "model_layers_18_attention_wk_weight";
    constants_info_[164].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[164].offset = 0;
    constants_info_[164].data_size = 8388608;
    constants_info_[164].from_folded = false;
    constants_info_[164].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[164].shape = {1024, 4096};
    constants_info_[164].stride = {4096, 1};
    constants_info_[164].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[164].original_fqn = "model.layers.18.attention.wk.weight";
    constants_info_[165].name = "model_layers_18_attention_wv_weight";
    constants_info_[165].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[165].offset = 0;
    constants_info_[165].data_size = 8388608;
    constants_info_[165].from_folded = false;
    constants_info_[165].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[165].shape = {1024, 4096};
    constants_info_[165].stride = {4096, 1};
    constants_info_[165].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[165].original_fqn = "model.layers.18.attention.wv.weight";
    constants_info_[166].name = "model_layers_18_attention_wo_weight";
    constants_info_[166].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[166].offset = 0;
    constants_info_[166].data_size = 33554432;
    constants_info_[166].from_folded = false;
    constants_info_[166].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[166].shape = {4096, 4096};
    constants_info_[166].stride = {4096, 1};
    constants_info_[166].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[166].original_fqn = "model.layers.18.attention.wo.weight";
    constants_info_[167].name = "model_layers_18_feed_forward_w1_weight";
    constants_info_[167].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[167].offset = 0;
    constants_info_[167].data_size = 117440512;
    constants_info_[167].from_folded = false;
    constants_info_[167].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[167].shape = {14336, 4096};
    constants_info_[167].stride = {4096, 1};
    constants_info_[167].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[167].original_fqn = "model.layers.18.feed_forward.w1.weight";
    constants_info_[168].name = "model_layers_18_feed_forward_w2_weight";
    constants_info_[168].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[168].offset = 0;
    constants_info_[168].data_size = 117440512;
    constants_info_[168].from_folded = false;
    constants_info_[168].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[168].shape = {4096, 14336};
    constants_info_[168].stride = {14336, 1};
    constants_info_[168].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[168].original_fqn = "model.layers.18.feed_forward.w2.weight";
    constants_info_[169].name = "model_layers_18_feed_forward_w3_weight";
    constants_info_[169].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[169].offset = 0;
    constants_info_[169].data_size = 117440512;
    constants_info_[169].from_folded = false;
    constants_info_[169].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[169].shape = {14336, 4096};
    constants_info_[169].stride = {4096, 1};
    constants_info_[169].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[169].original_fqn = "model.layers.18.feed_forward.w3.weight";
    constants_info_[170].name = "model_layers_18_ffn_norm_weight";
    constants_info_[170].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[170].offset = 0;
    constants_info_[170].data_size = 8192;
    constants_info_[170].from_folded = false;
    constants_info_[170].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[170].shape = {4096};
    constants_info_[170].stride = {1};
    constants_info_[170].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[170].original_fqn = "model.layers.18.ffn_norm.weight";
    constants_info_[171].name = "model_layers_18_attention_norm_weight";
    constants_info_[171].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[171].offset = 0;
    constants_info_[171].data_size = 8192;
    constants_info_[171].from_folded = false;
    constants_info_[171].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[171].shape = {4096};
    constants_info_[171].stride = {1};
    constants_info_[171].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[171].original_fqn = "model.layers.18.attention_norm.weight";
    constants_info_[172].name = "model_layers_19_attention_wq_weight";
    constants_info_[172].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[172].offset = 0;
    constants_info_[172].data_size = 33554432;
    constants_info_[172].from_folded = false;
    constants_info_[172].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[172].shape = {4096, 4096};
    constants_info_[172].stride = {4096, 1};
    constants_info_[172].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[172].original_fqn = "model.layers.19.attention.wq.weight";
    constants_info_[173].name = "model_layers_19_attention_wk_weight";
    constants_info_[173].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[173].offset = 0;
    constants_info_[173].data_size = 8388608;
    constants_info_[173].from_folded = false;
    constants_info_[173].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[173].shape = {1024, 4096};
    constants_info_[173].stride = {4096, 1};
    constants_info_[173].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[173].original_fqn = "model.layers.19.attention.wk.weight";
    constants_info_[174].name = "model_layers_19_attention_wv_weight";
    constants_info_[174].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[174].offset = 0;
    constants_info_[174].data_size = 8388608;
    constants_info_[174].from_folded = false;
    constants_info_[174].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[174].shape = {1024, 4096};
    constants_info_[174].stride = {4096, 1};
    constants_info_[174].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[174].original_fqn = "model.layers.19.attention.wv.weight";
    constants_info_[175].name = "model_layers_19_attention_wo_weight";
    constants_info_[175].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[175].offset = 0;
    constants_info_[175].data_size = 33554432;
    constants_info_[175].from_folded = false;
    constants_info_[175].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[175].shape = {4096, 4096};
    constants_info_[175].stride = {4096, 1};
    constants_info_[175].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[175].original_fqn = "model.layers.19.attention.wo.weight";
    constants_info_[176].name = "model_layers_19_feed_forward_w1_weight";
    constants_info_[176].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[176].offset = 0;
    constants_info_[176].data_size = 117440512;
    constants_info_[176].from_folded = false;
    constants_info_[176].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[176].shape = {14336, 4096};
    constants_info_[176].stride = {4096, 1};
    constants_info_[176].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[176].original_fqn = "model.layers.19.feed_forward.w1.weight";
    constants_info_[177].name = "model_layers_19_feed_forward_w2_weight";
    constants_info_[177].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[177].offset = 0;
    constants_info_[177].data_size = 117440512;
    constants_info_[177].from_folded = false;
    constants_info_[177].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[177].shape = {4096, 14336};
    constants_info_[177].stride = {14336, 1};
    constants_info_[177].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[177].original_fqn = "model.layers.19.feed_forward.w2.weight";
    constants_info_[178].name = "model_layers_19_feed_forward_w3_weight";
    constants_info_[178].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[178].offset = 0;
    constants_info_[178].data_size = 117440512;
    constants_info_[178].from_folded = false;
    constants_info_[178].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[178].shape = {14336, 4096};
    constants_info_[178].stride = {4096, 1};
    constants_info_[178].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[178].original_fqn = "model.layers.19.feed_forward.w3.weight";
    constants_info_[179].name = "model_layers_19_ffn_norm_weight";
    constants_info_[179].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[179].offset = 0;
    constants_info_[179].data_size = 8192;
    constants_info_[179].from_folded = false;
    constants_info_[179].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[179].shape = {4096};
    constants_info_[179].stride = {1};
    constants_info_[179].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[179].original_fqn = "model.layers.19.ffn_norm.weight";
    constants_info_[180].name = "model_layers_19_attention_norm_weight";
    constants_info_[180].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[180].offset = 0;
    constants_info_[180].data_size = 8192;
    constants_info_[180].from_folded = false;
    constants_info_[180].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[180].shape = {4096};
    constants_info_[180].stride = {1};
    constants_info_[180].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[180].original_fqn = "model.layers.19.attention_norm.weight";
    constants_info_[181].name = "model_layers_20_attention_wq_weight";
    constants_info_[181].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[181].offset = 0;
    constants_info_[181].data_size = 33554432;
    constants_info_[181].from_folded = false;
    constants_info_[181].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[181].shape = {4096, 4096};
    constants_info_[181].stride = {4096, 1};
    constants_info_[181].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[181].original_fqn = "model.layers.20.attention.wq.weight";
    constants_info_[182].name = "model_layers_20_attention_wk_weight";
    constants_info_[182].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[182].offset = 0;
    constants_info_[182].data_size = 8388608;
    constants_info_[182].from_folded = false;
    constants_info_[182].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[182].shape = {1024, 4096};
    constants_info_[182].stride = {4096, 1};
    constants_info_[182].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[182].original_fqn = "model.layers.20.attention.wk.weight";
    constants_info_[183].name = "model_layers_20_attention_wv_weight";
    constants_info_[183].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[183].offset = 0;
    constants_info_[183].data_size = 8388608;
    constants_info_[183].from_folded = false;
    constants_info_[183].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[183].shape = {1024, 4096};
    constants_info_[183].stride = {4096, 1};
    constants_info_[183].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[183].original_fqn = "model.layers.20.attention.wv.weight";
    constants_info_[184].name = "model_layers_20_attention_wo_weight";
    constants_info_[184].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[184].offset = 0;
    constants_info_[184].data_size = 33554432;
    constants_info_[184].from_folded = false;
    constants_info_[184].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[184].shape = {4096, 4096};
    constants_info_[184].stride = {4096, 1};
    constants_info_[184].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[184].original_fqn = "model.layers.20.attention.wo.weight";
    constants_info_[185].name = "model_layers_20_feed_forward_w1_weight";
    constants_info_[185].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[185].offset = 0;
    constants_info_[185].data_size = 117440512;
    constants_info_[185].from_folded = false;
    constants_info_[185].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[185].shape = {14336, 4096};
    constants_info_[185].stride = {4096, 1};
    constants_info_[185].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[185].original_fqn = "model.layers.20.feed_forward.w1.weight";
    constants_info_[186].name = "model_layers_20_feed_forward_w2_weight";
    constants_info_[186].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[186].offset = 0;
    constants_info_[186].data_size = 117440512;
    constants_info_[186].from_folded = false;
    constants_info_[186].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[186].shape = {4096, 14336};
    constants_info_[186].stride = {14336, 1};
    constants_info_[186].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[186].original_fqn = "model.layers.20.feed_forward.w2.weight";
    constants_info_[187].name = "model_layers_20_feed_forward_w3_weight";
    constants_info_[187].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[187].offset = 0;
    constants_info_[187].data_size = 117440512;
    constants_info_[187].from_folded = false;
    constants_info_[187].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[187].shape = {14336, 4096};
    constants_info_[187].stride = {4096, 1};
    constants_info_[187].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[187].original_fqn = "model.layers.20.feed_forward.w3.weight";
    constants_info_[188].name = "model_layers_20_ffn_norm_weight";
    constants_info_[188].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[188].offset = 0;
    constants_info_[188].data_size = 8192;
    constants_info_[188].from_folded = false;
    constants_info_[188].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[188].shape = {4096};
    constants_info_[188].stride = {1};
    constants_info_[188].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[188].original_fqn = "model.layers.20.ffn_norm.weight";
    constants_info_[189].name = "model_layers_20_attention_norm_weight";
    constants_info_[189].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[189].offset = 0;
    constants_info_[189].data_size = 8192;
    constants_info_[189].from_folded = false;
    constants_info_[189].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[189].shape = {4096};
    constants_info_[189].stride = {1};
    constants_info_[189].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[189].original_fqn = "model.layers.20.attention_norm.weight";
    constants_info_[190].name = "model_layers_21_attention_wq_weight";
    constants_info_[190].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[190].offset = 0;
    constants_info_[190].data_size = 33554432;
    constants_info_[190].from_folded = false;
    constants_info_[190].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[190].shape = {4096, 4096};
    constants_info_[190].stride = {4096, 1};
    constants_info_[190].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[190].original_fqn = "model.layers.21.attention.wq.weight";
    constants_info_[191].name = "model_layers_21_attention_wk_weight";
    constants_info_[191].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[191].offset = 0;
    constants_info_[191].data_size = 8388608;
    constants_info_[191].from_folded = false;
    constants_info_[191].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[191].shape = {1024, 4096};
    constants_info_[191].stride = {4096, 1};
    constants_info_[191].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[191].original_fqn = "model.layers.21.attention.wk.weight";
    constants_info_[192].name = "model_layers_21_attention_wv_weight";
    constants_info_[192].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[192].offset = 0;
    constants_info_[192].data_size = 8388608;
    constants_info_[192].from_folded = false;
    constants_info_[192].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[192].shape = {1024, 4096};
    constants_info_[192].stride = {4096, 1};
    constants_info_[192].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[192].original_fqn = "model.layers.21.attention.wv.weight";
    constants_info_[193].name = "model_layers_21_attention_wo_weight";
    constants_info_[193].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[193].offset = 0;
    constants_info_[193].data_size = 33554432;
    constants_info_[193].from_folded = false;
    constants_info_[193].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[193].shape = {4096, 4096};
    constants_info_[193].stride = {4096, 1};
    constants_info_[193].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[193].original_fqn = "model.layers.21.attention.wo.weight";
    constants_info_[194].name = "model_layers_21_feed_forward_w1_weight";
    constants_info_[194].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[194].offset = 0;
    constants_info_[194].data_size = 117440512;
    constants_info_[194].from_folded = false;
    constants_info_[194].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[194].shape = {14336, 4096};
    constants_info_[194].stride = {4096, 1};
    constants_info_[194].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[194].original_fqn = "model.layers.21.feed_forward.w1.weight";
    constants_info_[195].name = "model_layers_21_feed_forward_w2_weight";
    constants_info_[195].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[195].offset = 0;
    constants_info_[195].data_size = 117440512;
    constants_info_[195].from_folded = false;
    constants_info_[195].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[195].shape = {4096, 14336};
    constants_info_[195].stride = {14336, 1};
    constants_info_[195].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[195].original_fqn = "model.layers.21.feed_forward.w2.weight";
    constants_info_[196].name = "model_layers_21_feed_forward_w3_weight";
    constants_info_[196].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[196].offset = 0;
    constants_info_[196].data_size = 117440512;
    constants_info_[196].from_folded = false;
    constants_info_[196].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[196].shape = {14336, 4096};
    constants_info_[196].stride = {4096, 1};
    constants_info_[196].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[196].original_fqn = "model.layers.21.feed_forward.w3.weight";
    constants_info_[197].name = "model_layers_21_ffn_norm_weight";
    constants_info_[197].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[197].offset = 0;
    constants_info_[197].data_size = 8192;
    constants_info_[197].from_folded = false;
    constants_info_[197].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[197].shape = {4096};
    constants_info_[197].stride = {1};
    constants_info_[197].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[197].original_fqn = "model.layers.21.ffn_norm.weight";
    constants_info_[198].name = "model_layers_21_attention_norm_weight";
    constants_info_[198].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[198].offset = 0;
    constants_info_[198].data_size = 8192;
    constants_info_[198].from_folded = false;
    constants_info_[198].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[198].shape = {4096};
    constants_info_[198].stride = {1};
    constants_info_[198].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[198].original_fqn = "model.layers.21.attention_norm.weight";
    constants_info_[199].name = "model_layers_22_attention_wq_weight";
    constants_info_[199].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[199].offset = 0;
    constants_info_[199].data_size = 33554432;
    constants_info_[199].from_folded = false;
    constants_info_[199].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[199].shape = {4096, 4096};
    constants_info_[199].stride = {4096, 1};
    constants_info_[199].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[199].original_fqn = "model.layers.22.attention.wq.weight";
    constants_info_[200].name = "model_layers_22_attention_wk_weight";
    constants_info_[200].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[200].offset = 0;
    constants_info_[200].data_size = 8388608;
    constants_info_[200].from_folded = false;
    constants_info_[200].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[200].shape = {1024, 4096};
    constants_info_[200].stride = {4096, 1};
    constants_info_[200].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[200].original_fqn = "model.layers.22.attention.wk.weight";
    constants_info_[201].name = "model_layers_22_attention_wv_weight";
    constants_info_[201].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[201].offset = 0;
    constants_info_[201].data_size = 8388608;
    constants_info_[201].from_folded = false;
    constants_info_[201].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[201].shape = {1024, 4096};
    constants_info_[201].stride = {4096, 1};
    constants_info_[201].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[201].original_fqn = "model.layers.22.attention.wv.weight";
    constants_info_[202].name = "model_layers_22_attention_wo_weight";
    constants_info_[202].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[202].offset = 0;
    constants_info_[202].data_size = 33554432;
    constants_info_[202].from_folded = false;
    constants_info_[202].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[202].shape = {4096, 4096};
    constants_info_[202].stride = {4096, 1};
    constants_info_[202].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[202].original_fqn = "model.layers.22.attention.wo.weight";
    constants_info_[203].name = "model_layers_22_feed_forward_w1_weight";
    constants_info_[203].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[203].offset = 0;
    constants_info_[203].data_size = 117440512;
    constants_info_[203].from_folded = false;
    constants_info_[203].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[203].shape = {14336, 4096};
    constants_info_[203].stride = {4096, 1};
    constants_info_[203].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[203].original_fqn = "model.layers.22.feed_forward.w1.weight";
    constants_info_[204].name = "model_layers_22_feed_forward_w2_weight";
    constants_info_[204].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[204].offset = 0;
    constants_info_[204].data_size = 117440512;
    constants_info_[204].from_folded = false;
    constants_info_[204].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[204].shape = {4096, 14336};
    constants_info_[204].stride = {14336, 1};
    constants_info_[204].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[204].original_fqn = "model.layers.22.feed_forward.w2.weight";
    constants_info_[205].name = "model_layers_22_feed_forward_w3_weight";
    constants_info_[205].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[205].offset = 0;
    constants_info_[205].data_size = 117440512;
    constants_info_[205].from_folded = false;
    constants_info_[205].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[205].shape = {14336, 4096};
    constants_info_[205].stride = {4096, 1};
    constants_info_[205].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[205].original_fqn = "model.layers.22.feed_forward.w3.weight";
    constants_info_[206].name = "model_layers_22_ffn_norm_weight";
    constants_info_[206].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[206].offset = 0;
    constants_info_[206].data_size = 8192;
    constants_info_[206].from_folded = false;
    constants_info_[206].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[206].shape = {4096};
    constants_info_[206].stride = {1};
    constants_info_[206].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[206].original_fqn = "model.layers.22.ffn_norm.weight";
    constants_info_[207].name = "model_layers_22_attention_norm_weight";
    constants_info_[207].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[207].offset = 0;
    constants_info_[207].data_size = 8192;
    constants_info_[207].from_folded = false;
    constants_info_[207].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[207].shape = {4096};
    constants_info_[207].stride = {1};
    constants_info_[207].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[207].original_fqn = "model.layers.22.attention_norm.weight";
    constants_info_[208].name = "model_layers_23_attention_wq_weight";
    constants_info_[208].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[208].offset = 0;
    constants_info_[208].data_size = 33554432;
    constants_info_[208].from_folded = false;
    constants_info_[208].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[208].shape = {4096, 4096};
    constants_info_[208].stride = {4096, 1};
    constants_info_[208].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[208].original_fqn = "model.layers.23.attention.wq.weight";
    constants_info_[209].name = "model_layers_23_attention_wk_weight";
    constants_info_[209].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[209].offset = 0;
    constants_info_[209].data_size = 8388608;
    constants_info_[209].from_folded = false;
    constants_info_[209].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[209].shape = {1024, 4096};
    constants_info_[209].stride = {4096, 1};
    constants_info_[209].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[209].original_fqn = "model.layers.23.attention.wk.weight";
    constants_info_[210].name = "model_layers_23_attention_wv_weight";
    constants_info_[210].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[210].offset = 0;
    constants_info_[210].data_size = 8388608;
    constants_info_[210].from_folded = false;
    constants_info_[210].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[210].shape = {1024, 4096};
    constants_info_[210].stride = {4096, 1};
    constants_info_[210].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[210].original_fqn = "model.layers.23.attention.wv.weight";
    constants_info_[211].name = "model_layers_23_attention_wo_weight";
    constants_info_[211].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[211].offset = 0;
    constants_info_[211].data_size = 33554432;
    constants_info_[211].from_folded = false;
    constants_info_[211].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[211].shape = {4096, 4096};
    constants_info_[211].stride = {4096, 1};
    constants_info_[211].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[211].original_fqn = "model.layers.23.attention.wo.weight";
    constants_info_[212].name = "model_layers_23_feed_forward_w1_weight";
    constants_info_[212].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[212].offset = 0;
    constants_info_[212].data_size = 117440512;
    constants_info_[212].from_folded = false;
    constants_info_[212].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[212].shape = {14336, 4096};
    constants_info_[212].stride = {4096, 1};
    constants_info_[212].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[212].original_fqn = "model.layers.23.feed_forward.w1.weight";
    constants_info_[213].name = "model_layers_23_feed_forward_w2_weight";
    constants_info_[213].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[213].offset = 0;
    constants_info_[213].data_size = 117440512;
    constants_info_[213].from_folded = false;
    constants_info_[213].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[213].shape = {4096, 14336};
    constants_info_[213].stride = {14336, 1};
    constants_info_[213].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[213].original_fqn = "model.layers.23.feed_forward.w2.weight";
    constants_info_[214].name = "model_layers_23_feed_forward_w3_weight";
    constants_info_[214].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[214].offset = 0;
    constants_info_[214].data_size = 117440512;
    constants_info_[214].from_folded = false;
    constants_info_[214].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[214].shape = {14336, 4096};
    constants_info_[214].stride = {4096, 1};
    constants_info_[214].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[214].original_fqn = "model.layers.23.feed_forward.w3.weight";
    constants_info_[215].name = "model_layers_23_ffn_norm_weight";
    constants_info_[215].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[215].offset = 0;
    constants_info_[215].data_size = 8192;
    constants_info_[215].from_folded = false;
    constants_info_[215].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[215].shape = {4096};
    constants_info_[215].stride = {1};
    constants_info_[215].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[215].original_fqn = "model.layers.23.ffn_norm.weight";
    constants_info_[216].name = "model_layers_23_attention_norm_weight";
    constants_info_[216].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[216].offset = 0;
    constants_info_[216].data_size = 8192;
    constants_info_[216].from_folded = false;
    constants_info_[216].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[216].shape = {4096};
    constants_info_[216].stride = {1};
    constants_info_[216].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[216].original_fqn = "model.layers.23.attention_norm.weight";
    constants_info_[217].name = "model_layers_24_attention_wq_weight";
    constants_info_[217].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[217].offset = 0;
    constants_info_[217].data_size = 33554432;
    constants_info_[217].from_folded = false;
    constants_info_[217].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[217].shape = {4096, 4096};
    constants_info_[217].stride = {4096, 1};
    constants_info_[217].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[217].original_fqn = "model.layers.24.attention.wq.weight";
    constants_info_[218].name = "model_layers_24_attention_wk_weight";
    constants_info_[218].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[218].offset = 0;
    constants_info_[218].data_size = 8388608;
    constants_info_[218].from_folded = false;
    constants_info_[218].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[218].shape = {1024, 4096};
    constants_info_[218].stride = {4096, 1};
    constants_info_[218].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[218].original_fqn = "model.layers.24.attention.wk.weight";
    constants_info_[219].name = "model_layers_24_attention_wv_weight";
    constants_info_[219].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[219].offset = 0;
    constants_info_[219].data_size = 8388608;
    constants_info_[219].from_folded = false;
    constants_info_[219].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[219].shape = {1024, 4096};
    constants_info_[219].stride = {4096, 1};
    constants_info_[219].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[219].original_fqn = "model.layers.24.attention.wv.weight";
    constants_info_[220].name = "model_layers_24_attention_wo_weight";
    constants_info_[220].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[220].offset = 0;
    constants_info_[220].data_size = 33554432;
    constants_info_[220].from_folded = false;
    constants_info_[220].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[220].shape = {4096, 4096};
    constants_info_[220].stride = {4096, 1};
    constants_info_[220].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[220].original_fqn = "model.layers.24.attention.wo.weight";
    constants_info_[221].name = "model_layers_24_feed_forward_w1_weight";
    constants_info_[221].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[221].offset = 0;
    constants_info_[221].data_size = 117440512;
    constants_info_[221].from_folded = false;
    constants_info_[221].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[221].shape = {14336, 4096};
    constants_info_[221].stride = {4096, 1};
    constants_info_[221].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[221].original_fqn = "model.layers.24.feed_forward.w1.weight";
    constants_info_[222].name = "model_layers_24_feed_forward_w2_weight";
    constants_info_[222].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[222].offset = 0;
    constants_info_[222].data_size = 117440512;
    constants_info_[222].from_folded = false;
    constants_info_[222].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[222].shape = {4096, 14336};
    constants_info_[222].stride = {14336, 1};
    constants_info_[222].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[222].original_fqn = "model.layers.24.feed_forward.w2.weight";
    constants_info_[223].name = "model_layers_24_feed_forward_w3_weight";
    constants_info_[223].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[223].offset = 0;
    constants_info_[223].data_size = 117440512;
    constants_info_[223].from_folded = false;
    constants_info_[223].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[223].shape = {14336, 4096};
    constants_info_[223].stride = {4096, 1};
    constants_info_[223].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[223].original_fqn = "model.layers.24.feed_forward.w3.weight";
    constants_info_[224].name = "model_layers_24_ffn_norm_weight";
    constants_info_[224].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[224].offset = 0;
    constants_info_[224].data_size = 8192;
    constants_info_[224].from_folded = false;
    constants_info_[224].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[224].shape = {4096};
    constants_info_[224].stride = {1};
    constants_info_[224].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[224].original_fqn = "model.layers.24.ffn_norm.weight";
    constants_info_[225].name = "model_layers_24_attention_norm_weight";
    constants_info_[225].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[225].offset = 0;
    constants_info_[225].data_size = 8192;
    constants_info_[225].from_folded = false;
    constants_info_[225].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[225].shape = {4096};
    constants_info_[225].stride = {1};
    constants_info_[225].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[225].original_fqn = "model.layers.24.attention_norm.weight";
    constants_info_[226].name = "model_layers_25_attention_wq_weight";
    constants_info_[226].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[226].offset = 0;
    constants_info_[226].data_size = 33554432;
    constants_info_[226].from_folded = false;
    constants_info_[226].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[226].shape = {4096, 4096};
    constants_info_[226].stride = {4096, 1};
    constants_info_[226].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[226].original_fqn = "model.layers.25.attention.wq.weight";
    constants_info_[227].name = "model_layers_25_attention_wk_weight";
    constants_info_[227].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[227].offset = 0;
    constants_info_[227].data_size = 8388608;
    constants_info_[227].from_folded = false;
    constants_info_[227].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[227].shape = {1024, 4096};
    constants_info_[227].stride = {4096, 1};
    constants_info_[227].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[227].original_fqn = "model.layers.25.attention.wk.weight";
    constants_info_[228].name = "model_layers_25_attention_wv_weight";
    constants_info_[228].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[228].offset = 0;
    constants_info_[228].data_size = 8388608;
    constants_info_[228].from_folded = false;
    constants_info_[228].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[228].shape = {1024, 4096};
    constants_info_[228].stride = {4096, 1};
    constants_info_[228].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[228].original_fqn = "model.layers.25.attention.wv.weight";
    constants_info_[229].name = "model_layers_25_attention_wo_weight";
    constants_info_[229].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[229].offset = 0;
    constants_info_[229].data_size = 33554432;
    constants_info_[229].from_folded = false;
    constants_info_[229].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[229].shape = {4096, 4096};
    constants_info_[229].stride = {4096, 1};
    constants_info_[229].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[229].original_fqn = "model.layers.25.attention.wo.weight";
    constants_info_[230].name = "model_layers_25_feed_forward_w1_weight";
    constants_info_[230].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[230].offset = 0;
    constants_info_[230].data_size = 117440512;
    constants_info_[230].from_folded = false;
    constants_info_[230].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[230].shape = {14336, 4096};
    constants_info_[230].stride = {4096, 1};
    constants_info_[230].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[230].original_fqn = "model.layers.25.feed_forward.w1.weight";
    constants_info_[231].name = "model_layers_25_feed_forward_w2_weight";
    constants_info_[231].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[231].offset = 0;
    constants_info_[231].data_size = 117440512;
    constants_info_[231].from_folded = false;
    constants_info_[231].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[231].shape = {4096, 14336};
    constants_info_[231].stride = {14336, 1};
    constants_info_[231].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[231].original_fqn = "model.layers.25.feed_forward.w2.weight";
    constants_info_[232].name = "model_layers_25_feed_forward_w3_weight";
    constants_info_[232].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[232].offset = 0;
    constants_info_[232].data_size = 117440512;
    constants_info_[232].from_folded = false;
    constants_info_[232].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[232].shape = {14336, 4096};
    constants_info_[232].stride = {4096, 1};
    constants_info_[232].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[232].original_fqn = "model.layers.25.feed_forward.w3.weight";
    constants_info_[233].name = "model_layers_25_ffn_norm_weight";
    constants_info_[233].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[233].offset = 0;
    constants_info_[233].data_size = 8192;
    constants_info_[233].from_folded = false;
    constants_info_[233].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[233].shape = {4096};
    constants_info_[233].stride = {1};
    constants_info_[233].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[233].original_fqn = "model.layers.25.ffn_norm.weight";
    constants_info_[234].name = "model_layers_25_attention_norm_weight";
    constants_info_[234].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[234].offset = 0;
    constants_info_[234].data_size = 8192;
    constants_info_[234].from_folded = false;
    constants_info_[234].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[234].shape = {4096};
    constants_info_[234].stride = {1};
    constants_info_[234].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[234].original_fqn = "model.layers.25.attention_norm.weight";
    constants_info_[235].name = "model_layers_26_attention_wq_weight";
    constants_info_[235].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[235].offset = 0;
    constants_info_[235].data_size = 33554432;
    constants_info_[235].from_folded = false;
    constants_info_[235].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[235].shape = {4096, 4096};
    constants_info_[235].stride = {4096, 1};
    constants_info_[235].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[235].original_fqn = "model.layers.26.attention.wq.weight";
    constants_info_[236].name = "model_layers_26_attention_wk_weight";
    constants_info_[236].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[236].offset = 0;
    constants_info_[236].data_size = 8388608;
    constants_info_[236].from_folded = false;
    constants_info_[236].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[236].shape = {1024, 4096};
    constants_info_[236].stride = {4096, 1};
    constants_info_[236].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[236].original_fqn = "model.layers.26.attention.wk.weight";
    constants_info_[237].name = "model_layers_26_attention_wv_weight";
    constants_info_[237].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[237].offset = 0;
    constants_info_[237].data_size = 8388608;
    constants_info_[237].from_folded = false;
    constants_info_[237].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[237].shape = {1024, 4096};
    constants_info_[237].stride = {4096, 1};
    constants_info_[237].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[237].original_fqn = "model.layers.26.attention.wv.weight";
    constants_info_[238].name = "model_layers_26_attention_wo_weight";
    constants_info_[238].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[238].offset = 0;
    constants_info_[238].data_size = 33554432;
    constants_info_[238].from_folded = false;
    constants_info_[238].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[238].shape = {4096, 4096};
    constants_info_[238].stride = {4096, 1};
    constants_info_[238].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[238].original_fqn = "model.layers.26.attention.wo.weight";
    constants_info_[239].name = "model_layers_26_feed_forward_w1_weight";
    constants_info_[239].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[239].offset = 0;
    constants_info_[239].data_size = 117440512;
    constants_info_[239].from_folded = false;
    constants_info_[239].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[239].shape = {14336, 4096};
    constants_info_[239].stride = {4096, 1};
    constants_info_[239].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[239].original_fqn = "model.layers.26.feed_forward.w1.weight";
    constants_info_[240].name = "model_layers_26_feed_forward_w2_weight";
    constants_info_[240].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[240].offset = 0;
    constants_info_[240].data_size = 117440512;
    constants_info_[240].from_folded = false;
    constants_info_[240].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[240].shape = {4096, 14336};
    constants_info_[240].stride = {14336, 1};
    constants_info_[240].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[240].original_fqn = "model.layers.26.feed_forward.w2.weight";
    constants_info_[241].name = "model_layers_26_feed_forward_w3_weight";
    constants_info_[241].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[241].offset = 0;
    constants_info_[241].data_size = 117440512;
    constants_info_[241].from_folded = false;
    constants_info_[241].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[241].shape = {14336, 4096};
    constants_info_[241].stride = {4096, 1};
    constants_info_[241].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[241].original_fqn = "model.layers.26.feed_forward.w3.weight";
    constants_info_[242].name = "model_layers_26_ffn_norm_weight";
    constants_info_[242].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[242].offset = 0;
    constants_info_[242].data_size = 8192;
    constants_info_[242].from_folded = false;
    constants_info_[242].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[242].shape = {4096};
    constants_info_[242].stride = {1};
    constants_info_[242].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[242].original_fqn = "model.layers.26.ffn_norm.weight";
    constants_info_[243].name = "model_layers_26_attention_norm_weight";
    constants_info_[243].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[243].offset = 0;
    constants_info_[243].data_size = 8192;
    constants_info_[243].from_folded = false;
    constants_info_[243].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[243].shape = {4096};
    constants_info_[243].stride = {1};
    constants_info_[243].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[243].original_fqn = "model.layers.26.attention_norm.weight";
    constants_info_[244].name = "model_layers_27_attention_wq_weight";
    constants_info_[244].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[244].offset = 0;
    constants_info_[244].data_size = 33554432;
    constants_info_[244].from_folded = false;
    constants_info_[244].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[244].shape = {4096, 4096};
    constants_info_[244].stride = {4096, 1};
    constants_info_[244].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[244].original_fqn = "model.layers.27.attention.wq.weight";
    constants_info_[245].name = "model_layers_27_attention_wk_weight";
    constants_info_[245].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[245].offset = 0;
    constants_info_[245].data_size = 8388608;
    constants_info_[245].from_folded = false;
    constants_info_[245].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[245].shape = {1024, 4096};
    constants_info_[245].stride = {4096, 1};
    constants_info_[245].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[245].original_fqn = "model.layers.27.attention.wk.weight";
    constants_info_[246].name = "model_layers_27_attention_wv_weight";
    constants_info_[246].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[246].offset = 0;
    constants_info_[246].data_size = 8388608;
    constants_info_[246].from_folded = false;
    constants_info_[246].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[246].shape = {1024, 4096};
    constants_info_[246].stride = {4096, 1};
    constants_info_[246].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[246].original_fqn = "model.layers.27.attention.wv.weight";
    constants_info_[247].name = "model_layers_27_attention_wo_weight";
    constants_info_[247].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[247].offset = 0;
    constants_info_[247].data_size = 33554432;
    constants_info_[247].from_folded = false;
    constants_info_[247].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[247].shape = {4096, 4096};
    constants_info_[247].stride = {4096, 1};
    constants_info_[247].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[247].original_fqn = "model.layers.27.attention.wo.weight";
    constants_info_[248].name = "model_layers_27_feed_forward_w1_weight";
    constants_info_[248].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[248].offset = 0;
    constants_info_[248].data_size = 117440512;
    constants_info_[248].from_folded = false;
    constants_info_[248].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[248].shape = {14336, 4096};
    constants_info_[248].stride = {4096, 1};
    constants_info_[248].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[248].original_fqn = "model.layers.27.feed_forward.w1.weight";
    constants_info_[249].name = "model_layers_27_feed_forward_w2_weight";
    constants_info_[249].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[249].offset = 0;
    constants_info_[249].data_size = 117440512;
    constants_info_[249].from_folded = false;
    constants_info_[249].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[249].shape = {4096, 14336};
    constants_info_[249].stride = {14336, 1};
    constants_info_[249].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[249].original_fqn = "model.layers.27.feed_forward.w2.weight";
    constants_info_[250].name = "model_layers_27_feed_forward_w3_weight";
    constants_info_[250].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[250].offset = 0;
    constants_info_[250].data_size = 117440512;
    constants_info_[250].from_folded = false;
    constants_info_[250].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[250].shape = {14336, 4096};
    constants_info_[250].stride = {4096, 1};
    constants_info_[250].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[250].original_fqn = "model.layers.27.feed_forward.w3.weight";
    constants_info_[251].name = "model_layers_27_ffn_norm_weight";
    constants_info_[251].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[251].offset = 0;
    constants_info_[251].data_size = 8192;
    constants_info_[251].from_folded = false;
    constants_info_[251].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[251].shape = {4096};
    constants_info_[251].stride = {1};
    constants_info_[251].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[251].original_fqn = "model.layers.27.ffn_norm.weight";
    constants_info_[252].name = "model_layers_27_attention_norm_weight";
    constants_info_[252].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[252].offset = 0;
    constants_info_[252].data_size = 8192;
    constants_info_[252].from_folded = false;
    constants_info_[252].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[252].shape = {4096};
    constants_info_[252].stride = {1};
    constants_info_[252].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[252].original_fqn = "model.layers.27.attention_norm.weight";
    constants_info_[253].name = "model_layers_28_attention_wq_weight";
    constants_info_[253].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[253].offset = 0;
    constants_info_[253].data_size = 33554432;
    constants_info_[253].from_folded = false;
    constants_info_[253].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[253].shape = {4096, 4096};
    constants_info_[253].stride = {4096, 1};
    constants_info_[253].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[253].original_fqn = "model.layers.28.attention.wq.weight";
    constants_info_[254].name = "model_layers_28_attention_wk_weight";
    constants_info_[254].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[254].offset = 0;
    constants_info_[254].data_size = 8388608;
    constants_info_[254].from_folded = false;
    constants_info_[254].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[254].shape = {1024, 4096};
    constants_info_[254].stride = {4096, 1};
    constants_info_[254].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[254].original_fqn = "model.layers.28.attention.wk.weight";
    constants_info_[255].name = "model_layers_28_attention_wv_weight";
    constants_info_[255].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[255].offset = 0;
    constants_info_[255].data_size = 8388608;
    constants_info_[255].from_folded = false;
    constants_info_[255].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[255].shape = {1024, 4096};
    constants_info_[255].stride = {4096, 1};
    constants_info_[255].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[255].original_fqn = "model.layers.28.attention.wv.weight";
    constants_info_[256].name = "model_layers_28_attention_wo_weight";
    constants_info_[256].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[256].offset = 0;
    constants_info_[256].data_size = 33554432;
    constants_info_[256].from_folded = false;
    constants_info_[256].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[256].shape = {4096, 4096};
    constants_info_[256].stride = {4096, 1};
    constants_info_[256].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[256].original_fqn = "model.layers.28.attention.wo.weight";
    constants_info_[257].name = "model_layers_28_feed_forward_w1_weight";
    constants_info_[257].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[257].offset = 0;
    constants_info_[257].data_size = 117440512;
    constants_info_[257].from_folded = false;
    constants_info_[257].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[257].shape = {14336, 4096};
    constants_info_[257].stride = {4096, 1};
    constants_info_[257].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[257].original_fqn = "model.layers.28.feed_forward.w1.weight";
    constants_info_[258].name = "model_layers_28_feed_forward_w2_weight";
    constants_info_[258].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[258].offset = 0;
    constants_info_[258].data_size = 117440512;
    constants_info_[258].from_folded = false;
    constants_info_[258].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[258].shape = {4096, 14336};
    constants_info_[258].stride = {14336, 1};
    constants_info_[258].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[258].original_fqn = "model.layers.28.feed_forward.w2.weight";
    constants_info_[259].name = "model_layers_28_feed_forward_w3_weight";
    constants_info_[259].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[259].offset = 0;
    constants_info_[259].data_size = 117440512;
    constants_info_[259].from_folded = false;
    constants_info_[259].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[259].shape = {14336, 4096};
    constants_info_[259].stride = {4096, 1};
    constants_info_[259].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[259].original_fqn = "model.layers.28.feed_forward.w3.weight";
    constants_info_[260].name = "model_layers_28_ffn_norm_weight";
    constants_info_[260].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[260].offset = 0;
    constants_info_[260].data_size = 8192;
    constants_info_[260].from_folded = false;
    constants_info_[260].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[260].shape = {4096};
    constants_info_[260].stride = {1};
    constants_info_[260].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[260].original_fqn = "model.layers.28.ffn_norm.weight";
    constants_info_[261].name = "model_layers_28_attention_norm_weight";
    constants_info_[261].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[261].offset = 0;
    constants_info_[261].data_size = 8192;
    constants_info_[261].from_folded = false;
    constants_info_[261].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[261].shape = {4096};
    constants_info_[261].stride = {1};
    constants_info_[261].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[261].original_fqn = "model.layers.28.attention_norm.weight";
    constants_info_[262].name = "model_layers_29_attention_wq_weight";
    constants_info_[262].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[262].offset = 0;
    constants_info_[262].data_size = 33554432;
    constants_info_[262].from_folded = false;
    constants_info_[262].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[262].shape = {4096, 4096};
    constants_info_[262].stride = {4096, 1};
    constants_info_[262].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[262].original_fqn = "model.layers.29.attention.wq.weight";
    constants_info_[263].name = "model_layers_29_attention_wk_weight";
    constants_info_[263].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[263].offset = 0;
    constants_info_[263].data_size = 8388608;
    constants_info_[263].from_folded = false;
    constants_info_[263].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[263].shape = {1024, 4096};
    constants_info_[263].stride = {4096, 1};
    constants_info_[263].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[263].original_fqn = "model.layers.29.attention.wk.weight";
    constants_info_[264].name = "model_layers_29_attention_wv_weight";
    constants_info_[264].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[264].offset = 0;
    constants_info_[264].data_size = 8388608;
    constants_info_[264].from_folded = false;
    constants_info_[264].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[264].shape = {1024, 4096};
    constants_info_[264].stride = {4096, 1};
    constants_info_[264].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[264].original_fqn = "model.layers.29.attention.wv.weight";
    constants_info_[265].name = "model_layers_29_attention_wo_weight";
    constants_info_[265].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[265].offset = 0;
    constants_info_[265].data_size = 33554432;
    constants_info_[265].from_folded = false;
    constants_info_[265].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[265].shape = {4096, 4096};
    constants_info_[265].stride = {4096, 1};
    constants_info_[265].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[265].original_fqn = "model.layers.29.attention.wo.weight";
    constants_info_[266].name = "model_layers_29_feed_forward_w1_weight";
    constants_info_[266].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[266].offset = 0;
    constants_info_[266].data_size = 117440512;
    constants_info_[266].from_folded = false;
    constants_info_[266].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[266].shape = {14336, 4096};
    constants_info_[266].stride = {4096, 1};
    constants_info_[266].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[266].original_fqn = "model.layers.29.feed_forward.w1.weight";
    constants_info_[267].name = "model_layers_29_feed_forward_w2_weight";
    constants_info_[267].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[267].offset = 0;
    constants_info_[267].data_size = 117440512;
    constants_info_[267].from_folded = false;
    constants_info_[267].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[267].shape = {4096, 14336};
    constants_info_[267].stride = {14336, 1};
    constants_info_[267].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[267].original_fqn = "model.layers.29.feed_forward.w2.weight";
    constants_info_[268].name = "model_layers_29_feed_forward_w3_weight";
    constants_info_[268].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[268].offset = 0;
    constants_info_[268].data_size = 117440512;
    constants_info_[268].from_folded = false;
    constants_info_[268].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[268].shape = {14336, 4096};
    constants_info_[268].stride = {4096, 1};
    constants_info_[268].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[268].original_fqn = "model.layers.29.feed_forward.w3.weight";
    constants_info_[269].name = "model_layers_29_ffn_norm_weight";
    constants_info_[269].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[269].offset = 0;
    constants_info_[269].data_size = 8192;
    constants_info_[269].from_folded = false;
    constants_info_[269].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[269].shape = {4096};
    constants_info_[269].stride = {1};
    constants_info_[269].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[269].original_fqn = "model.layers.29.ffn_norm.weight";
    constants_info_[270].name = "model_layers_29_attention_norm_weight";
    constants_info_[270].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[270].offset = 0;
    constants_info_[270].data_size = 8192;
    constants_info_[270].from_folded = false;
    constants_info_[270].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[270].shape = {4096};
    constants_info_[270].stride = {1};
    constants_info_[270].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[270].original_fqn = "model.layers.29.attention_norm.weight";
    constants_info_[271].name = "model_layers_30_attention_wq_weight";
    constants_info_[271].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[271].offset = 0;
    constants_info_[271].data_size = 33554432;
    constants_info_[271].from_folded = false;
    constants_info_[271].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[271].shape = {4096, 4096};
    constants_info_[271].stride = {4096, 1};
    constants_info_[271].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[271].original_fqn = "model.layers.30.attention.wq.weight";
    constants_info_[272].name = "model_layers_30_attention_wk_weight";
    constants_info_[272].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[272].offset = 0;
    constants_info_[272].data_size = 8388608;
    constants_info_[272].from_folded = false;
    constants_info_[272].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[272].shape = {1024, 4096};
    constants_info_[272].stride = {4096, 1};
    constants_info_[272].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[272].original_fqn = "model.layers.30.attention.wk.weight";
    constants_info_[273].name = "model_layers_30_attention_wv_weight";
    constants_info_[273].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[273].offset = 0;
    constants_info_[273].data_size = 8388608;
    constants_info_[273].from_folded = false;
    constants_info_[273].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[273].shape = {1024, 4096};
    constants_info_[273].stride = {4096, 1};
    constants_info_[273].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[273].original_fqn = "model.layers.30.attention.wv.weight";
    constants_info_[274].name = "model_layers_30_attention_wo_weight";
    constants_info_[274].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[274].offset = 0;
    constants_info_[274].data_size = 33554432;
    constants_info_[274].from_folded = false;
    constants_info_[274].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[274].shape = {4096, 4096};
    constants_info_[274].stride = {4096, 1};
    constants_info_[274].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[274].original_fqn = "model.layers.30.attention.wo.weight";
    constants_info_[275].name = "model_layers_30_feed_forward_w1_weight";
    constants_info_[275].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[275].offset = 0;
    constants_info_[275].data_size = 117440512;
    constants_info_[275].from_folded = false;
    constants_info_[275].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[275].shape = {14336, 4096};
    constants_info_[275].stride = {4096, 1};
    constants_info_[275].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[275].original_fqn = "model.layers.30.feed_forward.w1.weight";
    constants_info_[276].name = "model_layers_30_feed_forward_w2_weight";
    constants_info_[276].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[276].offset = 0;
    constants_info_[276].data_size = 117440512;
    constants_info_[276].from_folded = false;
    constants_info_[276].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[276].shape = {4096, 14336};
    constants_info_[276].stride = {14336, 1};
    constants_info_[276].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[276].original_fqn = "model.layers.30.feed_forward.w2.weight";
    constants_info_[277].name = "model_layers_30_feed_forward_w3_weight";
    constants_info_[277].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[277].offset = 0;
    constants_info_[277].data_size = 117440512;
    constants_info_[277].from_folded = false;
    constants_info_[277].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[277].shape = {14336, 4096};
    constants_info_[277].stride = {4096, 1};
    constants_info_[277].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[277].original_fqn = "model.layers.30.feed_forward.w3.weight";
    constants_info_[278].name = "model_layers_30_ffn_norm_weight";
    constants_info_[278].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[278].offset = 0;
    constants_info_[278].data_size = 8192;
    constants_info_[278].from_folded = false;
    constants_info_[278].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[278].shape = {4096};
    constants_info_[278].stride = {1};
    constants_info_[278].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[278].original_fqn = "model.layers.30.ffn_norm.weight";
    constants_info_[279].name = "model_layers_30_attention_norm_weight";
    constants_info_[279].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[279].offset = 0;
    constants_info_[279].data_size = 8192;
    constants_info_[279].from_folded = false;
    constants_info_[279].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[279].shape = {4096};
    constants_info_[279].stride = {1};
    constants_info_[279].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[279].original_fqn = "model.layers.30.attention_norm.weight";
    constants_info_[280].name = "model_layers_31_attention_wq_weight";
    constants_info_[280].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[280].offset = 0;
    constants_info_[280].data_size = 33554432;
    constants_info_[280].from_folded = false;
    constants_info_[280].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[280].shape = {4096, 4096};
    constants_info_[280].stride = {4096, 1};
    constants_info_[280].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[280].original_fqn = "model.layers.31.attention.wq.weight";
    constants_info_[281].name = "model_layers_31_attention_wk_weight";
    constants_info_[281].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[281].offset = 0;
    constants_info_[281].data_size = 8388608;
    constants_info_[281].from_folded = false;
    constants_info_[281].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[281].shape = {1024, 4096};
    constants_info_[281].stride = {4096, 1};
    constants_info_[281].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[281].original_fqn = "model.layers.31.attention.wk.weight";
    constants_info_[282].name = "model_layers_31_attention_wv_weight";
    constants_info_[282].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[282].offset = 0;
    constants_info_[282].data_size = 8388608;
    constants_info_[282].from_folded = false;
    constants_info_[282].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[282].shape = {1024, 4096};
    constants_info_[282].stride = {4096, 1};
    constants_info_[282].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[282].original_fqn = "model.layers.31.attention.wv.weight";
    constants_info_[283].name = "model_layers_31_attention_wo_weight";
    constants_info_[283].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[283].offset = 0;
    constants_info_[283].data_size = 33554432;
    constants_info_[283].from_folded = false;
    constants_info_[283].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[283].shape = {4096, 4096};
    constants_info_[283].stride = {4096, 1};
    constants_info_[283].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[283].original_fqn = "model.layers.31.attention.wo.weight";
    constants_info_[284].name = "model_layers_31_feed_forward_w1_weight";
    constants_info_[284].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[284].offset = 0;
    constants_info_[284].data_size = 117440512;
    constants_info_[284].from_folded = false;
    constants_info_[284].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[284].shape = {14336, 4096};
    constants_info_[284].stride = {4096, 1};
    constants_info_[284].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[284].original_fqn = "model.layers.31.feed_forward.w1.weight";
    constants_info_[285].name = "model_layers_31_feed_forward_w2_weight";
    constants_info_[285].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[285].offset = 0;
    constants_info_[285].data_size = 117440512;
    constants_info_[285].from_folded = false;
    constants_info_[285].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[285].shape = {4096, 14336};
    constants_info_[285].stride = {14336, 1};
    constants_info_[285].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[285].original_fqn = "model.layers.31.feed_forward.w2.weight";
    constants_info_[286].name = "model_layers_31_feed_forward_w3_weight";
    constants_info_[286].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[286].offset = 0;
    constants_info_[286].data_size = 117440512;
    constants_info_[286].from_folded = false;
    constants_info_[286].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[286].shape = {14336, 4096};
    constants_info_[286].stride = {4096, 1};
    constants_info_[286].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[286].original_fqn = "model.layers.31.feed_forward.w3.weight";
    constants_info_[287].name = "model_layers_31_ffn_norm_weight";
    constants_info_[287].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[287].offset = 0;
    constants_info_[287].data_size = 8192;
    constants_info_[287].from_folded = false;
    constants_info_[287].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[287].shape = {4096};
    constants_info_[287].stride = {1};
    constants_info_[287].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[287].original_fqn = "model.layers.31.ffn_norm.weight";
    constants_info_[288].name = "model_layers_31_attention_norm_weight";
    constants_info_[288].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[288].offset = 0;
    constants_info_[288].data_size = 8192;
    constants_info_[288].from_folded = false;
    constants_info_[288].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[288].shape = {4096};
    constants_info_[288].stride = {1};
    constants_info_[288].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[288].original_fqn = "model.layers.31.attention_norm.weight";
    constants_info_[289].name = "model_norm_weight";
    constants_info_[289].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[289].offset = 0;
    constants_info_[289].data_size = 8192;
    constants_info_[289].from_folded = false;
    constants_info_[289].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[289].shape = {4096};
    constants_info_[289].stride = {1};
    constants_info_[289].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[289].original_fqn = "model.norm.weight";
    constants_info_[290].name = "model_output_weight";
    constants_info_[290].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[290].offset = 0;
    constants_info_[290].data_size = 1050673152;
    constants_info_[290].from_folded = false;
    constants_info_[290].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[290].shape = {128256, 4096};
    constants_info_[290].stride = {4096, 1};
    constants_info_[290].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[290].original_fqn = "model.output.weight";
    constants_info_[291].name = "model_freqs_cis";
    constants_info_[291].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[291].offset = 0;
    constants_info_[291].data_size = 67108864;
    constants_info_[291].from_folded = false;
    constants_info_[291].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[291].shape = {262144, 64, 2};
    constants_info_[291].stride = {128, 2, 1};
    constants_info_[291].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[291].original_fqn = "model.freqs_cis";
    constants_info_[292].name = "model_causal_mask";
    constants_info_[292].dtype = static_cast<int32_t>(cached_torch_dtype_bool);
    constants_info_[292].offset = 0;
    constants_info_[292].data_size = 92416;
    constants_info_[292].from_folded = false;
    constants_info_[292].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[292].shape = {304, 304};
    constants_info_[292].stride = {304, 1};
    constants_info_[292].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[292].original_fqn = "model.causal_mask";
    constants_info_[293].name = "model_layers_0_attention_kv_cache_0_k_cache";
    constants_info_[293].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[293].offset = 0;
    constants_info_[293].data_size = 622592;
    constants_info_[293].from_folded = false;
    constants_info_[293].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[293].shape = {1, 8, 304, 128};
    constants_info_[293].stride = {311296, 38912, 128, 1};
    constants_info_[293].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[293].original_fqn = "model.layers.0.attention.kv_cache.0.k_cache";
    constants_info_[294].name = "model_layers_0_attention_kv_cache_0_v_cache";
    constants_info_[294].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[294].offset = 0;
    constants_info_[294].data_size = 622592;
    constants_info_[294].from_folded = false;
    constants_info_[294].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[294].shape = {1, 8, 304, 128};
    constants_info_[294].stride = {311296, 38912, 128, 1};
    constants_info_[294].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[294].original_fqn = "model.layers.0.attention.kv_cache.0.v_cache";
    constants_info_[295].name = "model_layers_1_attention_kv_cache_0_k_cache";
    constants_info_[295].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[295].offset = 0;
    constants_info_[295].data_size = 622592;
    constants_info_[295].from_folded = false;
    constants_info_[295].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[295].shape = {1, 8, 304, 128};
    constants_info_[295].stride = {311296, 38912, 128, 1};
    constants_info_[295].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[295].original_fqn = "model.layers.1.attention.kv_cache.0.k_cache";
    constants_info_[296].name = "model_layers_1_attention_kv_cache_0_v_cache";
    constants_info_[296].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[296].offset = 0;
    constants_info_[296].data_size = 622592;
    constants_info_[296].from_folded = false;
    constants_info_[296].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[296].shape = {1, 8, 304, 128};
    constants_info_[296].stride = {311296, 38912, 128, 1};
    constants_info_[296].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[296].original_fqn = "model.layers.1.attention.kv_cache.0.v_cache";
    constants_info_[297].name = "model_layers_2_attention_kv_cache_0_k_cache";
    constants_info_[297].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[297].offset = 0;
    constants_info_[297].data_size = 622592;
    constants_info_[297].from_folded = false;
    constants_info_[297].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[297].shape = {1, 8, 304, 128};
    constants_info_[297].stride = {311296, 38912, 128, 1};
    constants_info_[297].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[297].original_fqn = "model.layers.2.attention.kv_cache.0.k_cache";
    constants_info_[298].name = "model_layers_2_attention_kv_cache_0_v_cache";
    constants_info_[298].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[298].offset = 0;
    constants_info_[298].data_size = 622592;
    constants_info_[298].from_folded = false;
    constants_info_[298].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[298].shape = {1, 8, 304, 128};
    constants_info_[298].stride = {311296, 38912, 128, 1};
    constants_info_[298].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[298].original_fqn = "model.layers.2.attention.kv_cache.0.v_cache";
    constants_info_[299].name = "model_layers_3_attention_kv_cache_0_k_cache";
    constants_info_[299].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[299].offset = 0;
    constants_info_[299].data_size = 622592;
    constants_info_[299].from_folded = false;
    constants_info_[299].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[299].shape = {1, 8, 304, 128};
    constants_info_[299].stride = {311296, 38912, 128, 1};
    constants_info_[299].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[299].original_fqn = "model.layers.3.attention.kv_cache.0.k_cache";
    constants_info_[300].name = "model_layers_3_attention_kv_cache_0_v_cache";
    constants_info_[300].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[300].offset = 0;
    constants_info_[300].data_size = 622592;
    constants_info_[300].from_folded = false;
    constants_info_[300].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[300].shape = {1, 8, 304, 128};
    constants_info_[300].stride = {311296, 38912, 128, 1};
    constants_info_[300].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[300].original_fqn = "model.layers.3.attention.kv_cache.0.v_cache";
    constants_info_[301].name = "model_layers_4_attention_kv_cache_0_k_cache";
    constants_info_[301].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[301].offset = 0;
    constants_info_[301].data_size = 622592;
    constants_info_[301].from_folded = false;
    constants_info_[301].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[301].shape = {1, 8, 304, 128};
    constants_info_[301].stride = {311296, 38912, 128, 1};
    constants_info_[301].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[301].original_fqn = "model.layers.4.attention.kv_cache.0.k_cache";
    constants_info_[302].name = "model_layers_4_attention_kv_cache_0_v_cache";
    constants_info_[302].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[302].offset = 0;
    constants_info_[302].data_size = 622592;
    constants_info_[302].from_folded = false;
    constants_info_[302].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[302].shape = {1, 8, 304, 128};
    constants_info_[302].stride = {311296, 38912, 128, 1};
    constants_info_[302].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[302].original_fqn = "model.layers.4.attention.kv_cache.0.v_cache";
    constants_info_[303].name = "model_layers_5_attention_kv_cache_0_k_cache";
    constants_info_[303].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[303].offset = 0;
    constants_info_[303].data_size = 622592;
    constants_info_[303].from_folded = false;
    constants_info_[303].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[303].shape = {1, 8, 304, 128};
    constants_info_[303].stride = {311296, 38912, 128, 1};
    constants_info_[303].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[303].original_fqn = "model.layers.5.attention.kv_cache.0.k_cache";
    constants_info_[304].name = "model_layers_5_attention_kv_cache_0_v_cache";
    constants_info_[304].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[304].offset = 0;
    constants_info_[304].data_size = 622592;
    constants_info_[304].from_folded = false;
    constants_info_[304].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[304].shape = {1, 8, 304, 128};
    constants_info_[304].stride = {311296, 38912, 128, 1};
    constants_info_[304].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[304].original_fqn = "model.layers.5.attention.kv_cache.0.v_cache";
    constants_info_[305].name = "model_layers_6_attention_kv_cache_0_k_cache";
    constants_info_[305].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[305].offset = 0;
    constants_info_[305].data_size = 622592;
    constants_info_[305].from_folded = false;
    constants_info_[305].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[305].shape = {1, 8, 304, 128};
    constants_info_[305].stride = {311296, 38912, 128, 1};
    constants_info_[305].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[305].original_fqn = "model.layers.6.attention.kv_cache.0.k_cache";
    constants_info_[306].name = "model_layers_6_attention_kv_cache_0_v_cache";
    constants_info_[306].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[306].offset = 0;
    constants_info_[306].data_size = 622592;
    constants_info_[306].from_folded = false;
    constants_info_[306].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[306].shape = {1, 8, 304, 128};
    constants_info_[306].stride = {311296, 38912, 128, 1};
    constants_info_[306].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[306].original_fqn = "model.layers.6.attention.kv_cache.0.v_cache";
    constants_info_[307].name = "model_layers_7_attention_kv_cache_0_k_cache";
    constants_info_[307].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[307].offset = 0;
    constants_info_[307].data_size = 622592;
    constants_info_[307].from_folded = false;
    constants_info_[307].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[307].shape = {1, 8, 304, 128};
    constants_info_[307].stride = {311296, 38912, 128, 1};
    constants_info_[307].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[307].original_fqn = "model.layers.7.attention.kv_cache.0.k_cache";
    constants_info_[308].name = "model_layers_7_attention_kv_cache_0_v_cache";
    constants_info_[308].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[308].offset = 0;
    constants_info_[308].data_size = 622592;
    constants_info_[308].from_folded = false;
    constants_info_[308].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[308].shape = {1, 8, 304, 128};
    constants_info_[308].stride = {311296, 38912, 128, 1};
    constants_info_[308].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[308].original_fqn = "model.layers.7.attention.kv_cache.0.v_cache";
    constants_info_[309].name = "model_layers_8_attention_kv_cache_0_k_cache";
    constants_info_[309].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[309].offset = 0;
    constants_info_[309].data_size = 622592;
    constants_info_[309].from_folded = false;
    constants_info_[309].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[309].shape = {1, 8, 304, 128};
    constants_info_[309].stride = {311296, 38912, 128, 1};
    constants_info_[309].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[309].original_fqn = "model.layers.8.attention.kv_cache.0.k_cache";
    constants_info_[310].name = "model_layers_8_attention_kv_cache_0_v_cache";
    constants_info_[310].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[310].offset = 0;
    constants_info_[310].data_size = 622592;
    constants_info_[310].from_folded = false;
    constants_info_[310].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[310].shape = {1, 8, 304, 128};
    constants_info_[310].stride = {311296, 38912, 128, 1};
    constants_info_[310].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[310].original_fqn = "model.layers.8.attention.kv_cache.0.v_cache";
    constants_info_[311].name = "model_layers_9_attention_kv_cache_0_k_cache";
    constants_info_[311].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[311].offset = 0;
    constants_info_[311].data_size = 622592;
    constants_info_[311].from_folded = false;
    constants_info_[311].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[311].shape = {1, 8, 304, 128};
    constants_info_[311].stride = {311296, 38912, 128, 1};
    constants_info_[311].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[311].original_fqn = "model.layers.9.attention.kv_cache.0.k_cache";
    constants_info_[312].name = "model_layers_9_attention_kv_cache_0_v_cache";
    constants_info_[312].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[312].offset = 0;
    constants_info_[312].data_size = 622592;
    constants_info_[312].from_folded = false;
    constants_info_[312].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[312].shape = {1, 8, 304, 128};
    constants_info_[312].stride = {311296, 38912, 128, 1};
    constants_info_[312].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[312].original_fqn = "model.layers.9.attention.kv_cache.0.v_cache";
    constants_info_[313].name = "model_layers_10_attention_kv_cache_0_k_cache";
    constants_info_[313].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[313].offset = 0;
    constants_info_[313].data_size = 622592;
    constants_info_[313].from_folded = false;
    constants_info_[313].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[313].shape = {1, 8, 304, 128};
    constants_info_[313].stride = {311296, 38912, 128, 1};
    constants_info_[313].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[313].original_fqn = "model.layers.10.attention.kv_cache.0.k_cache";
    constants_info_[314].name = "model_layers_10_attention_kv_cache_0_v_cache";
    constants_info_[314].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[314].offset = 0;
    constants_info_[314].data_size = 622592;
    constants_info_[314].from_folded = false;
    constants_info_[314].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[314].shape = {1, 8, 304, 128};
    constants_info_[314].stride = {311296, 38912, 128, 1};
    constants_info_[314].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[314].original_fqn = "model.layers.10.attention.kv_cache.0.v_cache";
    constants_info_[315].name = "model_layers_11_attention_kv_cache_0_k_cache";
    constants_info_[315].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[315].offset = 0;
    constants_info_[315].data_size = 622592;
    constants_info_[315].from_folded = false;
    constants_info_[315].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[315].shape = {1, 8, 304, 128};
    constants_info_[315].stride = {311296, 38912, 128, 1};
    constants_info_[315].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[315].original_fqn = "model.layers.11.attention.kv_cache.0.k_cache";
    constants_info_[316].name = "model_layers_11_attention_kv_cache_0_v_cache";
    constants_info_[316].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[316].offset = 0;
    constants_info_[316].data_size = 622592;
    constants_info_[316].from_folded = false;
    constants_info_[316].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[316].shape = {1, 8, 304, 128};
    constants_info_[316].stride = {311296, 38912, 128, 1};
    constants_info_[316].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[316].original_fqn = "model.layers.11.attention.kv_cache.0.v_cache";
    constants_info_[317].name = "model_layers_12_attention_kv_cache_0_k_cache";
    constants_info_[317].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[317].offset = 0;
    constants_info_[317].data_size = 622592;
    constants_info_[317].from_folded = false;
    constants_info_[317].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[317].shape = {1, 8, 304, 128};
    constants_info_[317].stride = {311296, 38912, 128, 1};
    constants_info_[317].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[317].original_fqn = "model.layers.12.attention.kv_cache.0.k_cache";
    constants_info_[318].name = "model_layers_12_attention_kv_cache_0_v_cache";
    constants_info_[318].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[318].offset = 0;
    constants_info_[318].data_size = 622592;
    constants_info_[318].from_folded = false;
    constants_info_[318].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[318].shape = {1, 8, 304, 128};
    constants_info_[318].stride = {311296, 38912, 128, 1};
    constants_info_[318].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[318].original_fqn = "model.layers.12.attention.kv_cache.0.v_cache";
    constants_info_[319].name = "model_layers_13_attention_kv_cache_0_k_cache";
    constants_info_[319].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[319].offset = 0;
    constants_info_[319].data_size = 622592;
    constants_info_[319].from_folded = false;
    constants_info_[319].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[319].shape = {1, 8, 304, 128};
    constants_info_[319].stride = {311296, 38912, 128, 1};
    constants_info_[319].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[319].original_fqn = "model.layers.13.attention.kv_cache.0.k_cache";
    constants_info_[320].name = "model_layers_13_attention_kv_cache_0_v_cache";
    constants_info_[320].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[320].offset = 0;
    constants_info_[320].data_size = 622592;
    constants_info_[320].from_folded = false;
    constants_info_[320].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[320].shape = {1, 8, 304, 128};
    constants_info_[320].stride = {311296, 38912, 128, 1};
    constants_info_[320].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[320].original_fqn = "model.layers.13.attention.kv_cache.0.v_cache";
    constants_info_[321].name = "model_layers_14_attention_kv_cache_0_k_cache";
    constants_info_[321].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[321].offset = 0;
    constants_info_[321].data_size = 622592;
    constants_info_[321].from_folded = false;
    constants_info_[321].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[321].shape = {1, 8, 304, 128};
    constants_info_[321].stride = {311296, 38912, 128, 1};
    constants_info_[321].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[321].original_fqn = "model.layers.14.attention.kv_cache.0.k_cache";
    constants_info_[322].name = "model_layers_14_attention_kv_cache_0_v_cache";
    constants_info_[322].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[322].offset = 0;
    constants_info_[322].data_size = 622592;
    constants_info_[322].from_folded = false;
    constants_info_[322].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[322].shape = {1, 8, 304, 128};
    constants_info_[322].stride = {311296, 38912, 128, 1};
    constants_info_[322].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[322].original_fqn = "model.layers.14.attention.kv_cache.0.v_cache";
    constants_info_[323].name = "model_layers_15_attention_kv_cache_0_k_cache";
    constants_info_[323].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[323].offset = 0;
    constants_info_[323].data_size = 622592;
    constants_info_[323].from_folded = false;
    constants_info_[323].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[323].shape = {1, 8, 304, 128};
    constants_info_[323].stride = {311296, 38912, 128, 1};
    constants_info_[323].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[323].original_fqn = "model.layers.15.attention.kv_cache.0.k_cache";
    constants_info_[324].name = "model_layers_15_attention_kv_cache_0_v_cache";
    constants_info_[324].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[324].offset = 0;
    constants_info_[324].data_size = 622592;
    constants_info_[324].from_folded = false;
    constants_info_[324].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[324].shape = {1, 8, 304, 128};
    constants_info_[324].stride = {311296, 38912, 128, 1};
    constants_info_[324].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[324].original_fqn = "model.layers.15.attention.kv_cache.0.v_cache";
    constants_info_[325].name = "model_layers_16_attention_kv_cache_0_k_cache";
    constants_info_[325].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[325].offset = 0;
    constants_info_[325].data_size = 622592;
    constants_info_[325].from_folded = false;
    constants_info_[325].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[325].shape = {1, 8, 304, 128};
    constants_info_[325].stride = {311296, 38912, 128, 1};
    constants_info_[325].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[325].original_fqn = "model.layers.16.attention.kv_cache.0.k_cache";
    constants_info_[326].name = "model_layers_16_attention_kv_cache_0_v_cache";
    constants_info_[326].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[326].offset = 0;
    constants_info_[326].data_size = 622592;
    constants_info_[326].from_folded = false;
    constants_info_[326].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[326].shape = {1, 8, 304, 128};
    constants_info_[326].stride = {311296, 38912, 128, 1};
    constants_info_[326].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[326].original_fqn = "model.layers.16.attention.kv_cache.0.v_cache";
    constants_info_[327].name = "model_layers_17_attention_kv_cache_0_k_cache";
    constants_info_[327].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[327].offset = 0;
    constants_info_[327].data_size = 622592;
    constants_info_[327].from_folded = false;
    constants_info_[327].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[327].shape = {1, 8, 304, 128};
    constants_info_[327].stride = {311296, 38912, 128, 1};
    constants_info_[327].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[327].original_fqn = "model.layers.17.attention.kv_cache.0.k_cache";
    constants_info_[328].name = "model_layers_17_attention_kv_cache_0_v_cache";
    constants_info_[328].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[328].offset = 0;
    constants_info_[328].data_size = 622592;
    constants_info_[328].from_folded = false;
    constants_info_[328].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[328].shape = {1, 8, 304, 128};
    constants_info_[328].stride = {311296, 38912, 128, 1};
    constants_info_[328].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[328].original_fqn = "model.layers.17.attention.kv_cache.0.v_cache";
    constants_info_[329].name = "model_layers_18_attention_kv_cache_0_k_cache";
    constants_info_[329].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[329].offset = 0;
    constants_info_[329].data_size = 622592;
    constants_info_[329].from_folded = false;
    constants_info_[329].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[329].shape = {1, 8, 304, 128};
    constants_info_[329].stride = {311296, 38912, 128, 1};
    constants_info_[329].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[329].original_fqn = "model.layers.18.attention.kv_cache.0.k_cache";
    constants_info_[330].name = "model_layers_18_attention_kv_cache_0_v_cache";
    constants_info_[330].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[330].offset = 0;
    constants_info_[330].data_size = 622592;
    constants_info_[330].from_folded = false;
    constants_info_[330].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[330].shape = {1, 8, 304, 128};
    constants_info_[330].stride = {311296, 38912, 128, 1};
    constants_info_[330].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[330].original_fqn = "model.layers.18.attention.kv_cache.0.v_cache";
    constants_info_[331].name = "model_layers_19_attention_kv_cache_0_k_cache";
    constants_info_[331].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[331].offset = 0;
    constants_info_[331].data_size = 622592;
    constants_info_[331].from_folded = false;
    constants_info_[331].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[331].shape = {1, 8, 304, 128};
    constants_info_[331].stride = {311296, 38912, 128, 1};
    constants_info_[331].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[331].original_fqn = "model.layers.19.attention.kv_cache.0.k_cache";
    constants_info_[332].name = "model_layers_19_attention_kv_cache_0_v_cache";
    constants_info_[332].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[332].offset = 0;
    constants_info_[332].data_size = 622592;
    constants_info_[332].from_folded = false;
    constants_info_[332].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[332].shape = {1, 8, 304, 128};
    constants_info_[332].stride = {311296, 38912, 128, 1};
    constants_info_[332].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[332].original_fqn = "model.layers.19.attention.kv_cache.0.v_cache";
    constants_info_[333].name = "model_layers_20_attention_kv_cache_0_k_cache";
    constants_info_[333].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[333].offset = 0;
    constants_info_[333].data_size = 622592;
    constants_info_[333].from_folded = false;
    constants_info_[333].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[333].shape = {1, 8, 304, 128};
    constants_info_[333].stride = {311296, 38912, 128, 1};
    constants_info_[333].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[333].original_fqn = "model.layers.20.attention.kv_cache.0.k_cache";
    constants_info_[334].name = "model_layers_20_attention_kv_cache_0_v_cache";
    constants_info_[334].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[334].offset = 0;
    constants_info_[334].data_size = 622592;
    constants_info_[334].from_folded = false;
    constants_info_[334].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[334].shape = {1, 8, 304, 128};
    constants_info_[334].stride = {311296, 38912, 128, 1};
    constants_info_[334].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[334].original_fqn = "model.layers.20.attention.kv_cache.0.v_cache";
    constants_info_[335].name = "model_layers_21_attention_kv_cache_0_k_cache";
    constants_info_[335].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[335].offset = 0;
    constants_info_[335].data_size = 622592;
    constants_info_[335].from_folded = false;
    constants_info_[335].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[335].shape = {1, 8, 304, 128};
    constants_info_[335].stride = {311296, 38912, 128, 1};
    constants_info_[335].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[335].original_fqn = "model.layers.21.attention.kv_cache.0.k_cache";
    constants_info_[336].name = "model_layers_21_attention_kv_cache_0_v_cache";
    constants_info_[336].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[336].offset = 0;
    constants_info_[336].data_size = 622592;
    constants_info_[336].from_folded = false;
    constants_info_[336].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[336].shape = {1, 8, 304, 128};
    constants_info_[336].stride = {311296, 38912, 128, 1};
    constants_info_[336].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[336].original_fqn = "model.layers.21.attention.kv_cache.0.v_cache";
    constants_info_[337].name = "model_layers_22_attention_kv_cache_0_k_cache";
    constants_info_[337].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[337].offset = 0;
    constants_info_[337].data_size = 622592;
    constants_info_[337].from_folded = false;
    constants_info_[337].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[337].shape = {1, 8, 304, 128};
    constants_info_[337].stride = {311296, 38912, 128, 1};
    constants_info_[337].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[337].original_fqn = "model.layers.22.attention.kv_cache.0.k_cache";
    constants_info_[338].name = "model_layers_22_attention_kv_cache_0_v_cache";
    constants_info_[338].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[338].offset = 0;
    constants_info_[338].data_size = 622592;
    constants_info_[338].from_folded = false;
    constants_info_[338].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[338].shape = {1, 8, 304, 128};
    constants_info_[338].stride = {311296, 38912, 128, 1};
    constants_info_[338].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[338].original_fqn = "model.layers.22.attention.kv_cache.0.v_cache";
    constants_info_[339].name = "model_layers_23_attention_kv_cache_0_k_cache";
    constants_info_[339].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[339].offset = 0;
    constants_info_[339].data_size = 622592;
    constants_info_[339].from_folded = false;
    constants_info_[339].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[339].shape = {1, 8, 304, 128};
    constants_info_[339].stride = {311296, 38912, 128, 1};
    constants_info_[339].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[339].original_fqn = "model.layers.23.attention.kv_cache.0.k_cache";
    constants_info_[340].name = "model_layers_23_attention_kv_cache_0_v_cache";
    constants_info_[340].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[340].offset = 0;
    constants_info_[340].data_size = 622592;
    constants_info_[340].from_folded = false;
    constants_info_[340].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[340].shape = {1, 8, 304, 128};
    constants_info_[340].stride = {311296, 38912, 128, 1};
    constants_info_[340].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[340].original_fqn = "model.layers.23.attention.kv_cache.0.v_cache";
    constants_info_[341].name = "model_layers_24_attention_kv_cache_0_k_cache";
    constants_info_[341].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[341].offset = 0;
    constants_info_[341].data_size = 622592;
    constants_info_[341].from_folded = false;
    constants_info_[341].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[341].shape = {1, 8, 304, 128};
    constants_info_[341].stride = {311296, 38912, 128, 1};
    constants_info_[341].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[341].original_fqn = "model.layers.24.attention.kv_cache.0.k_cache";
    constants_info_[342].name = "model_layers_24_attention_kv_cache_0_v_cache";
    constants_info_[342].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[342].offset = 0;
    constants_info_[342].data_size = 622592;
    constants_info_[342].from_folded = false;
    constants_info_[342].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[342].shape = {1, 8, 304, 128};
    constants_info_[342].stride = {311296, 38912, 128, 1};
    constants_info_[342].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[342].original_fqn = "model.layers.24.attention.kv_cache.0.v_cache";
    constants_info_[343].name = "model_layers_25_attention_kv_cache_0_k_cache";
    constants_info_[343].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[343].offset = 0;
    constants_info_[343].data_size = 622592;
    constants_info_[343].from_folded = false;
    constants_info_[343].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[343].shape = {1, 8, 304, 128};
    constants_info_[343].stride = {311296, 38912, 128, 1};
    constants_info_[343].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[343].original_fqn = "model.layers.25.attention.kv_cache.0.k_cache";
    constants_info_[344].name = "model_layers_25_attention_kv_cache_0_v_cache";
    constants_info_[344].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[344].offset = 0;
    constants_info_[344].data_size = 622592;
    constants_info_[344].from_folded = false;
    constants_info_[344].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[344].shape = {1, 8, 304, 128};
    constants_info_[344].stride = {311296, 38912, 128, 1};
    constants_info_[344].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[344].original_fqn = "model.layers.25.attention.kv_cache.0.v_cache";
    constants_info_[345].name = "model_layers_26_attention_kv_cache_0_k_cache";
    constants_info_[345].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[345].offset = 0;
    constants_info_[345].data_size = 622592;
    constants_info_[345].from_folded = false;
    constants_info_[345].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[345].shape = {1, 8, 304, 128};
    constants_info_[345].stride = {311296, 38912, 128, 1};
    constants_info_[345].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[345].original_fqn = "model.layers.26.attention.kv_cache.0.k_cache";
    constants_info_[346].name = "model_layers_26_attention_kv_cache_0_v_cache";
    constants_info_[346].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[346].offset = 0;
    constants_info_[346].data_size = 622592;
    constants_info_[346].from_folded = false;
    constants_info_[346].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[346].shape = {1, 8, 304, 128};
    constants_info_[346].stride = {311296, 38912, 128, 1};
    constants_info_[346].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[346].original_fqn = "model.layers.26.attention.kv_cache.0.v_cache";
    constants_info_[347].name = "model_layers_27_attention_kv_cache_0_k_cache";
    constants_info_[347].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[347].offset = 0;
    constants_info_[347].data_size = 622592;
    constants_info_[347].from_folded = false;
    constants_info_[347].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[347].shape = {1, 8, 304, 128};
    constants_info_[347].stride = {311296, 38912, 128, 1};
    constants_info_[347].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[347].original_fqn = "model.layers.27.attention.kv_cache.0.k_cache";
    constants_info_[348].name = "model_layers_27_attention_kv_cache_0_v_cache";
    constants_info_[348].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[348].offset = 0;
    constants_info_[348].data_size = 622592;
    constants_info_[348].from_folded = false;
    constants_info_[348].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[348].shape = {1, 8, 304, 128};
    constants_info_[348].stride = {311296, 38912, 128, 1};
    constants_info_[348].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[348].original_fqn = "model.layers.27.attention.kv_cache.0.v_cache";
    constants_info_[349].name = "model_layers_28_attention_kv_cache_0_k_cache";
    constants_info_[349].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[349].offset = 0;
    constants_info_[349].data_size = 622592;
    constants_info_[349].from_folded = false;
    constants_info_[349].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[349].shape = {1, 8, 304, 128};
    constants_info_[349].stride = {311296, 38912, 128, 1};
    constants_info_[349].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[349].original_fqn = "model.layers.28.attention.kv_cache.0.k_cache";
    constants_info_[350].name = "model_layers_28_attention_kv_cache_0_v_cache";
    constants_info_[350].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[350].offset = 0;
    constants_info_[350].data_size = 622592;
    constants_info_[350].from_folded = false;
    constants_info_[350].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[350].shape = {1, 8, 304, 128};
    constants_info_[350].stride = {311296, 38912, 128, 1};
    constants_info_[350].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[350].original_fqn = "model.layers.28.attention.kv_cache.0.v_cache";
    constants_info_[351].name = "model_layers_29_attention_kv_cache_0_k_cache";
    constants_info_[351].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[351].offset = 0;
    constants_info_[351].data_size = 622592;
    constants_info_[351].from_folded = false;
    constants_info_[351].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[351].shape = {1, 8, 304, 128};
    constants_info_[351].stride = {311296, 38912, 128, 1};
    constants_info_[351].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[351].original_fqn = "model.layers.29.attention.kv_cache.0.k_cache";
    constants_info_[352].name = "model_layers_29_attention_kv_cache_0_v_cache";
    constants_info_[352].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[352].offset = 0;
    constants_info_[352].data_size = 622592;
    constants_info_[352].from_folded = false;
    constants_info_[352].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[352].shape = {1, 8, 304, 128};
    constants_info_[352].stride = {311296, 38912, 128, 1};
    constants_info_[352].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[352].original_fqn = "model.layers.29.attention.kv_cache.0.v_cache";
    constants_info_[353].name = "model_layers_30_attention_kv_cache_0_k_cache";
    constants_info_[353].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[353].offset = 0;
    constants_info_[353].data_size = 622592;
    constants_info_[353].from_folded = false;
    constants_info_[353].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[353].shape = {1, 8, 304, 128};
    constants_info_[353].stride = {311296, 38912, 128, 1};
    constants_info_[353].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[353].original_fqn = "model.layers.30.attention.kv_cache.0.k_cache";
    constants_info_[354].name = "model_layers_30_attention_kv_cache_0_v_cache";
    constants_info_[354].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[354].offset = 0;
    constants_info_[354].data_size = 622592;
    constants_info_[354].from_folded = false;
    constants_info_[354].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[354].shape = {1, 8, 304, 128};
    constants_info_[354].stride = {311296, 38912, 128, 1};
    constants_info_[354].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[354].original_fqn = "model.layers.30.attention.kv_cache.0.v_cache";
    constants_info_[355].name = "model_layers_31_attention_kv_cache_0_k_cache";
    constants_info_[355].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[355].offset = 0;
    constants_info_[355].data_size = 622592;
    constants_info_[355].from_folded = false;
    constants_info_[355].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[355].shape = {1, 8, 304, 128};
    constants_info_[355].stride = {311296, 38912, 128, 1};
    constants_info_[355].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[355].original_fqn = "model.layers.31.attention.kv_cache.0.k_cache";
    constants_info_[356].name = "model_layers_31_attention_kv_cache_0_v_cache";
    constants_info_[356].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
    constants_info_[356].offset = 0;
    constants_info_[356].data_size = 622592;
    constants_info_[356].from_folded = false;
    constants_info_[356].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[356].shape = {1, 8, 304, 128};
    constants_info_[356].stride = {311296, 38912, 128, 1};
    constants_info_[356].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[356].original_fqn = "model.layers.31.attention.kv_cache.0.v_cache";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}]}, {\"type\": \"builtins.dict\", \"context\": \"[]\", \"children_spec\": []}]}]";
    out_spec_ = "[1, {\"type\": null, \"context\": null, \"children_spec\": []}]";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}
} // namespace torch::aot_inductor

using namespace torch::aot_inductor;
using namespace torch::neutron;

template <typename in_ptr0_type_, typename in_ptr1_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused__to_copy_embedding_mean_pow_0(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused__to_copy_embedding_mean_pow_0 == nullptr) {
        kernels_.triton_red_fused__to_copy_embedding_mean_pow_0 = loadKernel(__triton_red_fused__to_copy_embedding_mean_pow_0_start, "triton_red_fused__to_copy_embedding_mean_pow_0", 128); 
    }
    CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_2 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_3 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_4 = r0_numel;
    CUdeviceptr global_scratch_5 = 0;
    void* kernel_args_[] = {&var_1, &var_2, &var_3, &var_4, &global_scratch_5};
    launchKernel(kernels_.triton_red_fused__to_copy_embedding_mean_pow_0, grid_0, grid_1, grid_2, 32, 128, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_1(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_1 == nullptr) {
        kernels_.triton_red_fused_mm_1 = loadKernel(__triton_red_fused_mm_1_start, "triton_red_fused_mm_1", 64); 
    }
    CUdeviceptr var_6 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_7 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_8 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_9 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_10 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_11 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_12 = xnumel;
    int var_13 = r0_numel;
    CUdeviceptr global_scratch_14 = 0;
    void* kernel_args_[] = {&var_6, &var_7, &var_8, &var_9, &var_10, &var_11, &var_12, &var_13, &global_scratch_14};
    launchKernel(kernels_.triton_red_fused_mm_1, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename in_ptr5_type_, typename in_ptr6_type_, typename out_ptr0_type_, typename out_ptr2_type_, typename kernels_type_>
static inline void call_triton_red_fused_index_put_mm_2(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const in_ptr5_type_& in_ptr5,
    const in_ptr6_type_& in_ptr6,
    const out_ptr0_type_& out_ptr0,
    const out_ptr2_type_& out_ptr2,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (2 - 1)) / (2));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_index_put_mm_2 == nullptr) {
        kernels_.triton_red_fused_index_put_mm_2 = loadKernel(__triton_red_fused_index_put_mm_2_start, "triton_red_fused_index_put_mm_2", 4096); 
    }
    CUdeviceptr var_15 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_16 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_17 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_18 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_19 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_20 = reinterpret_cast<CUdeviceptr>(in_ptr5.data_ptr());
    CUdeviceptr var_21 = reinterpret_cast<CUdeviceptr>(in_ptr6.data_ptr());
    CUdeviceptr var_22 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    CUdeviceptr var_23 = reinterpret_cast<CUdeviceptr>(out_ptr2.data_ptr());
    int var_24 = xnumel;
    int var_25 = r0_numel;
    CUdeviceptr global_scratch_26 = 0;
    void* kernel_args_[] = {&var_15, &var_16, &var_17, &var_18, &var_19, &var_20, &var_21, &var_22, &var_23, &var_24, &var_25, &global_scratch_26};
    launchKernel(kernels_.triton_red_fused_index_put_mm_2, grid_0, grid_1, grid_2, 16, 4096, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_poi_fused_index_put_3(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_index_put_3 == nullptr) {
        kernels_.triton_poi_fused_index_put_3 = loadKernel(__triton_poi_fused_index_put_3_start, "triton_poi_fused_index_put_3", 0); 
    }
    CUdeviceptr var_27 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_28 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_29 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_30 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_31 = xnumel;
    CUdeviceptr global_scratch_32 = 0;
    void* kernel_args_[] = {&var_27, &var_28, &var_29, &var_30, &var_31, &global_scratch_32};
    launchKernel(kernels_.triton_poi_fused_index_put_3, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_poi_fused__to_copy_mul_4(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (16 - 1)) / (16));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused__to_copy_mul_4 == nullptr) {
        kernels_.triton_poi_fused__to_copy_mul_4 = loadKernel(__triton_poi_fused__to_copy_mul_4_start, "triton_poi_fused__to_copy_mul_4", 0); 
    }
    CUdeviceptr var_33 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_34 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_35 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_36 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_37 = xnumel;
    CUdeviceptr global_scratch_38 = 0;
    void* kernel_args_[] = {&var_33, &var_34, &var_35, &var_36, &var_37, &global_scratch_38};
    launchKernel(kernels_.triton_poi_fused__to_copy_mul_4, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused_bmm_5(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (32 - 1)) / (32));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_bmm_5 == nullptr) {
        kernels_.triton_red_fused_bmm_5 = loadKernel(__triton_red_fused_bmm_5_start, "triton_red_fused_bmm_5", 128); 
    }
    CUdeviceptr var_39 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_40 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_41 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_42 = xnumel;
    int var_43 = r0_numel;
    CUdeviceptr global_scratch_44 = 0;
    void* kernel_args_[] = {&var_39, &var_40, &var_41, &var_42, &var_43, &global_scratch_44};
    launchKernel(kernels_.triton_red_fused_bmm_5, grid_0, grid_1, grid_2, 8, 128, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename kernels_type_>
static inline void call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6 == nullptr) {
        kernels_.triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6 = loadKernel(__triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6_start, "triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6", 2048); 
    }
    CUdeviceptr var_45 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_46 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_47 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    int var_48 = xnumel;
    int var_49 = r0_numel;
    CUdeviceptr global_scratch_50 = 0;
    void* kernel_args_[] = {&var_45, &var_46, &var_47, &var_48, &var_49, &global_scratch_50};
    launchKernel(kernels_.triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6, grid_0, grid_1, grid_2, 1, 2048, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused_bmm_7(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_bmm_7 == nullptr) {
        kernels_.triton_red_fused_bmm_7 = loadKernel(__triton_red_fused_bmm_7_start, "triton_red_fused_bmm_7", 4128); 
    }
    CUdeviceptr var_51 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_52 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_53 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_54 = xnumel;
    int var_55 = r0_numel;
    CUdeviceptr global_scratch_56 = 0;
    void* kernel_args_[] = {&var_51, &var_52, &var_53, &var_54, &var_55, &global_scratch_56};
    launchKernel(kernels_.triton_red_fused_bmm_7, grid_0, grid_1, grid_2, 4, 4128, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_per_fused_bmm_8(
    const in_ptr0_type_& in_ptr0,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_per_fused_bmm_8 == nullptr) {
        kernels_.triton_per_fused_bmm_8 = loadKernel(__triton_per_fused_bmm_8_start, "triton_per_fused_bmm_8", 2048); 
    }
    CUdeviceptr var_57 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_58 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_59 = xnumel;
    int var_60 = r0_numel;
    CUdeviceptr global_scratch_61 = 0;
    void* kernel_args_[] = {&var_57, &var_58, &var_59, &var_60, &global_scratch_61};
    launchKernel(kernels_.triton_per_fused_bmm_8, grid_0, grid_1, grid_2, 4, 2048, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_9(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_9 == nullptr) {
        kernels_.triton_red_fused_mm_9 = loadKernel(__triton_red_fused_mm_9_start, "triton_red_fused_mm_9", 64); 
    }
    CUdeviceptr var_62 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_63 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_64 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_65 = xnumel;
    int var_66 = r0_numel;
    CUdeviceptr global_scratch_67 = 0;
    void* kernel_args_[] = {&var_62, &var_63, &var_64, &var_65, &var_66, &global_scratch_67};
    launchKernel(kernels_.triton_red_fused_mm_9, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10 == nullptr) {
        kernels_.triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10 = loadKernel(__triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10_start, "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10", 64); 
    }
    CUdeviceptr var_68 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_69 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_70 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_71 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_72 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_73 = r0_numel;
    CUdeviceptr global_scratch_74 = 0;
    void* kernel_args_[] = {&var_68, &var_69, &var_70, &var_71, &var_72, &var_73, &global_scratch_74};
    launchKernel(kernels_.triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename out_ptr0_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_11(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const out_ptr0_type_& out_ptr0,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_11 == nullptr) {
        kernels_.triton_red_fused_mm_11 = loadKernel(__triton_red_fused_mm_11_start, "triton_red_fused_mm_11", 64); 
    }
    CUdeviceptr var_75 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_76 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_77 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_78 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    CUdeviceptr var_79 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_80 = xnumel;
    int var_81 = r0_numel;
    CUdeviceptr global_scratch_82 = 0;
    void* kernel_args_[] = {&var_75, &var_76, &var_77, &var_78, &var_79, &var_80, &var_81, &global_scratch_82};
    launchKernel(kernels_.triton_red_fused_mm_11, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_12(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (4 - 1)) / (4));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_12 == nullptr) {
        kernels_.triton_red_fused_mm_12 = loadKernel(__triton_red_fused_mm_12_start, "triton_red_fused_mm_12", 8192); 
    }
    CUdeviceptr var_83 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_84 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_85 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_86 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_87 = xnumel;
    int var_88 = r0_numel;
    CUdeviceptr global_scratch_89 = 0;
    void* kernel_args_[] = {&var_83, &var_84, &var_85, &var_86, &var_87, &var_88, &global_scratch_89};
    launchKernel(kernels_.triton_red_fused_mm_12, grid_0, grid_1, grid_2, 16, 8192, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13 == nullptr) {
        kernels_.triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13 = loadKernel(__triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13_start, "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13", 64); 
    }
    CUdeviceptr var_90 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_91 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_92 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_93 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_94 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_95 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_96 = r0_numel;
    CUdeviceptr global_scratch_97 = 0;
    void* kernel_args_[] = {&var_90, &var_91, &var_92, &var_93, &var_94, &var_95, &var_96, &global_scratch_97};
    launchKernel(kernels_.triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_14(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_14 == nullptr) {
        kernels_.triton_red_fused_mm_14 = loadKernel(__triton_red_fused_mm_14_start, "triton_red_fused_mm_14", 64); 
    }
    CUdeviceptr var_98 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_99 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_100 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_101 = xnumel;
    int var_102 = r0_numel;
    CUdeviceptr global_scratch_103 = 0;
    void* kernel_args_[] = {&var_98, &var_99, &var_100, &var_101, &var_102, &global_scratch_103};
    launchKernel(kernels_.triton_red_fused_mm_14, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename out_ptr0_type_, typename out_ptr2_type_, typename kernels_type_>
static inline void call_triton_red_fused_index_put_mm_15(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const out_ptr0_type_& out_ptr0,
    const out_ptr2_type_& out_ptr2,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_index_put_mm_15 == nullptr) {
        kernels_.triton_red_fused_index_put_mm_15 = loadKernel(__triton_red_fused_index_put_mm_15_start, "triton_red_fused_index_put_mm_15", 64); 
    }
    CUdeviceptr var_104 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_105 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_106 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_107 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_108 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    CUdeviceptr var_109 = reinterpret_cast<CUdeviceptr>(out_ptr2.data_ptr());
    int var_110 = xnumel;
    int var_111 = r0_numel;
    CUdeviceptr global_scratch_112 = 0;
    void* kernel_args_[] = {&var_104, &var_105, &var_106, &var_107, &var_108, &var_109, &var_110, &var_111, &global_scratch_112};
    launchKernel(kernels_.triton_red_fused_index_put_mm_15, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename in_ptr5_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused_add_embedding_mm_mul_16(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const in_ptr5_type_& in_ptr5,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (4 - 1)) / (4));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_add_embedding_mm_mul_16 == nullptr) {
        kernels_.triton_red_fused_add_embedding_mm_mul_16 = loadKernel(__triton_red_fused_add_embedding_mm_mul_16_start, "triton_red_fused_add_embedding_mm_mul_16", 16384); 
    }
    CUdeviceptr var_113 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_114 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_115 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_116 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_117 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_118 = reinterpret_cast<CUdeviceptr>(in_ptr5.data_ptr());
    CUdeviceptr var_119 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_120 = xnumel;
    int var_121 = r0_numel;
    CUdeviceptr global_scratch_122 = 0;
    void* kernel_args_[] = {&var_113, &var_114, &var_115, &var_116, &var_117, &var_118, &var_119, &var_120, &var_121, &global_scratch_122};
    launchKernel(kernels_.triton_red_fused_add_embedding_mm_mul_16, grid_0, grid_1, grid_2, 16, 16384, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused__to_copy_mean_pow_17(
    const in_ptr0_type_& in_ptr0,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused__to_copy_mean_pow_17 == nullptr) {
        kernels_.triton_red_fused__to_copy_mean_pow_17 = loadKernel(__triton_red_fused__to_copy_mean_pow_17_start, "triton_red_fused__to_copy_mean_pow_17", 64); 
    }
    CUdeviceptr var_123 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_124 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_125 = r0_numel;
    CUdeviceptr global_scratch_126 = 0;
    void* kernel_args_[] = {&var_123, &var_124, &var_125, &global_scratch_126};
    launchKernel(kernels_.triton_red_fused__to_copy_mean_pow_17, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename out_ptr0_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_18(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const out_ptr0_type_& out_ptr0,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_18 == nullptr) {
        kernels_.triton_red_fused_mm_18 = loadKernel(__triton_red_fused_mm_18_start, "triton_red_fused_mm_18", 64); 
    }
    CUdeviceptr var_127 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_128 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_129 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_130 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_131 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_132 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    CUdeviceptr var_133 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_134 = xnumel;
    int var_135 = r0_numel;
    CUdeviceptr global_scratch_136 = 0;
    void* kernel_args_[] = {&var_127, &var_128, &var_129, &var_130, &var_131, &var_132, &var_133, &var_134, &var_135, &global_scratch_136};
    launchKernel(kernels_.triton_red_fused_mm_18, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused__to_copy_add_mean_mul_pow_19(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused__to_copy_add_mean_mul_pow_19 == nullptr) {
        kernels_.triton_red_fused__to_copy_add_mean_mul_pow_19 = loadKernel(__triton_red_fused__to_copy_add_mean_mul_pow_19_start, "triton_red_fused__to_copy_add_mean_mul_pow_19", 128); 
    }
    CUdeviceptr var_137 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_138 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_139 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_140 = r0_numel;
    CUdeviceptr global_scratch_141 = 0;
    void* kernel_args_[] = {&var_137, &var_138, &var_139, &var_140, &global_scratch_141};
    launchKernel(kernels_.triton_red_fused__to_copy_add_mean_mul_pow_19, grid_0, grid_1, grid_2, 32, 128, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename out_ptr0_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_20(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const out_ptr0_type_& out_ptr0,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_20 == nullptr) {
        kernels_.triton_red_fused_mm_20 = loadKernel(__triton_red_fused_mm_20_start, "triton_red_fused_mm_20", 32); 
    }
    CUdeviceptr var_142 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_143 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_144 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_145 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_146 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_147 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    int var_148 = xnumel;
    int var_149 = r0_numel;
    CUdeviceptr global_scratch_150 = 0;
    void* kernel_args_[] = {&var_142, &var_143, &var_144, &var_145, &var_146, &var_147, &var_148, &var_149, &global_scratch_150};
    launchKernel(kernels_.triton_red_fused_mm_20, grid_0, grid_1, grid_2, 8, 32, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename in_ptr5_type_, typename in_ptr6_type_, typename out_ptr0_type_, typename out_ptr2_type_, typename kernels_type_>
static inline void call_triton_red_fused_index_put_mm_21(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const in_ptr5_type_& in_ptr5,
    const in_ptr6_type_& in_ptr6,
    const out_ptr0_type_& out_ptr0,
    const out_ptr2_type_& out_ptr2,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (2 - 1)) / (2));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_index_put_mm_21 == nullptr) {
        kernels_.triton_red_fused_index_put_mm_21 = loadKernel(__triton_red_fused_index_put_mm_21_start, "triton_red_fused_index_put_mm_21", 2048); 
    }
    CUdeviceptr var_151 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_152 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_153 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_154 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_155 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_156 = reinterpret_cast<CUdeviceptr>(in_ptr5.data_ptr());
    CUdeviceptr var_157 = reinterpret_cast<CUdeviceptr>(in_ptr6.data_ptr());
    CUdeviceptr var_158 = reinterpret_cast<CUdeviceptr>(out_ptr0.data_ptr());
    CUdeviceptr var_159 = reinterpret_cast<CUdeviceptr>(out_ptr2.data_ptr());
    int var_160 = xnumel;
    int var_161 = r0_numel;
    CUdeviceptr global_scratch_162 = 0;
    void* kernel_args_[] = {&var_151, &var_152, &var_153, &var_154, &var_155, &var_156, &var_157, &var_158, &var_159, &var_160, &var_161, &global_scratch_162};
    launchKernel(kernels_.triton_red_fused_index_put_mm_21, grid_0, grid_1, grid_2, 16, 2048, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22 == nullptr) {
        kernels_.triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22 = loadKernel(__triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22_start, "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22", 64); 
    }
    CUdeviceptr var_163 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_164 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_165 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_166 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_167 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_168 = r0_numel;
    CUdeviceptr global_scratch_169 = 0;
    void* kernel_args_[] = {&var_163, &var_164, &var_165, &var_166, &var_167, &var_168, &global_scratch_169};
    launchKernel(kernels_.triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23 == nullptr) {
        kernels_.triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23 = loadKernel(__triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23_start, "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23", 64); 
    }
    CUdeviceptr var_170 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_171 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_172 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_173 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_174 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_175 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_176 = r0_numel;
    CUdeviceptr global_scratch_177 = 0;
    void* kernel_args_[] = {&var_170, &var_171, &var_172, &var_173, &var_174, &var_175, &var_176, &global_scratch_177};
    launchKernel(kernels_.triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23, grid_0, grid_1, grid_2, 16, 64, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename kernels_type_>
static inline void call_triton_red_fused_add_mm_mul_24(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = xnumel;
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_add_mm_mul_24 == nullptr) {
        kernels_.triton_red_fused_add_mm_mul_24 = loadKernel(__triton_red_fused_add_mm_mul_24_start, "triton_red_fused_add_mm_mul_24", 128); 
    }
    CUdeviceptr var_178 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_179 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_180 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_181 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_182 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_183 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    int var_184 = xnumel;
    int var_185 = r0_numel;
    CUdeviceptr global_scratch_186 = 0;
    void* kernel_args_[] = {&var_178, &var_179, &var_180, &var_181, &var_182, &var_183, &var_184, &var_185, &global_scratch_186};
    launchKernel(kernels_.triton_red_fused_add_mm_mul_24, grid_0, grid_1, grid_2, 32, 128, kernel_args_, stream_);
}

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename in_ptr3_type_, typename in_ptr4_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_triton_red_fused_mm_25(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const in_ptr3_type_& in_ptr3,
    const in_ptr4_type_& in_ptr4,
    const out_ptr1_type_& out_ptr1,
    int64_t xnumel,
    int64_t r0_numel,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    uint32_t grid_0 = ((xnumel + (2 - 1)) / (2));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_red_fused_mm_25 == nullptr) {
        kernels_.triton_red_fused_mm_25 = loadKernel(__triton_red_fused_mm_25_start, "triton_red_fused_mm_25", 128); 
    }
    CUdeviceptr var_187 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    CUdeviceptr var_188 = reinterpret_cast<CUdeviceptr>(in_ptr1.data_ptr());
    CUdeviceptr var_189 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_190 = reinterpret_cast<CUdeviceptr>(in_ptr3.data_ptr());
    CUdeviceptr var_191 = reinterpret_cast<CUdeviceptr>(in_ptr4.data_ptr());
    CUdeviceptr var_192 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int var_193 = xnumel;
    int var_194 = r0_numel;
    CUdeviceptr global_scratch_195 = 0;
    void* kernel_args_[] = {&var_187, &var_188, &var_189, &var_190, &var_191, &var_192, &var_193, &var_194, &global_scratch_195};
    launchKernel(kernels_.triton_red_fused_mm_25, grid_0, grid_1, grid_2, 16, 128, kernel_args_, stream_);
}

namespace torch::aot_inductor {

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

bool _check_aoti_runtime_check_inputs_env() {
    const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
    const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
    return result;
}

AOTI_NOINLINE static void __check_inputs_outputs(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
    if (!_check_aoti_runtime_check_inputs_env()){
        return;
    }
    ConstantHandle arg357_1 = ConstantHandle(input_handles[0]);
    int32_t arg357_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg357_1, &arg357_1_dtype));

    int32_t arg357_1_expected_dtype = aoti_torch_dtype_int32();
    if (arg357_1_expected_dtype != arg357_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg357_1_expected_dtype << "(at::kInt), "
           << "but got: " << arg357_1_dtype << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg357_1_size = arg357_1.sizes();

    if (1 != arg357_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 0, "
           << "expected: 1, " << "but got: " << arg357_1_size[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (1 != arg357_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 1, " << "but got: " << arg357_1_size[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg357_1_stride = arg357_1.strides();

    if (1 != arg357_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 1, " << "but got: " << arg357_1_stride[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (1 != arg357_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg357_1_stride[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    ConstantHandle arg358_1 = ConstantHandle(input_handles[1]);
    int32_t arg358_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg358_1, &arg358_1_dtype));

    int32_t arg358_1_expected_dtype = aoti_torch_dtype_int32();
    if (arg358_1_expected_dtype != arg358_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched dtype, "
           << "expected: " << arg358_1_expected_dtype << "(at::kInt), "
           << "but got: " << arg358_1_dtype << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg358_1_size = arg358_1.sizes();

    if (1 != arg358_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched dim value at 0, "
           << "expected: 1, " << "but got: " << arg358_1_size[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg358_1_stride = arg358_1.strides();

    if (1 != arg358_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched stride value at 0, "
           << "expected: 1, " << "but got: " << arg358_1_stride[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }
}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {

__check_inputs_outputs(input_handles, output_handles);

    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 2);
    auto arg357_1 = std::move(inputs[0]);
    auto arg358_1 = std::move(inputs[1]);
    [[maybe_unused]] auto model_tok_embeddings_weight = constants_->at(0);
    [[maybe_unused]] auto model_layers_0_attention_wq_weight = constants_->at(1);
    [[maybe_unused]] auto model_layers_0_attention_wk_weight = constants_->at(2);
    [[maybe_unused]] auto model_layers_0_attention_wv_weight = constants_->at(3);
    [[maybe_unused]] auto model_layers_0_attention_wo_weight = constants_->at(4);
    [[maybe_unused]] auto model_layers_0_feed_forward_w1_weight = constants_->at(5);
    [[maybe_unused]] auto model_layers_0_feed_forward_w2_weight = constants_->at(6);
    [[maybe_unused]] auto model_layers_0_feed_forward_w3_weight = constants_->at(7);
    [[maybe_unused]] auto model_layers_0_ffn_norm_weight = constants_->at(8);
    [[maybe_unused]] auto model_layers_0_attention_norm_weight = constants_->at(9);
    [[maybe_unused]] auto model_layers_1_attention_wq_weight = constants_->at(10);
    [[maybe_unused]] auto model_layers_1_attention_wk_weight = constants_->at(11);
    [[maybe_unused]] auto model_layers_1_attention_wv_weight = constants_->at(12);
    [[maybe_unused]] auto model_layers_1_attention_wo_weight = constants_->at(13);
    [[maybe_unused]] auto model_layers_1_feed_forward_w1_weight = constants_->at(14);
    [[maybe_unused]] auto model_layers_1_feed_forward_w2_weight = constants_->at(15);
    [[maybe_unused]] auto model_layers_1_feed_forward_w3_weight = constants_->at(16);
    [[maybe_unused]] auto model_layers_1_ffn_norm_weight = constants_->at(17);
    [[maybe_unused]] auto model_layers_1_attention_norm_weight = constants_->at(18);
    [[maybe_unused]] auto model_layers_2_attention_wq_weight = constants_->at(19);
    [[maybe_unused]] auto model_layers_2_attention_wk_weight = constants_->at(20);
    [[maybe_unused]] auto model_layers_2_attention_wv_weight = constants_->at(21);
    [[maybe_unused]] auto model_layers_2_attention_wo_weight = constants_->at(22);
    [[maybe_unused]] auto model_layers_2_feed_forward_w1_weight = constants_->at(23);
    [[maybe_unused]] auto model_layers_2_feed_forward_w2_weight = constants_->at(24);
    [[maybe_unused]] auto model_layers_2_feed_forward_w3_weight = constants_->at(25);
    [[maybe_unused]] auto model_layers_2_ffn_norm_weight = constants_->at(26);
    [[maybe_unused]] auto model_layers_2_attention_norm_weight = constants_->at(27);
    [[maybe_unused]] auto model_layers_3_attention_wq_weight = constants_->at(28);
    [[maybe_unused]] auto model_layers_3_attention_wk_weight = constants_->at(29);
    [[maybe_unused]] auto model_layers_3_attention_wv_weight = constants_->at(30);
    [[maybe_unused]] auto model_layers_3_attention_wo_weight = constants_->at(31);
    [[maybe_unused]] auto model_layers_3_feed_forward_w1_weight = constants_->at(32);
    [[maybe_unused]] auto model_layers_3_feed_forward_w2_weight = constants_->at(33);
    [[maybe_unused]] auto model_layers_3_feed_forward_w3_weight = constants_->at(34);
    [[maybe_unused]] auto model_layers_3_ffn_norm_weight = constants_->at(35);
    [[maybe_unused]] auto model_layers_3_attention_norm_weight = constants_->at(36);
    [[maybe_unused]] auto model_layers_4_attention_wq_weight = constants_->at(37);
    [[maybe_unused]] auto model_layers_4_attention_wk_weight = constants_->at(38);
    [[maybe_unused]] auto model_layers_4_attention_wv_weight = constants_->at(39);
    [[maybe_unused]] auto model_layers_4_attention_wo_weight = constants_->at(40);
    [[maybe_unused]] auto model_layers_4_feed_forward_w1_weight = constants_->at(41);
    [[maybe_unused]] auto model_layers_4_feed_forward_w2_weight = constants_->at(42);
    [[maybe_unused]] auto model_layers_4_feed_forward_w3_weight = constants_->at(43);
    [[maybe_unused]] auto model_layers_4_ffn_norm_weight = constants_->at(44);
    [[maybe_unused]] auto model_layers_4_attention_norm_weight = constants_->at(45);
    [[maybe_unused]] auto model_layers_5_attention_wq_weight = constants_->at(46);
    [[maybe_unused]] auto model_layers_5_attention_wk_weight = constants_->at(47);
    [[maybe_unused]] auto model_layers_5_attention_wv_weight = constants_->at(48);
    [[maybe_unused]] auto model_layers_5_attention_wo_weight = constants_->at(49);
    [[maybe_unused]] auto model_layers_5_feed_forward_w1_weight = constants_->at(50);
    [[maybe_unused]] auto model_layers_5_feed_forward_w2_weight = constants_->at(51);
    [[maybe_unused]] auto model_layers_5_feed_forward_w3_weight = constants_->at(52);
    [[maybe_unused]] auto model_layers_5_ffn_norm_weight = constants_->at(53);
    [[maybe_unused]] auto model_layers_5_attention_norm_weight = constants_->at(54);
    [[maybe_unused]] auto model_layers_6_attention_wq_weight = constants_->at(55);
    [[maybe_unused]] auto model_layers_6_attention_wk_weight = constants_->at(56);
    [[maybe_unused]] auto model_layers_6_attention_wv_weight = constants_->at(57);
    [[maybe_unused]] auto model_layers_6_attention_wo_weight = constants_->at(58);
    [[maybe_unused]] auto model_layers_6_feed_forward_w1_weight = constants_->at(59);
    [[maybe_unused]] auto model_layers_6_feed_forward_w2_weight = constants_->at(60);
    [[maybe_unused]] auto model_layers_6_feed_forward_w3_weight = constants_->at(61);
    [[maybe_unused]] auto model_layers_6_ffn_norm_weight = constants_->at(62);
    [[maybe_unused]] auto model_layers_6_attention_norm_weight = constants_->at(63);
    [[maybe_unused]] auto model_layers_7_attention_wq_weight = constants_->at(64);
    [[maybe_unused]] auto model_layers_7_attention_wk_weight = constants_->at(65);
    [[maybe_unused]] auto model_layers_7_attention_wv_weight = constants_->at(66);
    [[maybe_unused]] auto model_layers_7_attention_wo_weight = constants_->at(67);
    [[maybe_unused]] auto model_layers_7_feed_forward_w1_weight = constants_->at(68);
    [[maybe_unused]] auto model_layers_7_feed_forward_w2_weight = constants_->at(69);
    [[maybe_unused]] auto model_layers_7_feed_forward_w3_weight = constants_->at(70);
    [[maybe_unused]] auto model_layers_7_ffn_norm_weight = constants_->at(71);
    [[maybe_unused]] auto model_layers_7_attention_norm_weight = constants_->at(72);
    [[maybe_unused]] auto model_layers_8_attention_wq_weight = constants_->at(73);
    [[maybe_unused]] auto model_layers_8_attention_wk_weight = constants_->at(74);
    [[maybe_unused]] auto model_layers_8_attention_wv_weight = constants_->at(75);
    [[maybe_unused]] auto model_layers_8_attention_wo_weight = constants_->at(76);
    [[maybe_unused]] auto model_layers_8_feed_forward_w1_weight = constants_->at(77);
    [[maybe_unused]] auto model_layers_8_feed_forward_w2_weight = constants_->at(78);
    [[maybe_unused]] auto model_layers_8_feed_forward_w3_weight = constants_->at(79);
    [[maybe_unused]] auto model_layers_8_ffn_norm_weight = constants_->at(80);
    [[maybe_unused]] auto model_layers_8_attention_norm_weight = constants_->at(81);
    [[maybe_unused]] auto model_layers_9_attention_wq_weight = constants_->at(82);
    [[maybe_unused]] auto model_layers_9_attention_wk_weight = constants_->at(83);
    [[maybe_unused]] auto model_layers_9_attention_wv_weight = constants_->at(84);
    [[maybe_unused]] auto model_layers_9_attention_wo_weight = constants_->at(85);
    [[maybe_unused]] auto model_layers_9_feed_forward_w1_weight = constants_->at(86);
    [[maybe_unused]] auto model_layers_9_feed_forward_w2_weight = constants_->at(87);
    [[maybe_unused]] auto model_layers_9_feed_forward_w3_weight = constants_->at(88);
    [[maybe_unused]] auto model_layers_9_ffn_norm_weight = constants_->at(89);
    [[maybe_unused]] auto model_layers_9_attention_norm_weight = constants_->at(90);
    [[maybe_unused]] auto model_layers_10_attention_wq_weight = constants_->at(91);
    [[maybe_unused]] auto model_layers_10_attention_wk_weight = constants_->at(92);
    [[maybe_unused]] auto model_layers_10_attention_wv_weight = constants_->at(93);
    [[maybe_unused]] auto model_layers_10_attention_wo_weight = constants_->at(94);
    [[maybe_unused]] auto model_layers_10_feed_forward_w1_weight = constants_->at(95);
    [[maybe_unused]] auto model_layers_10_feed_forward_w2_weight = constants_->at(96);
    [[maybe_unused]] auto model_layers_10_feed_forward_w3_weight = constants_->at(97);
    [[maybe_unused]] auto model_layers_10_ffn_norm_weight = constants_->at(98);
    [[maybe_unused]] auto model_layers_10_attention_norm_weight = constants_->at(99);
    [[maybe_unused]] auto model_layers_11_attention_wq_weight = constants_->at(100);
    [[maybe_unused]] auto model_layers_11_attention_wk_weight = constants_->at(101);
    [[maybe_unused]] auto model_layers_11_attention_wv_weight = constants_->at(102);
    [[maybe_unused]] auto model_layers_11_attention_wo_weight = constants_->at(103);
    [[maybe_unused]] auto model_layers_11_feed_forward_w1_weight = constants_->at(104);
    [[maybe_unused]] auto model_layers_11_feed_forward_w2_weight = constants_->at(105);
    [[maybe_unused]] auto model_layers_11_feed_forward_w3_weight = constants_->at(106);
    [[maybe_unused]] auto model_layers_11_ffn_norm_weight = constants_->at(107);
    [[maybe_unused]] auto model_layers_11_attention_norm_weight = constants_->at(108);
    [[maybe_unused]] auto model_layers_12_attention_wq_weight = constants_->at(109);
    [[maybe_unused]] auto model_layers_12_attention_wk_weight = constants_->at(110);
    [[maybe_unused]] auto model_layers_12_attention_wv_weight = constants_->at(111);
    [[maybe_unused]] auto model_layers_12_attention_wo_weight = constants_->at(112);
    [[maybe_unused]] auto model_layers_12_feed_forward_w1_weight = constants_->at(113);
    [[maybe_unused]] auto model_layers_12_feed_forward_w2_weight = constants_->at(114);
    [[maybe_unused]] auto model_layers_12_feed_forward_w3_weight = constants_->at(115);
    [[maybe_unused]] auto model_layers_12_ffn_norm_weight = constants_->at(116);
    [[maybe_unused]] auto model_layers_12_attention_norm_weight = constants_->at(117);
    [[maybe_unused]] auto model_layers_13_attention_wq_weight = constants_->at(118);
    [[maybe_unused]] auto model_layers_13_attention_wk_weight = constants_->at(119);
    [[maybe_unused]] auto model_layers_13_attention_wv_weight = constants_->at(120);
    [[maybe_unused]] auto model_layers_13_attention_wo_weight = constants_->at(121);
    [[maybe_unused]] auto model_layers_13_feed_forward_w1_weight = constants_->at(122);
    [[maybe_unused]] auto model_layers_13_feed_forward_w2_weight = constants_->at(123);
    [[maybe_unused]] auto model_layers_13_feed_forward_w3_weight = constants_->at(124);
    [[maybe_unused]] auto model_layers_13_ffn_norm_weight = constants_->at(125);
    [[maybe_unused]] auto model_layers_13_attention_norm_weight = constants_->at(126);
    [[maybe_unused]] auto model_layers_14_attention_wq_weight = constants_->at(127);
    [[maybe_unused]] auto model_layers_14_attention_wk_weight = constants_->at(128);
    [[maybe_unused]] auto model_layers_14_attention_wv_weight = constants_->at(129);
    [[maybe_unused]] auto model_layers_14_attention_wo_weight = constants_->at(130);
    [[maybe_unused]] auto model_layers_14_feed_forward_w1_weight = constants_->at(131);
    [[maybe_unused]] auto model_layers_14_feed_forward_w2_weight = constants_->at(132);
    [[maybe_unused]] auto model_layers_14_feed_forward_w3_weight = constants_->at(133);
    [[maybe_unused]] auto model_layers_14_ffn_norm_weight = constants_->at(134);
    [[maybe_unused]] auto model_layers_14_attention_norm_weight = constants_->at(135);
    [[maybe_unused]] auto model_layers_15_attention_wq_weight = constants_->at(136);
    [[maybe_unused]] auto model_layers_15_attention_wk_weight = constants_->at(137);
    [[maybe_unused]] auto model_layers_15_attention_wv_weight = constants_->at(138);
    [[maybe_unused]] auto model_layers_15_attention_wo_weight = constants_->at(139);
    [[maybe_unused]] auto model_layers_15_feed_forward_w1_weight = constants_->at(140);
    [[maybe_unused]] auto model_layers_15_feed_forward_w2_weight = constants_->at(141);
    [[maybe_unused]] auto model_layers_15_feed_forward_w3_weight = constants_->at(142);
    [[maybe_unused]] auto model_layers_15_ffn_norm_weight = constants_->at(143);
    [[maybe_unused]] auto model_layers_15_attention_norm_weight = constants_->at(144);
    [[maybe_unused]] auto model_layers_16_attention_wq_weight = constants_->at(145);
    [[maybe_unused]] auto model_layers_16_attention_wk_weight = constants_->at(146);
    [[maybe_unused]] auto model_layers_16_attention_wv_weight = constants_->at(147);
    [[maybe_unused]] auto model_layers_16_attention_wo_weight = constants_->at(148);
    [[maybe_unused]] auto model_layers_16_feed_forward_w1_weight = constants_->at(149);
    [[maybe_unused]] auto model_layers_16_feed_forward_w2_weight = constants_->at(150);
    [[maybe_unused]] auto model_layers_16_feed_forward_w3_weight = constants_->at(151);
    [[maybe_unused]] auto model_layers_16_ffn_norm_weight = constants_->at(152);
    [[maybe_unused]] auto model_layers_16_attention_norm_weight = constants_->at(153);
    [[maybe_unused]] auto model_layers_17_attention_wq_weight = constants_->at(154);
    [[maybe_unused]] auto model_layers_17_attention_wk_weight = constants_->at(155);
    [[maybe_unused]] auto model_layers_17_attention_wv_weight = constants_->at(156);
    [[maybe_unused]] auto model_layers_17_attention_wo_weight = constants_->at(157);
    [[maybe_unused]] auto model_layers_17_feed_forward_w1_weight = constants_->at(158);
    [[maybe_unused]] auto model_layers_17_feed_forward_w2_weight = constants_->at(159);
    [[maybe_unused]] auto model_layers_17_feed_forward_w3_weight = constants_->at(160);
    [[maybe_unused]] auto model_layers_17_ffn_norm_weight = constants_->at(161);
    [[maybe_unused]] auto model_layers_17_attention_norm_weight = constants_->at(162);
    [[maybe_unused]] auto model_layers_18_attention_wq_weight = constants_->at(163);
    [[maybe_unused]] auto model_layers_18_attention_wk_weight = constants_->at(164);
    [[maybe_unused]] auto model_layers_18_attention_wv_weight = constants_->at(165);
    [[maybe_unused]] auto model_layers_18_attention_wo_weight = constants_->at(166);
    [[maybe_unused]] auto model_layers_18_feed_forward_w1_weight = constants_->at(167);
    [[maybe_unused]] auto model_layers_18_feed_forward_w2_weight = constants_->at(168);
    [[maybe_unused]] auto model_layers_18_feed_forward_w3_weight = constants_->at(169);
    [[maybe_unused]] auto model_layers_18_ffn_norm_weight = constants_->at(170);
    [[maybe_unused]] auto model_layers_18_attention_norm_weight = constants_->at(171);
    [[maybe_unused]] auto model_layers_19_attention_wq_weight = constants_->at(172);
    [[maybe_unused]] auto model_layers_19_attention_wk_weight = constants_->at(173);
    [[maybe_unused]] auto model_layers_19_attention_wv_weight = constants_->at(174);
    [[maybe_unused]] auto model_layers_19_attention_wo_weight = constants_->at(175);
    [[maybe_unused]] auto model_layers_19_feed_forward_w1_weight = constants_->at(176);
    [[maybe_unused]] auto model_layers_19_feed_forward_w2_weight = constants_->at(177);
    [[maybe_unused]] auto model_layers_19_feed_forward_w3_weight = constants_->at(178);
    [[maybe_unused]] auto model_layers_19_ffn_norm_weight = constants_->at(179);
    [[maybe_unused]] auto model_layers_19_attention_norm_weight = constants_->at(180);
    [[maybe_unused]] auto model_layers_20_attention_wq_weight = constants_->at(181);
    [[maybe_unused]] auto model_layers_20_attention_wk_weight = constants_->at(182);
    [[maybe_unused]] auto model_layers_20_attention_wv_weight = constants_->at(183);
    [[maybe_unused]] auto model_layers_20_attention_wo_weight = constants_->at(184);
    [[maybe_unused]] auto model_layers_20_feed_forward_w1_weight = constants_->at(185);
    [[maybe_unused]] auto model_layers_20_feed_forward_w2_weight = constants_->at(186);
    [[maybe_unused]] auto model_layers_20_feed_forward_w3_weight = constants_->at(187);
    [[maybe_unused]] auto model_layers_20_ffn_norm_weight = constants_->at(188);
    [[maybe_unused]] auto model_layers_20_attention_norm_weight = constants_->at(189);
    [[maybe_unused]] auto model_layers_21_attention_wq_weight = constants_->at(190);
    [[maybe_unused]] auto model_layers_21_attention_wk_weight = constants_->at(191);
    [[maybe_unused]] auto model_layers_21_attention_wv_weight = constants_->at(192);
    [[maybe_unused]] auto model_layers_21_attention_wo_weight = constants_->at(193);
    [[maybe_unused]] auto model_layers_21_feed_forward_w1_weight = constants_->at(194);
    [[maybe_unused]] auto model_layers_21_feed_forward_w2_weight = constants_->at(195);
    [[maybe_unused]] auto model_layers_21_feed_forward_w3_weight = constants_->at(196);
    [[maybe_unused]] auto model_layers_21_ffn_norm_weight = constants_->at(197);
    [[maybe_unused]] auto model_layers_21_attention_norm_weight = constants_->at(198);
    [[maybe_unused]] auto model_layers_22_attention_wq_weight = constants_->at(199);
    [[maybe_unused]] auto model_layers_22_attention_wk_weight = constants_->at(200);
    [[maybe_unused]] auto model_layers_22_attention_wv_weight = constants_->at(201);
    [[maybe_unused]] auto model_layers_22_attention_wo_weight = constants_->at(202);
    [[maybe_unused]] auto model_layers_22_feed_forward_w1_weight = constants_->at(203);
    [[maybe_unused]] auto model_layers_22_feed_forward_w2_weight = constants_->at(204);
    [[maybe_unused]] auto model_layers_22_feed_forward_w3_weight = constants_->at(205);
    [[maybe_unused]] auto model_layers_22_ffn_norm_weight = constants_->at(206);
    [[maybe_unused]] auto model_layers_22_attention_norm_weight = constants_->at(207);
    [[maybe_unused]] auto model_layers_23_attention_wq_weight = constants_->at(208);
    [[maybe_unused]] auto model_layers_23_attention_wk_weight = constants_->at(209);
    [[maybe_unused]] auto model_layers_23_attention_wv_weight = constants_->at(210);
    [[maybe_unused]] auto model_layers_23_attention_wo_weight = constants_->at(211);
    [[maybe_unused]] auto model_layers_23_feed_forward_w1_weight = constants_->at(212);
    [[maybe_unused]] auto model_layers_23_feed_forward_w2_weight = constants_->at(213);
    [[maybe_unused]] auto model_layers_23_feed_forward_w3_weight = constants_->at(214);
    [[maybe_unused]] auto model_layers_23_ffn_norm_weight = constants_->at(215);
    [[maybe_unused]] auto model_layers_23_attention_norm_weight = constants_->at(216);
    [[maybe_unused]] auto model_layers_24_attention_wq_weight = constants_->at(217);
    [[maybe_unused]] auto model_layers_24_attention_wk_weight = constants_->at(218);
    [[maybe_unused]] auto model_layers_24_attention_wv_weight = constants_->at(219);
    [[maybe_unused]] auto model_layers_24_attention_wo_weight = constants_->at(220);
    [[maybe_unused]] auto model_layers_24_feed_forward_w1_weight = constants_->at(221);
    [[maybe_unused]] auto model_layers_24_feed_forward_w2_weight = constants_->at(222);
    [[maybe_unused]] auto model_layers_24_feed_forward_w3_weight = constants_->at(223);
    [[maybe_unused]] auto model_layers_24_ffn_norm_weight = constants_->at(224);
    [[maybe_unused]] auto model_layers_24_attention_norm_weight = constants_->at(225);
    [[maybe_unused]] auto model_layers_25_attention_wq_weight = constants_->at(226);
    [[maybe_unused]] auto model_layers_25_attention_wk_weight = constants_->at(227);
    [[maybe_unused]] auto model_layers_25_attention_wv_weight = constants_->at(228);
    [[maybe_unused]] auto model_layers_25_attention_wo_weight = constants_->at(229);
    [[maybe_unused]] auto model_layers_25_feed_forward_w1_weight = constants_->at(230);
    [[maybe_unused]] auto model_layers_25_feed_forward_w2_weight = constants_->at(231);
    [[maybe_unused]] auto model_layers_25_feed_forward_w3_weight = constants_->at(232);
    [[maybe_unused]] auto model_layers_25_ffn_norm_weight = constants_->at(233);
    [[maybe_unused]] auto model_layers_25_attention_norm_weight = constants_->at(234);
    [[maybe_unused]] auto model_layers_26_attention_wq_weight = constants_->at(235);
    [[maybe_unused]] auto model_layers_26_attention_wk_weight = constants_->at(236);
    [[maybe_unused]] auto model_layers_26_attention_wv_weight = constants_->at(237);
    [[maybe_unused]] auto model_layers_26_attention_wo_weight = constants_->at(238);
    [[maybe_unused]] auto model_layers_26_feed_forward_w1_weight = constants_->at(239);
    [[maybe_unused]] auto model_layers_26_feed_forward_w2_weight = constants_->at(240);
    [[maybe_unused]] auto model_layers_26_feed_forward_w3_weight = constants_->at(241);
    [[maybe_unused]] auto model_layers_26_ffn_norm_weight = constants_->at(242);
    [[maybe_unused]] auto model_layers_26_attention_norm_weight = constants_->at(243);
    [[maybe_unused]] auto model_layers_27_attention_wq_weight = constants_->at(244);
    [[maybe_unused]] auto model_layers_27_attention_wk_weight = constants_->at(245);
    [[maybe_unused]] auto model_layers_27_attention_wv_weight = constants_->at(246);
    [[maybe_unused]] auto model_layers_27_attention_wo_weight = constants_->at(247);
    [[maybe_unused]] auto model_layers_27_feed_forward_w1_weight = constants_->at(248);
    [[maybe_unused]] auto model_layers_27_feed_forward_w2_weight = constants_->at(249);
    [[maybe_unused]] auto model_layers_27_feed_forward_w3_weight = constants_->at(250);
    [[maybe_unused]] auto model_layers_27_ffn_norm_weight = constants_->at(251);
    [[maybe_unused]] auto model_layers_27_attention_norm_weight = constants_->at(252);
    [[maybe_unused]] auto model_layers_28_attention_wq_weight = constants_->at(253);
    [[maybe_unused]] auto model_layers_28_attention_wk_weight = constants_->at(254);
    [[maybe_unused]] auto model_layers_28_attention_wv_weight = constants_->at(255);
    [[maybe_unused]] auto model_layers_28_attention_wo_weight = constants_->at(256);
    [[maybe_unused]] auto model_layers_28_feed_forward_w1_weight = constants_->at(257);
    [[maybe_unused]] auto model_layers_28_feed_forward_w2_weight = constants_->at(258);
    [[maybe_unused]] auto model_layers_28_feed_forward_w3_weight = constants_->at(259);
    [[maybe_unused]] auto model_layers_28_ffn_norm_weight = constants_->at(260);
    [[maybe_unused]] auto model_layers_28_attention_norm_weight = constants_->at(261);
    [[maybe_unused]] auto model_layers_29_attention_wq_weight = constants_->at(262);
    [[maybe_unused]] auto model_layers_29_attention_wk_weight = constants_->at(263);
    [[maybe_unused]] auto model_layers_29_attention_wv_weight = constants_->at(264);
    [[maybe_unused]] auto model_layers_29_attention_wo_weight = constants_->at(265);
    [[maybe_unused]] auto model_layers_29_feed_forward_w1_weight = constants_->at(266);
    [[maybe_unused]] auto model_layers_29_feed_forward_w2_weight = constants_->at(267);
    [[maybe_unused]] auto model_layers_29_feed_forward_w3_weight = constants_->at(268);
    [[maybe_unused]] auto model_layers_29_ffn_norm_weight = constants_->at(269);
    [[maybe_unused]] auto model_layers_29_attention_norm_weight = constants_->at(270);
    [[maybe_unused]] auto model_layers_30_attention_wq_weight = constants_->at(271);
    [[maybe_unused]] auto model_layers_30_attention_wk_weight = constants_->at(272);
    [[maybe_unused]] auto model_layers_30_attention_wv_weight = constants_->at(273);
    [[maybe_unused]] auto model_layers_30_attention_wo_weight = constants_->at(274);
    [[maybe_unused]] auto model_layers_30_feed_forward_w1_weight = constants_->at(275);
    [[maybe_unused]] auto model_layers_30_feed_forward_w2_weight = constants_->at(276);
    [[maybe_unused]] auto model_layers_30_feed_forward_w3_weight = constants_->at(277);
    [[maybe_unused]] auto model_layers_30_ffn_norm_weight = constants_->at(278);
    [[maybe_unused]] auto model_layers_30_attention_norm_weight = constants_->at(279);
    [[maybe_unused]] auto model_layers_31_attention_wq_weight = constants_->at(280);
    [[maybe_unused]] auto model_layers_31_attention_wk_weight = constants_->at(281);
    [[maybe_unused]] auto model_layers_31_attention_wv_weight = constants_->at(282);
    [[maybe_unused]] auto model_layers_31_attention_wo_weight = constants_->at(283);
    [[maybe_unused]] auto model_layers_31_feed_forward_w1_weight = constants_->at(284);
    [[maybe_unused]] auto model_layers_31_feed_forward_w2_weight = constants_->at(285);
    [[maybe_unused]] auto model_layers_31_feed_forward_w3_weight = constants_->at(286);
    [[maybe_unused]] auto model_layers_31_ffn_norm_weight = constants_->at(287);
    [[maybe_unused]] auto model_layers_31_attention_norm_weight = constants_->at(288);
    [[maybe_unused]] auto model_norm_weight = constants_->at(289);
    [[maybe_unused]] auto model_output_weight = constants_->at(290);
    [[maybe_unused]] auto model_freqs_cis = constants_->at(291);
    [[maybe_unused]] auto model_causal_mask = constants_->at(292);
    [[maybe_unused]] auto model_layers_0_attention_kv_cache_0_k_cache = constants_->at(293);
    [[maybe_unused]] auto model_layers_0_attention_kv_cache_0_v_cache = constants_->at(294);
    [[maybe_unused]] auto model_layers_1_attention_kv_cache_0_k_cache = constants_->at(295);
    [[maybe_unused]] auto model_layers_1_attention_kv_cache_0_v_cache = constants_->at(296);
    [[maybe_unused]] auto model_layers_2_attention_kv_cache_0_k_cache = constants_->at(297);
    [[maybe_unused]] auto model_layers_2_attention_kv_cache_0_v_cache = constants_->at(298);
    [[maybe_unused]] auto model_layers_3_attention_kv_cache_0_k_cache = constants_->at(299);
    [[maybe_unused]] auto model_layers_3_attention_kv_cache_0_v_cache = constants_->at(300);
    [[maybe_unused]] auto model_layers_4_attention_kv_cache_0_k_cache = constants_->at(301);
    [[maybe_unused]] auto model_layers_4_attention_kv_cache_0_v_cache = constants_->at(302);
    [[maybe_unused]] auto model_layers_5_attention_kv_cache_0_k_cache = constants_->at(303);
    [[maybe_unused]] auto model_layers_5_attention_kv_cache_0_v_cache = constants_->at(304);
    [[maybe_unused]] auto model_layers_6_attention_kv_cache_0_k_cache = constants_->at(305);
    [[maybe_unused]] auto model_layers_6_attention_kv_cache_0_v_cache = constants_->at(306);
    [[maybe_unused]] auto model_layers_7_attention_kv_cache_0_k_cache = constants_->at(307);
    [[maybe_unused]] auto model_layers_7_attention_kv_cache_0_v_cache = constants_->at(308);
    [[maybe_unused]] auto model_layers_8_attention_kv_cache_0_k_cache = constants_->at(309);
    [[maybe_unused]] auto model_layers_8_attention_kv_cache_0_v_cache = constants_->at(310);
    [[maybe_unused]] auto model_layers_9_attention_kv_cache_0_k_cache = constants_->at(311);
    [[maybe_unused]] auto model_layers_9_attention_kv_cache_0_v_cache = constants_->at(312);
    [[maybe_unused]] auto model_layers_10_attention_kv_cache_0_k_cache = constants_->at(313);
    [[maybe_unused]] auto model_layers_10_attention_kv_cache_0_v_cache = constants_->at(314);
    [[maybe_unused]] auto model_layers_11_attention_kv_cache_0_k_cache = constants_->at(315);
    [[maybe_unused]] auto model_layers_11_attention_kv_cache_0_v_cache = constants_->at(316);
    [[maybe_unused]] auto model_layers_12_attention_kv_cache_0_k_cache = constants_->at(317);
    [[maybe_unused]] auto model_layers_12_attention_kv_cache_0_v_cache = constants_->at(318);
    [[maybe_unused]] auto model_layers_13_attention_kv_cache_0_k_cache = constants_->at(319);
    [[maybe_unused]] auto model_layers_13_attention_kv_cache_0_v_cache = constants_->at(320);
    [[maybe_unused]] auto model_layers_14_attention_kv_cache_0_k_cache = constants_->at(321);
    [[maybe_unused]] auto model_layers_14_attention_kv_cache_0_v_cache = constants_->at(322);
    [[maybe_unused]] auto model_layers_15_attention_kv_cache_0_k_cache = constants_->at(323);
    [[maybe_unused]] auto model_layers_15_attention_kv_cache_0_v_cache = constants_->at(324);
    [[maybe_unused]] auto model_layers_16_attention_kv_cache_0_k_cache = constants_->at(325);
    [[maybe_unused]] auto model_layers_16_attention_kv_cache_0_v_cache = constants_->at(326);
    [[maybe_unused]] auto model_layers_17_attention_kv_cache_0_k_cache = constants_->at(327);
    [[maybe_unused]] auto model_layers_17_attention_kv_cache_0_v_cache = constants_->at(328);
    [[maybe_unused]] auto model_layers_18_attention_kv_cache_0_k_cache = constants_->at(329);
    [[maybe_unused]] auto model_layers_18_attention_kv_cache_0_v_cache = constants_->at(330);
    [[maybe_unused]] auto model_layers_19_attention_kv_cache_0_k_cache = constants_->at(331);
    [[maybe_unused]] auto model_layers_19_attention_kv_cache_0_v_cache = constants_->at(332);
    [[maybe_unused]] auto model_layers_20_attention_kv_cache_0_k_cache = constants_->at(333);
    [[maybe_unused]] auto model_layers_20_attention_kv_cache_0_v_cache = constants_->at(334);
    [[maybe_unused]] auto model_layers_21_attention_kv_cache_0_k_cache = constants_->at(335);
    [[maybe_unused]] auto model_layers_21_attention_kv_cache_0_v_cache = constants_->at(336);
    [[maybe_unused]] auto model_layers_22_attention_kv_cache_0_k_cache = constants_->at(337);
    [[maybe_unused]] auto model_layers_22_attention_kv_cache_0_v_cache = constants_->at(338);
    [[maybe_unused]] auto model_layers_23_attention_kv_cache_0_k_cache = constants_->at(339);
    [[maybe_unused]] auto model_layers_23_attention_kv_cache_0_v_cache = constants_->at(340);
    [[maybe_unused]] auto model_layers_24_attention_kv_cache_0_k_cache = constants_->at(341);
    [[maybe_unused]] auto model_layers_24_attention_kv_cache_0_v_cache = constants_->at(342);
    [[maybe_unused]] auto model_layers_25_attention_kv_cache_0_k_cache = constants_->at(343);
    [[maybe_unused]] auto model_layers_25_attention_kv_cache_0_v_cache = constants_->at(344);
    [[maybe_unused]] auto model_layers_26_attention_kv_cache_0_k_cache = constants_->at(345);
    [[maybe_unused]] auto model_layers_26_attention_kv_cache_0_v_cache = constants_->at(346);
    [[maybe_unused]] auto model_layers_27_attention_kv_cache_0_k_cache = constants_->at(347);
    [[maybe_unused]] auto model_layers_27_attention_kv_cache_0_v_cache = constants_->at(348);
    [[maybe_unused]] auto model_layers_28_attention_kv_cache_0_k_cache = constants_->at(349);
    [[maybe_unused]] auto model_layers_28_attention_kv_cache_0_v_cache = constants_->at(350);
    [[maybe_unused]] auto model_layers_29_attention_kv_cache_0_k_cache = constants_->at(351);
    [[maybe_unused]] auto model_layers_29_attention_kv_cache_0_v_cache = constants_->at(352);
    [[maybe_unused]] auto model_layers_30_attention_kv_cache_0_k_cache = constants_->at(353);
    [[maybe_unused]] auto model_layers_30_attention_kv_cache_0_v_cache = constants_->at(354);
    [[maybe_unused]] auto model_layers_31_attention_kv_cache_0_k_cache = constants_->at(355);
    [[maybe_unused]] auto model_layers_31_attention_kv_cache_0_v_cache = constants_->at(356);

    if ((long(arg357_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 0 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg357_1_aligned;
        aoti_torch_clone_preserve_strides(arg357_1, &arg357_1_aligned);
        arg357_1 = std::move(RAIIAtenTensorHandle(arg357_1_aligned));
    }

    if ((long(arg358_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 1 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg358_1_aligned;
        aoti_torch_clone_preserve_strides(arg358_1, &arg358_1_aligned);
        arg358_1 = std::move(RAIIAtenTensorHandle(arg358_1_aligned));
    }
    inputs.clear();
    auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());

    AOTICudaStreamGuard stream_guard(stream, this->device_idx_);
    static constexpr int64_t int_array_0[] = {1L, 1L, 1L};
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(3, int_array_0, int_array_0, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    // Topologically Sorted Source Nodes: [embedding, rms_norm], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_embedding_mean_pow_0(arg357_1, model_tok_embeddings_weight, buf0, 1L, 4096L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_1[] = {1L, 4096L};
    static constexpr int64_t int_array_2[] = {4096L, 1L};
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_1, int_array_2, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    // Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
    call_triton_red_fused_mm_1(arg357_1, model_tok_embeddings_weight, buf0, model_layers_0_attention_norm_weight, model_layers_0_attention_wq_weight, buf1, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_3[] = {1L, 1024L};
    static constexpr int64_t int_array_4[] = {1024L, 1L};
    AtenTensorHandle buf2_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_3, int_array_4, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf2_handle));
    RAIIAtenTensorHandle buf2(buf2_handle);
    // Topologically Sorted Source Nodes: [linear_1, linear_2, index_put__1], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_2(arg357_1, model_tok_embeddings_weight, buf0, model_layers_0_attention_norm_weight, model_layers_0_attention_wk_weight, model_layers_0_attention_wv_weight, arg358_1, buf2, model_layers_0_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put_], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf2, model_freqs_cis, model_layers_0_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_5[] = {1L, 32L, 1L, 128L};
    static constexpr int64_t int_array_6[] = {4096L, 128L, 128L, 1L};
    AtenTensorHandle buf6_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_5, int_array_6, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf6_handle));
    RAIIAtenTensorHandle buf6(buf6_handle);
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf1, arg358_1, model_freqs_cis, buf6, 4096L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_7[] = {32L, 1L, 304L};
    static constexpr int64_t int_array_8[] = {304L, 304L, 1L};
    AtenTensorHandle buf7_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(3, int_array_7, int_array_8, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf7_handle));
    RAIIAtenTensorHandle buf7(buf7_handle);
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf6, model_layers_0_attention_kv_cache_0_k_cache, buf7, 9728L, 128L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_9[] = {1L, 32L, 1L, 304L};
    static constexpr int64_t int_array_10[] = {9728L, 304L, 9728L, 1L};
    auto buf11 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf7, 4, int_array_9, int_array_10, 0L)); buf7.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf11, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_11[] = {32L, 1L, 128L, 3L};
    static constexpr int64_t int_array_12[] = {384L, 12288L, 1L, 128L};
    AtenTensorHandle buf12_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_11, int_array_12, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf12_handle));
    RAIIAtenTensorHandle buf12(buf12_handle);
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf11, model_layers_0_attention_kv_cache_0_v_cache, buf12, 12288L, 102L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_13[] = {32L, 1L, 128L};
    static constexpr int64_t int_array_14[] = {128L, 128L, 1L};
    auto buf13 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf6, 3, int_array_13, int_array_14, 0L)); buf6.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf12, buf13, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf14 = std::move(buf1);  // reuse
    // Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf13, model_layers_0_attention_wo_weight, buf14, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_15[] = {1L, 1L, 4096L};
    static constexpr int64_t int_array_16[] = {4096L, 4096L, 1L};
    AtenTensorHandle buf16_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(3, int_array_15, int_array_16, cached_torch_dtype_bfloat16, cached_torch_device_type_cuda, this->device_idx_, &buf16_handle));
    RAIIAtenTensorHandle buf16(buf16_handle);
    // Topologically Sorted Source Nodes: [embedding, mul_8, add_2, rms_norm_1], Original ATen: [aten.embedding, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10(arg357_1, model_tok_embeddings_weight, buf14, model_layers_0_ffn_norm_weight, buf16, 1L, 4096L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_17[] = {1L, 14336L};
    static constexpr int64_t int_array_18[] = {14336L, 1L};
    AtenTensorHandle buf17_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_17, int_array_18, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf17_handle));
    RAIIAtenTensorHandle buf17(buf17_handle);
    AtenTensorHandle buf18_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_17, int_array_18, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf18_handle));
    RAIIAtenTensorHandle buf18(buf18_handle);
    // Topologically Sorted Source Nodes: [linear_4, linear_5], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf16, model_layers_0_feed_forward_w1_weight, model_layers_0_feed_forward_w3_weight, buf17, buf18, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf19 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf13, 2, int_array_1, int_array_2, 0L)); buf13.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf17, buf18, model_layers_0_feed_forward_w2_weight, buf19, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf21 = std::move(buf16);  // reuse
    // Topologically Sorted Source Nodes: [embedding, mul_8, add_2, mul_10, add_3, rms_norm_2], Original ATen: [aten.embedding, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_13(arg357_1, model_tok_embeddings_weight, buf14, buf19, model_layers_1_attention_norm_weight, buf21, 1L, 4096L, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf22_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_1, int_array_2, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf22_handle));
    RAIIAtenTensorHandle buf22(buf22_handle);
    // Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf21, model_layers_1_attention_wq_weight, buf22, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf23 = std::move(buf2);  // reuse
    // Topologically Sorted Source Nodes: [linear_8, linear_9, index_put__3], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf21, model_layers_1_attention_wk_weight, model_layers_1_attention_wv_weight, arg358_1, buf23, model_layers_1_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__2], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf23, model_freqs_cis, model_layers_1_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf27_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_5, int_array_6, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf27_handle));
    RAIIAtenTensorHandle buf27(buf27_handle);
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_1], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf22, arg358_1, model_freqs_cis, buf27, 4096L, stream, kernels, this->cubin_dir_);
    auto buf28 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf11, 3, int_array_7, int_array_8, 0L)); buf11.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_1], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf27, model_layers_1_attention_kv_cache_0_k_cache, buf28, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf32 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf28, 4, int_array_9, int_array_10, 0L)); buf28.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_1], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf32, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf33 = std::move(buf12);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_1], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf32, model_layers_1_attention_kv_cache_0_v_cache, buf33, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf34 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf27, 3, int_array_13, int_array_14, 0L)); buf27.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_1], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf33, buf34, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf36 = std::move(buf21);  // reuse
    // Topologically Sorted Source Nodes: [embedding, mul_8, add_2, mul_10, add_3, linear_10, mul_19, add_6], Original ATen: [aten.embedding, aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_embedding_mm_mul_16(buf34, model_layers_1_attention_wo_weight, arg357_1, model_tok_embeddings_weight, buf14, buf19, buf36, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    arg357_1.reset();
    auto buf37 = std::move(buf0);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_3], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf36, buf37, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf38 = std::move(buf18);  // reuse
    auto buf39 = std::move(buf17);  // reuse
    // Topologically Sorted Source Nodes: [linear_11, linear_12], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf36, buf37, model_layers_1_ffn_norm_weight, model_layers_1_feed_forward_w1_weight, model_layers_1_feed_forward_w3_weight, buf38, buf39, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf40 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf34, 2, int_array_1, int_array_2, 0L)); buf34.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf38, buf39, model_layers_1_feed_forward_w2_weight, buf40, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf41 = std::move(buf37);  // reuse
    // Topologically Sorted Source Nodes: [mul_21, add_7, rms_norm_4], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf36, buf40, buf41, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf42 = std::move(buf19);  // reuse
    // Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf36, buf40, buf41, model_layers_2_attention_norm_weight, model_layers_2_attention_wq_weight, buf42, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf43 = std::move(buf23);  // reuse
    // Topologically Sorted Source Nodes: [linear_15, linear_16, index_put__5], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf36, buf40, buf41, model_layers_2_attention_norm_weight, model_layers_2_attention_wk_weight, model_layers_2_attention_wv_weight, arg358_1, buf43, model_layers_2_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__4], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf43, model_freqs_cis, model_layers_2_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf47 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf14, 4, int_array_5, int_array_6, 0L)); buf14.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_2], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf42, arg358_1, model_freqs_cis, buf47, 4096L, stream, kernels, this->cubin_dir_);
    auto buf48 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf32, 3, int_array_7, int_array_8, 0L)); buf32.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_2], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf47, model_layers_2_attention_kv_cache_0_k_cache, buf48, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf52 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf48, 4, int_array_9, int_array_10, 0L)); buf48.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_2], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf52, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf53 = std::move(buf33);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_2], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf52, model_layers_2_attention_kv_cache_0_v_cache, buf53, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf54 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf47, 3, int_array_13, int_array_14, 0L)); buf47.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_2], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf53, buf54, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf55 = std::move(buf42);  // reuse
    // Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf54, model_layers_2_attention_wo_weight, buf55, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf57_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(3, int_array_15, int_array_16, cached_torch_dtype_bfloat16, cached_torch_device_type_cuda, this->device_idx_, &buf57_handle));
    RAIIAtenTensorHandle buf57(buf57_handle);
    // Topologically Sorted Source Nodes: [mul_21, add_7, mul_30, add_10, rms_norm_5], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf36, buf40, buf55, model_layers_2_ffn_norm_weight, buf57, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf58 = std::move(buf39);  // reuse
    auto buf59 = std::move(buf38);  // reuse
    // Topologically Sorted Source Nodes: [linear_18, linear_19], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf57, model_layers_2_feed_forward_w1_weight, model_layers_2_feed_forward_w3_weight, buf58, buf59, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf60 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf54, 2, int_array_1, int_array_2, 0L)); buf54.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf58, buf59, model_layers_2_feed_forward_w2_weight, buf60, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf62 = std::move(buf57);  // reuse
    // Topologically Sorted Source Nodes: [mul_21, add_7, mul_30, add_10, mul_32, add_11, rms_norm_6], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf36, buf40, buf55, buf60, model_layers_3_attention_norm_weight, buf62, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf63 = std::move(buf22);  // reuse
    // Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf62, model_layers_3_attention_wq_weight, buf63, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf64 = std::move(buf43);  // reuse
    // Topologically Sorted Source Nodes: [linear_22, linear_23, index_put__7], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf62, model_layers_3_attention_wk_weight, model_layers_3_attention_wv_weight, arg358_1, buf64, model_layers_3_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__6], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf64, model_freqs_cis, model_layers_3_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf68_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(4, int_array_5, int_array_6, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf68_handle));
    RAIIAtenTensorHandle buf68(buf68_handle);
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_3], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf63, arg358_1, model_freqs_cis, buf68, 4096L, stream, kernels, this->cubin_dir_);
    auto buf69 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf52, 3, int_array_7, int_array_8, 0L)); buf52.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_3], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf68, model_layers_3_attention_kv_cache_0_k_cache, buf69, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf73 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf69, 4, int_array_9, int_array_10, 0L)); buf69.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_3], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf73, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf74 = std::move(buf53);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_3], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf73, model_layers_3_attention_kv_cache_0_v_cache, buf74, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf75 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf68, 3, int_array_13, int_array_14, 0L)); buf68.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_3], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf74, buf75, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf77 = std::move(buf36);  // reuse
    // Topologically Sorted Source Nodes: [mul_21, add_7, mul_30, add_10, mul_32, add_11, linear_24, mul_41, add_14], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf77, buf75, model_layers_3_attention_wo_weight, buf40, buf55, buf60, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf78 = std::move(buf41);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_7], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf77, buf78, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf79 = std::move(buf59);  // reuse
    auto buf80 = std::move(buf58);  // reuse
    // Topologically Sorted Source Nodes: [linear_25, linear_26], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf77, buf78, model_layers_3_ffn_norm_weight, model_layers_3_feed_forward_w1_weight, model_layers_3_feed_forward_w3_weight, buf79, buf80, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf81 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf75, 2, int_array_1, int_array_2, 0L)); buf75.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf79, buf80, model_layers_3_feed_forward_w2_weight, buf81, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf82 = std::move(buf78);  // reuse
    // Topologically Sorted Source Nodes: [mul_43, add_15, rms_norm_8], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf77, buf81, buf82, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf83 = std::move(buf60);  // reuse
    // Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf77, buf81, buf82, model_layers_4_attention_norm_weight, model_layers_4_attention_wq_weight, buf83, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf84 = std::move(buf64);  // reuse
    // Topologically Sorted Source Nodes: [linear_29, linear_30, index_put__9], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf77, buf81, buf82, model_layers_4_attention_norm_weight, model_layers_4_attention_wk_weight, model_layers_4_attention_wv_weight, arg358_1, buf84, model_layers_4_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__8], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf84, model_freqs_cis, model_layers_4_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf88 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf55, 4, int_array_5, int_array_6, 0L)); buf55.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_4], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf83, arg358_1, model_freqs_cis, buf88, 4096L, stream, kernels, this->cubin_dir_);
    auto buf89 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf73, 3, int_array_7, int_array_8, 0L)); buf73.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_4], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf88, model_layers_4_attention_kv_cache_0_k_cache, buf89, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf93 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf89, 4, int_array_9, int_array_10, 0L)); buf89.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_4], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf93, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf94 = std::move(buf74);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_4], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf93, model_layers_4_attention_kv_cache_0_v_cache, buf94, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf95 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf88, 3, int_array_13, int_array_14, 0L)); buf88.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_4], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf94, buf95, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf96 = std::move(buf83);  // reuse
    // Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf95, model_layers_4_attention_wo_weight, buf96, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf98 = std::move(buf62);  // reuse
    // Topologically Sorted Source Nodes: [mul_43, add_15, mul_52, add_18, rms_norm_9], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf77, buf81, buf96, model_layers_4_ffn_norm_weight, buf98, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf99 = std::move(buf80);  // reuse
    auto buf100 = std::move(buf79);  // reuse
    // Topologically Sorted Source Nodes: [linear_32, linear_33], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf98, model_layers_4_feed_forward_w1_weight, model_layers_4_feed_forward_w3_weight, buf99, buf100, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf101 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf95, 2, int_array_1, int_array_2, 0L)); buf95.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf99, buf100, model_layers_4_feed_forward_w2_weight, buf101, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf103 = std::move(buf98);  // reuse
    // Topologically Sorted Source Nodes: [mul_43, add_15, mul_52, add_18, mul_54, add_19, rms_norm_10], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf77, buf81, buf96, buf101, model_layers_5_attention_norm_weight, buf103, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf104 = std::move(buf40);  // reuse
    // Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf103, model_layers_5_attention_wq_weight, buf104, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf105 = std::move(buf84);  // reuse
    // Topologically Sorted Source Nodes: [linear_36, linear_37, index_put__11], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf103, model_layers_5_attention_wk_weight, model_layers_5_attention_wv_weight, arg358_1, buf105, model_layers_5_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__10], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf105, model_freqs_cis, model_layers_5_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf109 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf63, 4, int_array_5, int_array_6, 0L)); buf63.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_5], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf104, arg358_1, model_freqs_cis, buf109, 4096L, stream, kernels, this->cubin_dir_);
    auto buf110 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf93, 3, int_array_7, int_array_8, 0L)); buf93.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_5], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf109, model_layers_5_attention_kv_cache_0_k_cache, buf110, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf114 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf110, 4, int_array_9, int_array_10, 0L)); buf110.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_5], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf114, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf115 = std::move(buf94);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_5], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf114, model_layers_5_attention_kv_cache_0_v_cache, buf115, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf116 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf109, 3, int_array_13, int_array_14, 0L)); buf109.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_5], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf115, buf116, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf118 = std::move(buf77);  // reuse
    // Topologically Sorted Source Nodes: [mul_43, add_15, mul_52, add_18, mul_54, add_19, linear_38, mul_63, add_22], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf118, buf116, model_layers_5_attention_wo_weight, buf81, buf96, buf101, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf119 = std::move(buf82);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_11], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf118, buf119, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf120 = std::move(buf99);  // reuse
    auto buf121 = std::move(buf100);  // reuse
    // Topologically Sorted Source Nodes: [linear_39, linear_40], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf118, buf119, model_layers_5_ffn_norm_weight, model_layers_5_feed_forward_w1_weight, model_layers_5_feed_forward_w3_weight, buf120, buf121, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf122 = std::move(buf96);  // reuse
    // Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf120, buf121, model_layers_5_feed_forward_w2_weight, buf122, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf123 = std::move(buf119);  // reuse
    // Topologically Sorted Source Nodes: [mul_65, add_23, rms_norm_12], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf118, buf122, buf123, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf124 = std::move(buf81);  // reuse
    // Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf118, buf122, buf123, model_layers_6_attention_norm_weight, model_layers_6_attention_wq_weight, buf124, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf125 = std::move(buf105);  // reuse
    // Topologically Sorted Source Nodes: [linear_43, linear_44, index_put__13], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf118, buf122, buf123, model_layers_6_attention_norm_weight, model_layers_6_attention_wk_weight, model_layers_6_attention_wv_weight, arg358_1, buf125, model_layers_6_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__12], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf125, model_freqs_cis, model_layers_6_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf129 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf116, 4, int_array_5, int_array_6, 0L)); buf116.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_6], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf124, arg358_1, model_freqs_cis, buf129, 4096L, stream, kernels, this->cubin_dir_);
    auto buf130 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf114, 3, int_array_7, int_array_8, 0L)); buf114.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_6], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf129, model_layers_6_attention_kv_cache_0_k_cache, buf130, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf134 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf130, 4, int_array_9, int_array_10, 0L)); buf130.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_6], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf134, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf135 = std::move(buf115);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_6], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf134, model_layers_6_attention_kv_cache_0_v_cache, buf135, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf136 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf129, 3, int_array_13, int_array_14, 0L)); buf129.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_6], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf135, buf136, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf137 = std::move(buf124);  // reuse
    // Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf136, model_layers_6_attention_wo_weight, buf137, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf139 = std::move(buf103);  // reuse
    // Topologically Sorted Source Nodes: [mul_65, add_23, mul_74, add_26, rms_norm_13], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf118, buf122, buf137, model_layers_6_ffn_norm_weight, buf139, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf140 = std::move(buf121);  // reuse
    auto buf141 = std::move(buf120);  // reuse
    // Topologically Sorted Source Nodes: [linear_46, linear_47], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf139, model_layers_6_feed_forward_w1_weight, model_layers_6_feed_forward_w3_weight, buf140, buf141, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf142 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf136, 2, int_array_1, int_array_2, 0L)); buf136.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf140, buf141, model_layers_6_feed_forward_w2_weight, buf142, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf144 = std::move(buf139);  // reuse
    // Topologically Sorted Source Nodes: [mul_65, add_23, mul_74, add_26, mul_76, add_27, rms_norm_14], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf118, buf122, buf137, buf142, model_layers_7_attention_norm_weight, buf144, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf145 = std::move(buf101);  // reuse
    // Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf144, model_layers_7_attention_wq_weight, buf145, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf146 = std::move(buf125);  // reuse
    // Topologically Sorted Source Nodes: [linear_50, linear_51, index_put__15], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf144, model_layers_7_attention_wk_weight, model_layers_7_attention_wv_weight, arg358_1, buf146, model_layers_7_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__14], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf146, model_freqs_cis, model_layers_7_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf150 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf104, 4, int_array_5, int_array_6, 0L)); buf104.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_7], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf145, arg358_1, model_freqs_cis, buf150, 4096L, stream, kernels, this->cubin_dir_);
    auto buf151 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf134, 3, int_array_7, int_array_8, 0L)); buf134.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_7], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf150, model_layers_7_attention_kv_cache_0_k_cache, buf151, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf155 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf151, 4, int_array_9, int_array_10, 0L)); buf151.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_7], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf155, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf156 = std::move(buf135);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_7], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf155, model_layers_7_attention_kv_cache_0_v_cache, buf156, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf157 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf150, 3, int_array_13, int_array_14, 0L)); buf150.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_7], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf156, buf157, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf159 = std::move(buf118);  // reuse
    // Topologically Sorted Source Nodes: [mul_65, add_23, mul_74, add_26, mul_76, add_27, linear_52, mul_85, add_30], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf159, buf157, model_layers_7_attention_wo_weight, buf122, buf137, buf142, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf160 = std::move(buf123);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_15], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf159, buf160, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf161 = std::move(buf141);  // reuse
    auto buf162 = std::move(buf140);  // reuse
    // Topologically Sorted Source Nodes: [linear_53, linear_54], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf159, buf160, model_layers_7_ffn_norm_weight, model_layers_7_feed_forward_w1_weight, model_layers_7_feed_forward_w3_weight, buf161, buf162, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf163 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf157, 2, int_array_1, int_array_2, 0L)); buf157.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf161, buf162, model_layers_7_feed_forward_w2_weight, buf163, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf164 = std::move(buf160);  // reuse
    // Topologically Sorted Source Nodes: [mul_87, add_31, rms_norm_16], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf159, buf163, buf164, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf165 = std::move(buf142);  // reuse
    // Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf159, buf163, buf164, model_layers_8_attention_norm_weight, model_layers_8_attention_wq_weight, buf165, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf166 = std::move(buf146);  // reuse
    // Topologically Sorted Source Nodes: [linear_57, linear_58, index_put__17], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf159, buf163, buf164, model_layers_8_attention_norm_weight, model_layers_8_attention_wk_weight, model_layers_8_attention_wv_weight, arg358_1, buf166, model_layers_8_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__16], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf166, model_freqs_cis, model_layers_8_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf170 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf137, 4, int_array_5, int_array_6, 0L)); buf137.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_8], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf165, arg358_1, model_freqs_cis, buf170, 4096L, stream, kernels, this->cubin_dir_);
    auto buf171 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf155, 3, int_array_7, int_array_8, 0L)); buf155.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_8], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf170, model_layers_8_attention_kv_cache_0_k_cache, buf171, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf175 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf171, 4, int_array_9, int_array_10, 0L)); buf171.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_8], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf175, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf176 = std::move(buf156);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_8], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf175, model_layers_8_attention_kv_cache_0_v_cache, buf176, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf177 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf170, 3, int_array_13, int_array_14, 0L)); buf170.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_8], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf176, buf177, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf178 = std::move(buf165);  // reuse
    // Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf177, model_layers_8_attention_wo_weight, buf178, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf180 = std::move(buf144);  // reuse
    // Topologically Sorted Source Nodes: [mul_87, add_31, mul_96, add_34, rms_norm_17], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf159, buf163, buf178, model_layers_8_ffn_norm_weight, buf180, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf181 = std::move(buf162);  // reuse
    auto buf182 = std::move(buf161);  // reuse
    // Topologically Sorted Source Nodes: [linear_60, linear_61], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf180, model_layers_8_feed_forward_w1_weight, model_layers_8_feed_forward_w3_weight, buf181, buf182, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf183 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf177, 2, int_array_1, int_array_2, 0L)); buf177.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf181, buf182, model_layers_8_feed_forward_w2_weight, buf183, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf185 = std::move(buf180);  // reuse
    // Topologically Sorted Source Nodes: [mul_87, add_31, mul_96, add_34, mul_98, add_35, rms_norm_18], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf159, buf163, buf178, buf183, model_layers_9_attention_norm_weight, buf185, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf186 = std::move(buf122);  // reuse
    // Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf185, model_layers_9_attention_wq_weight, buf186, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf187 = std::move(buf166);  // reuse
    // Topologically Sorted Source Nodes: [linear_64, linear_65, index_put__19], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf185, model_layers_9_attention_wk_weight, model_layers_9_attention_wv_weight, arg358_1, buf187, model_layers_9_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__18], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf187, model_freqs_cis, model_layers_9_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf191 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf145, 4, int_array_5, int_array_6, 0L)); buf145.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_9], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf186, arg358_1, model_freqs_cis, buf191, 4096L, stream, kernels, this->cubin_dir_);
    auto buf192 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf175, 3, int_array_7, int_array_8, 0L)); buf175.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_9], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf191, model_layers_9_attention_kv_cache_0_k_cache, buf192, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf196 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf192, 4, int_array_9, int_array_10, 0L)); buf192.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_9], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf196, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf197 = std::move(buf176);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_9], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf196, model_layers_9_attention_kv_cache_0_v_cache, buf197, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf198 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf191, 3, int_array_13, int_array_14, 0L)); buf191.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_9], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf197, buf198, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf200 = std::move(buf159);  // reuse
    // Topologically Sorted Source Nodes: [mul_87, add_31, mul_96, add_34, mul_98, add_35, linear_66, mul_107, add_38], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf200, buf198, model_layers_9_attention_wo_weight, buf163, buf178, buf183, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf201 = std::move(buf164);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_19], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf200, buf201, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf202 = std::move(buf182);  // reuse
    auto buf203 = std::move(buf181);  // reuse
    // Topologically Sorted Source Nodes: [linear_67, linear_68], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf200, buf201, model_layers_9_ffn_norm_weight, model_layers_9_feed_forward_w1_weight, model_layers_9_feed_forward_w3_weight, buf202, buf203, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf204 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf198, 2, int_array_1, int_array_2, 0L)); buf198.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf202, buf203, model_layers_9_feed_forward_w2_weight, buf204, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf205 = std::move(buf201);  // reuse
    // Topologically Sorted Source Nodes: [mul_109, add_39, rms_norm_20], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf200, buf204, buf205, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf206 = std::move(buf183);  // reuse
    // Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf200, buf204, buf205, model_layers_10_attention_norm_weight, model_layers_10_attention_wq_weight, buf206, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf207 = std::move(buf187);  // reuse
    // Topologically Sorted Source Nodes: [linear_71, linear_72, index_put__21], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf200, buf204, buf205, model_layers_10_attention_norm_weight, model_layers_10_attention_wk_weight, model_layers_10_attention_wv_weight, arg358_1, buf207, model_layers_10_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__20], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf207, model_freqs_cis, model_layers_10_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf211 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf178, 4, int_array_5, int_array_6, 0L)); buf178.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_10], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf206, arg358_1, model_freqs_cis, buf211, 4096L, stream, kernels, this->cubin_dir_);
    auto buf212 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf196, 3, int_array_7, int_array_8, 0L)); buf196.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_10], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf211, model_layers_10_attention_kv_cache_0_k_cache, buf212, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf216 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf212, 4, int_array_9, int_array_10, 0L)); buf212.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_10], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf216, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf217 = std::move(buf197);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_10], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf216, model_layers_10_attention_kv_cache_0_v_cache, buf217, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf218 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf211, 3, int_array_13, int_array_14, 0L)); buf211.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_10], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf217, buf218, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf219 = std::move(buf206);  // reuse
    // Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf218, model_layers_10_attention_wo_weight, buf219, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf221 = std::move(buf185);  // reuse
    // Topologically Sorted Source Nodes: [mul_109, add_39, mul_118, add_42, rms_norm_21], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf200, buf204, buf219, model_layers_10_ffn_norm_weight, buf221, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf222 = std::move(buf203);  // reuse
    auto buf223 = std::move(buf202);  // reuse
    // Topologically Sorted Source Nodes: [linear_74, linear_75], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf221, model_layers_10_feed_forward_w1_weight, model_layers_10_feed_forward_w3_weight, buf222, buf223, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf224 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf218, 2, int_array_1, int_array_2, 0L)); buf218.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf222, buf223, model_layers_10_feed_forward_w2_weight, buf224, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf226 = std::move(buf221);  // reuse
    // Topologically Sorted Source Nodes: [mul_109, add_39, mul_118, add_42, mul_120, add_43, rms_norm_22], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf200, buf204, buf219, buf224, model_layers_11_attention_norm_weight, buf226, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf227 = std::move(buf163);  // reuse
    // Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf226, model_layers_11_attention_wq_weight, buf227, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf228 = std::move(buf207);  // reuse
    // Topologically Sorted Source Nodes: [linear_78, linear_79, index_put__23], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf226, model_layers_11_attention_wk_weight, model_layers_11_attention_wv_weight, arg358_1, buf228, model_layers_11_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__22], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf228, model_freqs_cis, model_layers_11_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf232 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf186, 4, int_array_5, int_array_6, 0L)); buf186.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_11], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf227, arg358_1, model_freqs_cis, buf232, 4096L, stream, kernels, this->cubin_dir_);
    auto buf233 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf216, 3, int_array_7, int_array_8, 0L)); buf216.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_11], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf232, model_layers_11_attention_kv_cache_0_k_cache, buf233, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf237 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf233, 4, int_array_9, int_array_10, 0L)); buf233.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_11], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf237, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf238 = std::move(buf217);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_11], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf237, model_layers_11_attention_kv_cache_0_v_cache, buf238, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf239 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf232, 3, int_array_13, int_array_14, 0L)); buf232.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_11], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf238, buf239, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf241 = std::move(buf200);  // reuse
    // Topologically Sorted Source Nodes: [mul_109, add_39, mul_118, add_42, mul_120, add_43, linear_80, mul_129, add_46], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf241, buf239, model_layers_11_attention_wo_weight, buf204, buf219, buf224, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf242 = std::move(buf205);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_23], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf241, buf242, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf243 = std::move(buf223);  // reuse
    auto buf244 = std::move(buf222);  // reuse
    // Topologically Sorted Source Nodes: [linear_81, linear_82], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf241, buf242, model_layers_11_ffn_norm_weight, model_layers_11_feed_forward_w1_weight, model_layers_11_feed_forward_w3_weight, buf243, buf244, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf245 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf239, 2, int_array_1, int_array_2, 0L)); buf239.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_83], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf243, buf244, model_layers_11_feed_forward_w2_weight, buf245, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf246 = std::move(buf242);  // reuse
    // Topologically Sorted Source Nodes: [mul_131, add_47, rms_norm_24], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf241, buf245, buf246, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf247 = std::move(buf224);  // reuse
    // Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf241, buf245, buf246, model_layers_12_attention_norm_weight, model_layers_12_attention_wq_weight, buf247, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf248 = std::move(buf228);  // reuse
    // Topologically Sorted Source Nodes: [linear_85, linear_86, index_put__25], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf241, buf245, buf246, model_layers_12_attention_norm_weight, model_layers_12_attention_wk_weight, model_layers_12_attention_wv_weight, arg358_1, buf248, model_layers_12_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__24], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf248, model_freqs_cis, model_layers_12_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf252 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf219, 4, int_array_5, int_array_6, 0L)); buf219.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_12], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf247, arg358_1, model_freqs_cis, buf252, 4096L, stream, kernels, this->cubin_dir_);
    auto buf253 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf237, 3, int_array_7, int_array_8, 0L)); buf237.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_12], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf252, model_layers_12_attention_kv_cache_0_k_cache, buf253, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf257 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf253, 4, int_array_9, int_array_10, 0L)); buf253.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_12], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf257, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf258 = std::move(buf238);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_12], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf257, model_layers_12_attention_kv_cache_0_v_cache, buf258, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf259 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf252, 3, int_array_13, int_array_14, 0L)); buf252.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_12], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf258, buf259, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf260 = std::move(buf247);  // reuse
    // Topologically Sorted Source Nodes: [linear_87], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf259, model_layers_12_attention_wo_weight, buf260, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf262 = std::move(buf226);  // reuse
    // Topologically Sorted Source Nodes: [mul_131, add_47, mul_140, add_50, rms_norm_25], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf241, buf245, buf260, model_layers_12_ffn_norm_weight, buf262, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf263 = std::move(buf244);  // reuse
    auto buf264 = std::move(buf243);  // reuse
    // Topologically Sorted Source Nodes: [linear_88, linear_89], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf262, model_layers_12_feed_forward_w1_weight, model_layers_12_feed_forward_w3_weight, buf263, buf264, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf265 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf259, 2, int_array_1, int_array_2, 0L)); buf259.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_90], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf263, buf264, model_layers_12_feed_forward_w2_weight, buf265, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf267 = std::move(buf262);  // reuse
    // Topologically Sorted Source Nodes: [mul_131, add_47, mul_140, add_50, mul_142, add_51, rms_norm_26], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf241, buf245, buf260, buf265, model_layers_13_attention_norm_weight, buf267, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf268 = std::move(buf204);  // reuse
    // Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf267, model_layers_13_attention_wq_weight, buf268, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf269 = std::move(buf248);  // reuse
    // Topologically Sorted Source Nodes: [linear_92, linear_93, index_put__27], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf267, model_layers_13_attention_wk_weight, model_layers_13_attention_wv_weight, arg358_1, buf269, model_layers_13_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__26], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf269, model_freqs_cis, model_layers_13_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf273 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf227, 4, int_array_5, int_array_6, 0L)); buf227.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_13], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf268, arg358_1, model_freqs_cis, buf273, 4096L, stream, kernels, this->cubin_dir_);
    auto buf274 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf257, 3, int_array_7, int_array_8, 0L)); buf257.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_13], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf273, model_layers_13_attention_kv_cache_0_k_cache, buf274, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf278 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf274, 4, int_array_9, int_array_10, 0L)); buf274.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_13], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf278, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf279 = std::move(buf258);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_13], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf278, model_layers_13_attention_kv_cache_0_v_cache, buf279, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf280 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf273, 3, int_array_13, int_array_14, 0L)); buf273.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_13], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf279, buf280, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf282 = std::move(buf241);  // reuse
    // Topologically Sorted Source Nodes: [mul_131, add_47, mul_140, add_50, mul_142, add_51, linear_94, mul_151, add_54], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf282, buf280, model_layers_13_attention_wo_weight, buf245, buf260, buf265, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf283 = std::move(buf246);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_27], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf282, buf283, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf284 = std::move(buf264);  // reuse
    auto buf285 = std::move(buf263);  // reuse
    // Topologically Sorted Source Nodes: [linear_95, linear_96], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf282, buf283, model_layers_13_ffn_norm_weight, model_layers_13_feed_forward_w1_weight, model_layers_13_feed_forward_w3_weight, buf284, buf285, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf286 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf280, 2, int_array_1, int_array_2, 0L)); buf280.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf284, buf285, model_layers_13_feed_forward_w2_weight, buf286, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf287 = std::move(buf283);  // reuse
    // Topologically Sorted Source Nodes: [mul_153, add_55, rms_norm_28], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf282, buf286, buf287, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf288 = std::move(buf265);  // reuse
    // Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf282, buf286, buf287, model_layers_14_attention_norm_weight, model_layers_14_attention_wq_weight, buf288, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf289 = std::move(buf269);  // reuse
    // Topologically Sorted Source Nodes: [linear_99, linear_100, index_put__29], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf282, buf286, buf287, model_layers_14_attention_norm_weight, model_layers_14_attention_wk_weight, model_layers_14_attention_wv_weight, arg358_1, buf289, model_layers_14_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__28], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf289, model_freqs_cis, model_layers_14_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf293 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf260, 4, int_array_5, int_array_6, 0L)); buf260.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_14], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf288, arg358_1, model_freqs_cis, buf293, 4096L, stream, kernels, this->cubin_dir_);
    auto buf294 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf278, 3, int_array_7, int_array_8, 0L)); buf278.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_14], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf293, model_layers_14_attention_kv_cache_0_k_cache, buf294, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf298 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf294, 4, int_array_9, int_array_10, 0L)); buf294.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_14], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf298, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf299 = std::move(buf279);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_14], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf298, model_layers_14_attention_kv_cache_0_v_cache, buf299, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf300 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf293, 3, int_array_13, int_array_14, 0L)); buf293.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_14], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf299, buf300, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf301 = std::move(buf288);  // reuse
    // Topologically Sorted Source Nodes: [linear_101], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf300, model_layers_14_attention_wo_weight, buf301, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf303 = std::move(buf267);  // reuse
    // Topologically Sorted Source Nodes: [mul_153, add_55, mul_162, add_58, rms_norm_29], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf282, buf286, buf301, model_layers_14_ffn_norm_weight, buf303, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf304 = std::move(buf285);  // reuse
    auto buf305 = std::move(buf284);  // reuse
    // Topologically Sorted Source Nodes: [linear_102, linear_103], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf303, model_layers_14_feed_forward_w1_weight, model_layers_14_feed_forward_w3_weight, buf304, buf305, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf306 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf300, 2, int_array_1, int_array_2, 0L)); buf300.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf304, buf305, model_layers_14_feed_forward_w2_weight, buf306, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf308 = std::move(buf303);  // reuse
    // Topologically Sorted Source Nodes: [mul_153, add_55, mul_162, add_58, mul_164, add_59, rms_norm_30], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf282, buf286, buf301, buf306, model_layers_15_attention_norm_weight, buf308, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf309 = std::move(buf245);  // reuse
    // Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf308, model_layers_15_attention_wq_weight, buf309, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf310 = std::move(buf289);  // reuse
    // Topologically Sorted Source Nodes: [linear_106, linear_107, index_put__31], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf308, model_layers_15_attention_wk_weight, model_layers_15_attention_wv_weight, arg358_1, buf310, model_layers_15_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__30], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf310, model_freqs_cis, model_layers_15_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf314 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf268, 4, int_array_5, int_array_6, 0L)); buf268.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_15], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf309, arg358_1, model_freqs_cis, buf314, 4096L, stream, kernels, this->cubin_dir_);
    auto buf315 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf298, 3, int_array_7, int_array_8, 0L)); buf298.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_15], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf314, model_layers_15_attention_kv_cache_0_k_cache, buf315, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf319 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf315, 4, int_array_9, int_array_10, 0L)); buf315.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_15], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf319, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf320 = std::move(buf299);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_15], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf319, model_layers_15_attention_kv_cache_0_v_cache, buf320, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf321 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf314, 3, int_array_13, int_array_14, 0L)); buf314.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_15], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf320, buf321, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf323 = std::move(buf282);  // reuse
    // Topologically Sorted Source Nodes: [mul_153, add_55, mul_162, add_58, mul_164, add_59, linear_108, mul_173, add_62], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf323, buf321, model_layers_15_attention_wo_weight, buf286, buf301, buf306, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf324 = std::move(buf287);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_31], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf323, buf324, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf325 = std::move(buf305);  // reuse
    auto buf326 = std::move(buf304);  // reuse
    // Topologically Sorted Source Nodes: [linear_109, linear_110], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf323, buf324, model_layers_15_ffn_norm_weight, model_layers_15_feed_forward_w1_weight, model_layers_15_feed_forward_w3_weight, buf325, buf326, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf327 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf321, 2, int_array_1, int_array_2, 0L)); buf321.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_111], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf325, buf326, model_layers_15_feed_forward_w2_weight, buf327, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf328 = std::move(buf324);  // reuse
    // Topologically Sorted Source Nodes: [mul_175, add_63, rms_norm_32], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf323, buf327, buf328, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf329 = std::move(buf306);  // reuse
    // Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf323, buf327, buf328, model_layers_16_attention_norm_weight, model_layers_16_attention_wq_weight, buf329, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf330 = std::move(buf310);  // reuse
    // Topologically Sorted Source Nodes: [linear_113, linear_114, index_put__33], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf323, buf327, buf328, model_layers_16_attention_norm_weight, model_layers_16_attention_wk_weight, model_layers_16_attention_wv_weight, arg358_1, buf330, model_layers_16_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__32], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf330, model_freqs_cis, model_layers_16_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf334 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf301, 4, int_array_5, int_array_6, 0L)); buf301.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_16], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf329, arg358_1, model_freqs_cis, buf334, 4096L, stream, kernels, this->cubin_dir_);
    auto buf335 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf319, 3, int_array_7, int_array_8, 0L)); buf319.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_16], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf334, model_layers_16_attention_kv_cache_0_k_cache, buf335, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf339 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf335, 4, int_array_9, int_array_10, 0L)); buf335.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_16], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf339, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf340 = std::move(buf320);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_16], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf339, model_layers_16_attention_kv_cache_0_v_cache, buf340, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf341 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf334, 3, int_array_13, int_array_14, 0L)); buf334.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_16], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf340, buf341, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf342 = std::move(buf329);  // reuse
    // Topologically Sorted Source Nodes: [linear_115], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf341, model_layers_16_attention_wo_weight, buf342, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf344 = std::move(buf308);  // reuse
    // Topologically Sorted Source Nodes: [mul_175, add_63, mul_184, add_66, rms_norm_33], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf323, buf327, buf342, model_layers_16_ffn_norm_weight, buf344, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf345 = std::move(buf326);  // reuse
    auto buf346 = std::move(buf325);  // reuse
    // Topologically Sorted Source Nodes: [linear_116, linear_117], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf344, model_layers_16_feed_forward_w1_weight, model_layers_16_feed_forward_w3_weight, buf345, buf346, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf347 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf341, 2, int_array_1, int_array_2, 0L)); buf341.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_118], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf345, buf346, model_layers_16_feed_forward_w2_weight, buf347, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf349 = std::move(buf344);  // reuse
    // Topologically Sorted Source Nodes: [mul_175, add_63, mul_184, add_66, mul_186, add_67, rms_norm_34], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf323, buf327, buf342, buf347, model_layers_17_attention_norm_weight, buf349, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf350 = std::move(buf286);  // reuse
    // Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf349, model_layers_17_attention_wq_weight, buf350, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf351 = std::move(buf330);  // reuse
    // Topologically Sorted Source Nodes: [linear_120, linear_121, index_put__35], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf349, model_layers_17_attention_wk_weight, model_layers_17_attention_wv_weight, arg358_1, buf351, model_layers_17_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__34], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf351, model_freqs_cis, model_layers_17_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf355 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf309, 4, int_array_5, int_array_6, 0L)); buf309.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_17], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf350, arg358_1, model_freqs_cis, buf355, 4096L, stream, kernels, this->cubin_dir_);
    auto buf356 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf339, 3, int_array_7, int_array_8, 0L)); buf339.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_17], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf355, model_layers_17_attention_kv_cache_0_k_cache, buf356, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf360 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf356, 4, int_array_9, int_array_10, 0L)); buf356.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_17], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf360, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf361 = std::move(buf340);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_17], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf360, model_layers_17_attention_kv_cache_0_v_cache, buf361, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf362 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf355, 3, int_array_13, int_array_14, 0L)); buf355.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_17], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf361, buf362, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf364 = std::move(buf323);  // reuse
    // Topologically Sorted Source Nodes: [mul_175, add_63, mul_184, add_66, mul_186, add_67, linear_122, mul_195, add_70], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf364, buf362, model_layers_17_attention_wo_weight, buf327, buf342, buf347, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf365 = std::move(buf328);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_35], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf364, buf365, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf366 = std::move(buf346);  // reuse
    auto buf367 = std::move(buf345);  // reuse
    // Topologically Sorted Source Nodes: [linear_123, linear_124], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf364, buf365, model_layers_17_ffn_norm_weight, model_layers_17_feed_forward_w1_weight, model_layers_17_feed_forward_w3_weight, buf366, buf367, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf368 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf362, 2, int_array_1, int_array_2, 0L)); buf362.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_125], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf366, buf367, model_layers_17_feed_forward_w2_weight, buf368, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf369 = std::move(buf365);  // reuse
    // Topologically Sorted Source Nodes: [mul_197, add_71, rms_norm_36], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf364, buf368, buf369, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf370 = std::move(buf347);  // reuse
    // Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf364, buf368, buf369, model_layers_18_attention_norm_weight, model_layers_18_attention_wq_weight, buf370, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf371 = std::move(buf351);  // reuse
    // Topologically Sorted Source Nodes: [linear_127, linear_128, index_put__37], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf364, buf368, buf369, model_layers_18_attention_norm_weight, model_layers_18_attention_wk_weight, model_layers_18_attention_wv_weight, arg358_1, buf371, model_layers_18_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__36], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf371, model_freqs_cis, model_layers_18_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf375 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf342, 4, int_array_5, int_array_6, 0L)); buf342.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_18], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf370, arg358_1, model_freqs_cis, buf375, 4096L, stream, kernels, this->cubin_dir_);
    auto buf376 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf360, 3, int_array_7, int_array_8, 0L)); buf360.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_18], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf375, model_layers_18_attention_kv_cache_0_k_cache, buf376, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf380 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf376, 4, int_array_9, int_array_10, 0L)); buf376.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_18], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf380, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf381 = std::move(buf361);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_18], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf380, model_layers_18_attention_kv_cache_0_v_cache, buf381, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf382 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf375, 3, int_array_13, int_array_14, 0L)); buf375.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_18], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf381, buf382, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf383 = std::move(buf370);  // reuse
    // Topologically Sorted Source Nodes: [linear_129], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf382, model_layers_18_attention_wo_weight, buf383, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf385 = std::move(buf349);  // reuse
    // Topologically Sorted Source Nodes: [mul_197, add_71, mul_206, add_74, rms_norm_37], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf364, buf368, buf383, model_layers_18_ffn_norm_weight, buf385, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf386 = std::move(buf367);  // reuse
    auto buf387 = std::move(buf366);  // reuse
    // Topologically Sorted Source Nodes: [linear_130, linear_131], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf385, model_layers_18_feed_forward_w1_weight, model_layers_18_feed_forward_w3_weight, buf386, buf387, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf388 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf382, 2, int_array_1, int_array_2, 0L)); buf382.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_132], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf386, buf387, model_layers_18_feed_forward_w2_weight, buf388, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf390 = std::move(buf385);  // reuse
    // Topologically Sorted Source Nodes: [mul_197, add_71, mul_206, add_74, mul_208, add_75, rms_norm_38], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf364, buf368, buf383, buf388, model_layers_19_attention_norm_weight, buf390, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf391 = std::move(buf327);  // reuse
    // Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf390, model_layers_19_attention_wq_weight, buf391, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf392 = std::move(buf371);  // reuse
    // Topologically Sorted Source Nodes: [linear_134, linear_135, index_put__39], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf390, model_layers_19_attention_wk_weight, model_layers_19_attention_wv_weight, arg358_1, buf392, model_layers_19_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__38], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf392, model_freqs_cis, model_layers_19_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf396 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf350, 4, int_array_5, int_array_6, 0L)); buf350.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_19], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf391, arg358_1, model_freqs_cis, buf396, 4096L, stream, kernels, this->cubin_dir_);
    auto buf397 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf380, 3, int_array_7, int_array_8, 0L)); buf380.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_19], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf396, model_layers_19_attention_kv_cache_0_k_cache, buf397, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf401 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf397, 4, int_array_9, int_array_10, 0L)); buf397.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_19], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf401, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf402 = std::move(buf381);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_19], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf401, model_layers_19_attention_kv_cache_0_v_cache, buf402, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf403 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf396, 3, int_array_13, int_array_14, 0L)); buf396.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_19], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf402, buf403, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf405 = std::move(buf364);  // reuse
    // Topologically Sorted Source Nodes: [mul_197, add_71, mul_206, add_74, mul_208, add_75, linear_136, mul_217, add_78], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf405, buf403, model_layers_19_attention_wo_weight, buf368, buf383, buf388, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf406 = std::move(buf369);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_39], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf405, buf406, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf407 = std::move(buf387);  // reuse
    auto buf408 = std::move(buf386);  // reuse
    // Topologically Sorted Source Nodes: [linear_137, linear_138], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf405, buf406, model_layers_19_ffn_norm_weight, model_layers_19_feed_forward_w1_weight, model_layers_19_feed_forward_w3_weight, buf407, buf408, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf409 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf403, 2, int_array_1, int_array_2, 0L)); buf403.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_139], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf407, buf408, model_layers_19_feed_forward_w2_weight, buf409, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf410 = std::move(buf406);  // reuse
    // Topologically Sorted Source Nodes: [mul_219, add_79, rms_norm_40], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf405, buf409, buf410, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf411 = std::move(buf388);  // reuse
    // Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf405, buf409, buf410, model_layers_20_attention_norm_weight, model_layers_20_attention_wq_weight, buf411, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf412 = std::move(buf392);  // reuse
    // Topologically Sorted Source Nodes: [linear_141, linear_142, index_put__41], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf405, buf409, buf410, model_layers_20_attention_norm_weight, model_layers_20_attention_wk_weight, model_layers_20_attention_wv_weight, arg358_1, buf412, model_layers_20_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__40], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf412, model_freqs_cis, model_layers_20_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf416 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf383, 4, int_array_5, int_array_6, 0L)); buf383.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_20], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf411, arg358_1, model_freqs_cis, buf416, 4096L, stream, kernels, this->cubin_dir_);
    auto buf417 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf401, 3, int_array_7, int_array_8, 0L)); buf401.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_20], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf416, model_layers_20_attention_kv_cache_0_k_cache, buf417, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf421 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf417, 4, int_array_9, int_array_10, 0L)); buf417.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_20], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf421, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf422 = std::move(buf402);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_20], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf421, model_layers_20_attention_kv_cache_0_v_cache, buf422, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf423 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf416, 3, int_array_13, int_array_14, 0L)); buf416.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_20], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf422, buf423, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf424 = std::move(buf411);  // reuse
    // Topologically Sorted Source Nodes: [linear_143], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf423, model_layers_20_attention_wo_weight, buf424, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf426 = std::move(buf390);  // reuse
    // Topologically Sorted Source Nodes: [mul_219, add_79, mul_228, add_82, rms_norm_41], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf405, buf409, buf424, model_layers_20_ffn_norm_weight, buf426, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf427 = std::move(buf408);  // reuse
    auto buf428 = std::move(buf407);  // reuse
    // Topologically Sorted Source Nodes: [linear_144, linear_145], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf426, model_layers_20_feed_forward_w1_weight, model_layers_20_feed_forward_w3_weight, buf427, buf428, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf429 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf423, 2, int_array_1, int_array_2, 0L)); buf423.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_146], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf427, buf428, model_layers_20_feed_forward_w2_weight, buf429, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf431 = std::move(buf426);  // reuse
    // Topologically Sorted Source Nodes: [mul_219, add_79, mul_228, add_82, mul_230, add_83, rms_norm_42], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf405, buf409, buf424, buf429, model_layers_21_attention_norm_weight, buf431, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf432 = std::move(buf368);  // reuse
    // Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf431, model_layers_21_attention_wq_weight, buf432, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf433 = std::move(buf412);  // reuse
    // Topologically Sorted Source Nodes: [linear_148, linear_149, index_put__43], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf431, model_layers_21_attention_wk_weight, model_layers_21_attention_wv_weight, arg358_1, buf433, model_layers_21_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__42], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf433, model_freqs_cis, model_layers_21_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf437 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf391, 4, int_array_5, int_array_6, 0L)); buf391.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_21], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf432, arg358_1, model_freqs_cis, buf437, 4096L, stream, kernels, this->cubin_dir_);
    auto buf438 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf421, 3, int_array_7, int_array_8, 0L)); buf421.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_21], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf437, model_layers_21_attention_kv_cache_0_k_cache, buf438, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf442 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf438, 4, int_array_9, int_array_10, 0L)); buf438.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_21], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf442, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf443 = std::move(buf422);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_21], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf442, model_layers_21_attention_kv_cache_0_v_cache, buf443, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf444 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf437, 3, int_array_13, int_array_14, 0L)); buf437.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_21], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf443, buf444, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf446 = std::move(buf405);  // reuse
    // Topologically Sorted Source Nodes: [mul_219, add_79, mul_228, add_82, mul_230, add_83, linear_150, mul_239, add_86], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf446, buf444, model_layers_21_attention_wo_weight, buf409, buf424, buf429, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf447 = std::move(buf410);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_43], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf446, buf447, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf448 = std::move(buf428);  // reuse
    auto buf449 = std::move(buf427);  // reuse
    // Topologically Sorted Source Nodes: [linear_151, linear_152], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf446, buf447, model_layers_21_ffn_norm_weight, model_layers_21_feed_forward_w1_weight, model_layers_21_feed_forward_w3_weight, buf448, buf449, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf450 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf444, 2, int_array_1, int_array_2, 0L)); buf444.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_153], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf448, buf449, model_layers_21_feed_forward_w2_weight, buf450, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf451 = std::move(buf447);  // reuse
    // Topologically Sorted Source Nodes: [mul_241, add_87, rms_norm_44], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf446, buf450, buf451, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf452 = std::move(buf429);  // reuse
    // Topologically Sorted Source Nodes: [linear_154], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf446, buf450, buf451, model_layers_22_attention_norm_weight, model_layers_22_attention_wq_weight, buf452, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf453 = std::move(buf433);  // reuse
    // Topologically Sorted Source Nodes: [linear_155, linear_156, index_put__45], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf446, buf450, buf451, model_layers_22_attention_norm_weight, model_layers_22_attention_wk_weight, model_layers_22_attention_wv_weight, arg358_1, buf453, model_layers_22_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__44], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf453, model_freqs_cis, model_layers_22_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf457 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf424, 4, int_array_5, int_array_6, 0L)); buf424.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_22], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf452, arg358_1, model_freqs_cis, buf457, 4096L, stream, kernels, this->cubin_dir_);
    auto buf458 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf442, 3, int_array_7, int_array_8, 0L)); buf442.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_22], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf457, model_layers_22_attention_kv_cache_0_k_cache, buf458, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf462 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf458, 4, int_array_9, int_array_10, 0L)); buf458.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_22], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf462, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf463 = std::move(buf443);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_22], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf462, model_layers_22_attention_kv_cache_0_v_cache, buf463, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf464 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf457, 3, int_array_13, int_array_14, 0L)); buf457.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_22], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf463, buf464, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf465 = std::move(buf452);  // reuse
    // Topologically Sorted Source Nodes: [linear_157], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf464, model_layers_22_attention_wo_weight, buf465, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf467 = std::move(buf431);  // reuse
    // Topologically Sorted Source Nodes: [mul_241, add_87, mul_250, add_90, rms_norm_45], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf446, buf450, buf465, model_layers_22_ffn_norm_weight, buf467, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf468 = std::move(buf449);  // reuse
    auto buf469 = std::move(buf448);  // reuse
    // Topologically Sorted Source Nodes: [linear_158, linear_159], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf467, model_layers_22_feed_forward_w1_weight, model_layers_22_feed_forward_w3_weight, buf468, buf469, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf470 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf464, 2, int_array_1, int_array_2, 0L)); buf464.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_160], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf468, buf469, model_layers_22_feed_forward_w2_weight, buf470, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf472 = std::move(buf467);  // reuse
    // Topologically Sorted Source Nodes: [mul_241, add_87, mul_250, add_90, mul_252, add_91, rms_norm_46], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf446, buf450, buf465, buf470, model_layers_23_attention_norm_weight, buf472, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf473 = std::move(buf409);  // reuse
    // Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf472, model_layers_23_attention_wq_weight, buf473, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf474 = std::move(buf453);  // reuse
    // Topologically Sorted Source Nodes: [linear_162, linear_163, index_put__47], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf472, model_layers_23_attention_wk_weight, model_layers_23_attention_wv_weight, arg358_1, buf474, model_layers_23_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__46], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf474, model_freqs_cis, model_layers_23_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf478 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf432, 4, int_array_5, int_array_6, 0L)); buf432.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_23], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf473, arg358_1, model_freqs_cis, buf478, 4096L, stream, kernels, this->cubin_dir_);
    auto buf479 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf462, 3, int_array_7, int_array_8, 0L)); buf462.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_23], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf478, model_layers_23_attention_kv_cache_0_k_cache, buf479, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf483 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf479, 4, int_array_9, int_array_10, 0L)); buf479.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_23], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf483, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf484 = std::move(buf463);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_23], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf483, model_layers_23_attention_kv_cache_0_v_cache, buf484, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf485 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf478, 3, int_array_13, int_array_14, 0L)); buf478.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_23], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf484, buf485, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf487 = std::move(buf446);  // reuse
    // Topologically Sorted Source Nodes: [mul_241, add_87, mul_250, add_90, mul_252, add_91, linear_164, mul_261, add_94], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf487, buf485, model_layers_23_attention_wo_weight, buf450, buf465, buf470, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf488 = std::move(buf451);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_47], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf487, buf488, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf489 = std::move(buf469);  // reuse
    auto buf490 = std::move(buf468);  // reuse
    // Topologically Sorted Source Nodes: [linear_165, linear_166], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf487, buf488, model_layers_23_ffn_norm_weight, model_layers_23_feed_forward_w1_weight, model_layers_23_feed_forward_w3_weight, buf489, buf490, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf491 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf485, 2, int_array_1, int_array_2, 0L)); buf485.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_167], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf489, buf490, model_layers_23_feed_forward_w2_weight, buf491, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf492 = std::move(buf488);  // reuse
    // Topologically Sorted Source Nodes: [mul_263, add_95, rms_norm_48], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf487, buf491, buf492, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf493 = std::move(buf470);  // reuse
    // Topologically Sorted Source Nodes: [linear_168], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf487, buf491, buf492, model_layers_24_attention_norm_weight, model_layers_24_attention_wq_weight, buf493, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf494 = std::move(buf474);  // reuse
    // Topologically Sorted Source Nodes: [linear_169, linear_170, index_put__49], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf487, buf491, buf492, model_layers_24_attention_norm_weight, model_layers_24_attention_wk_weight, model_layers_24_attention_wv_weight, arg358_1, buf494, model_layers_24_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__48], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf494, model_freqs_cis, model_layers_24_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf498 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf465, 4, int_array_5, int_array_6, 0L)); buf465.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_24], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf493, arg358_1, model_freqs_cis, buf498, 4096L, stream, kernels, this->cubin_dir_);
    auto buf499 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf483, 3, int_array_7, int_array_8, 0L)); buf483.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_24], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf498, model_layers_24_attention_kv_cache_0_k_cache, buf499, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf503 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf499, 4, int_array_9, int_array_10, 0L)); buf499.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_24], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf503, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf504 = std::move(buf484);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_24], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf503, model_layers_24_attention_kv_cache_0_v_cache, buf504, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf505 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf498, 3, int_array_13, int_array_14, 0L)); buf498.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_24], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf504, buf505, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf506 = std::move(buf493);  // reuse
    // Topologically Sorted Source Nodes: [linear_171], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf505, model_layers_24_attention_wo_weight, buf506, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf508 = std::move(buf472);  // reuse
    // Topologically Sorted Source Nodes: [mul_263, add_95, mul_272, add_98, rms_norm_49], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf487, buf491, buf506, model_layers_24_ffn_norm_weight, buf508, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf509 = std::move(buf490);  // reuse
    auto buf510 = std::move(buf489);  // reuse
    // Topologically Sorted Source Nodes: [linear_172, linear_173], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf508, model_layers_24_feed_forward_w1_weight, model_layers_24_feed_forward_w3_weight, buf509, buf510, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf511 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf505, 2, int_array_1, int_array_2, 0L)); buf505.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_174], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf509, buf510, model_layers_24_feed_forward_w2_weight, buf511, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf513 = std::move(buf508);  // reuse
    // Topologically Sorted Source Nodes: [mul_263, add_95, mul_272, add_98, mul_274, add_99, rms_norm_50], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf487, buf491, buf506, buf511, model_layers_25_attention_norm_weight, buf513, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf514 = std::move(buf450);  // reuse
    // Topologically Sorted Source Nodes: [linear_175], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf513, model_layers_25_attention_wq_weight, buf514, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf515 = std::move(buf494);  // reuse
    // Topologically Sorted Source Nodes: [linear_176, linear_177, index_put__51], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf513, model_layers_25_attention_wk_weight, model_layers_25_attention_wv_weight, arg358_1, buf515, model_layers_25_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__50], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf515, model_freqs_cis, model_layers_25_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf519 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf473, 4, int_array_5, int_array_6, 0L)); buf473.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_25], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf514, arg358_1, model_freqs_cis, buf519, 4096L, stream, kernels, this->cubin_dir_);
    auto buf520 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf503, 3, int_array_7, int_array_8, 0L)); buf503.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_25], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf519, model_layers_25_attention_kv_cache_0_k_cache, buf520, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf524 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf520, 4, int_array_9, int_array_10, 0L)); buf520.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_25], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf524, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf525 = std::move(buf504);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_25], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf524, model_layers_25_attention_kv_cache_0_v_cache, buf525, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf526 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf519, 3, int_array_13, int_array_14, 0L)); buf519.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_25], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf525, buf526, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf528 = std::move(buf487);  // reuse
    // Topologically Sorted Source Nodes: [mul_263, add_95, mul_272, add_98, mul_274, add_99, linear_178, mul_283, add_102], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf528, buf526, model_layers_25_attention_wo_weight, buf491, buf506, buf511, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf529 = std::move(buf492);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_51], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf528, buf529, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf530 = std::move(buf510);  // reuse
    auto buf531 = std::move(buf509);  // reuse
    // Topologically Sorted Source Nodes: [linear_179, linear_180], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf528, buf529, model_layers_25_ffn_norm_weight, model_layers_25_feed_forward_w1_weight, model_layers_25_feed_forward_w3_weight, buf530, buf531, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf532 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf526, 2, int_array_1, int_array_2, 0L)); buf526.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_181], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf530, buf531, model_layers_25_feed_forward_w2_weight, buf532, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf533 = std::move(buf529);  // reuse
    // Topologically Sorted Source Nodes: [mul_285, add_103, rms_norm_52], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf528, buf532, buf533, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf534 = std::move(buf511);  // reuse
    // Topologically Sorted Source Nodes: [linear_182], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf528, buf532, buf533, model_layers_26_attention_norm_weight, model_layers_26_attention_wq_weight, buf534, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf535 = std::move(buf515);  // reuse
    // Topologically Sorted Source Nodes: [linear_183, linear_184, index_put__53], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf528, buf532, buf533, model_layers_26_attention_norm_weight, model_layers_26_attention_wk_weight, model_layers_26_attention_wv_weight, arg358_1, buf535, model_layers_26_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__52], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf535, model_freqs_cis, model_layers_26_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf539 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf506, 4, int_array_5, int_array_6, 0L)); buf506.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_26], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf534, arg358_1, model_freqs_cis, buf539, 4096L, stream, kernels, this->cubin_dir_);
    auto buf540 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf524, 3, int_array_7, int_array_8, 0L)); buf524.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_26], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf539, model_layers_26_attention_kv_cache_0_k_cache, buf540, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf544 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf540, 4, int_array_9, int_array_10, 0L)); buf540.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_26], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf544, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf545 = std::move(buf525);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_26], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf544, model_layers_26_attention_kv_cache_0_v_cache, buf545, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf546 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf539, 3, int_array_13, int_array_14, 0L)); buf539.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_26], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf545, buf546, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf547 = std::move(buf534);  // reuse
    // Topologically Sorted Source Nodes: [linear_185], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf546, model_layers_26_attention_wo_weight, buf547, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf549 = std::move(buf513);  // reuse
    // Topologically Sorted Source Nodes: [mul_285, add_103, mul_294, add_106, rms_norm_53], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf528, buf532, buf547, model_layers_26_ffn_norm_weight, buf549, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf550 = std::move(buf531);  // reuse
    auto buf551 = std::move(buf530);  // reuse
    // Topologically Sorted Source Nodes: [linear_186, linear_187], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf549, model_layers_26_feed_forward_w1_weight, model_layers_26_feed_forward_w3_weight, buf550, buf551, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf552 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf546, 2, int_array_1, int_array_2, 0L)); buf546.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_188], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf550, buf551, model_layers_26_feed_forward_w2_weight, buf552, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf554 = std::move(buf549);  // reuse
    // Topologically Sorted Source Nodes: [mul_285, add_103, mul_294, add_106, mul_296, add_107, rms_norm_54], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf528, buf532, buf547, buf552, model_layers_27_attention_norm_weight, buf554, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf555 = std::move(buf491);  // reuse
    // Topologically Sorted Source Nodes: [linear_189], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf554, model_layers_27_attention_wq_weight, buf555, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf556 = std::move(buf535);  // reuse
    // Topologically Sorted Source Nodes: [linear_190, linear_191, index_put__55], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf554, model_layers_27_attention_wk_weight, model_layers_27_attention_wv_weight, arg358_1, buf556, model_layers_27_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__54], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf556, model_freqs_cis, model_layers_27_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf560 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf514, 4, int_array_5, int_array_6, 0L)); buf514.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_27], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf555, arg358_1, model_freqs_cis, buf560, 4096L, stream, kernels, this->cubin_dir_);
    auto buf561 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf544, 3, int_array_7, int_array_8, 0L)); buf544.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_27], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf560, model_layers_27_attention_kv_cache_0_k_cache, buf561, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf565 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf561, 4, int_array_9, int_array_10, 0L)); buf561.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_27], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf565, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf566 = std::move(buf545);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_27], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf565, model_layers_27_attention_kv_cache_0_v_cache, buf566, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf567 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf560, 3, int_array_13, int_array_14, 0L)); buf560.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_27], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf566, buf567, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf569 = std::move(buf528);  // reuse
    // Topologically Sorted Source Nodes: [mul_285, add_103, mul_294, add_106, mul_296, add_107, linear_192, mul_305, add_110], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf569, buf567, model_layers_27_attention_wo_weight, buf532, buf547, buf552, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf570 = std::move(buf533);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_55], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf569, buf570, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf571 = std::move(buf551);  // reuse
    auto buf572 = std::move(buf550);  // reuse
    // Topologically Sorted Source Nodes: [linear_193, linear_194], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf569, buf570, model_layers_27_ffn_norm_weight, model_layers_27_feed_forward_w1_weight, model_layers_27_feed_forward_w3_weight, buf571, buf572, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf573 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf567, 2, int_array_1, int_array_2, 0L)); buf567.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_195], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf571, buf572, model_layers_27_feed_forward_w2_weight, buf573, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf574 = std::move(buf570);  // reuse
    // Topologically Sorted Source Nodes: [mul_307, add_111, rms_norm_56], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf569, buf573, buf574, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf575 = std::move(buf552);  // reuse
    // Topologically Sorted Source Nodes: [linear_196], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf569, buf573, buf574, model_layers_28_attention_norm_weight, model_layers_28_attention_wq_weight, buf575, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf576 = std::move(buf556);  // reuse
    // Topologically Sorted Source Nodes: [linear_197, linear_198, index_put__57], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf569, buf573, buf574, model_layers_28_attention_norm_weight, model_layers_28_attention_wk_weight, model_layers_28_attention_wv_weight, arg358_1, buf576, model_layers_28_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__56], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf576, model_freqs_cis, model_layers_28_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf580 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf547, 4, int_array_5, int_array_6, 0L)); buf547.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_28], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf575, arg358_1, model_freqs_cis, buf580, 4096L, stream, kernels, this->cubin_dir_);
    auto buf581 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf565, 3, int_array_7, int_array_8, 0L)); buf565.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_28], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf580, model_layers_28_attention_kv_cache_0_k_cache, buf581, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf585 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf581, 4, int_array_9, int_array_10, 0L)); buf581.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_28], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf585, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf586 = std::move(buf566);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_28], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf585, model_layers_28_attention_kv_cache_0_v_cache, buf586, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf587 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf580, 3, int_array_13, int_array_14, 0L)); buf580.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_28], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf586, buf587, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf588 = std::move(buf575);  // reuse
    // Topologically Sorted Source Nodes: [linear_199], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf587, model_layers_28_attention_wo_weight, buf588, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf590 = std::move(buf554);  // reuse
    // Topologically Sorted Source Nodes: [mul_307, add_111, mul_316, add_114, rms_norm_57], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf569, buf573, buf588, model_layers_28_ffn_norm_weight, buf590, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf591 = std::move(buf572);  // reuse
    auto buf592 = std::move(buf571);  // reuse
    // Topologically Sorted Source Nodes: [linear_200, linear_201], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf590, model_layers_28_feed_forward_w1_weight, model_layers_28_feed_forward_w3_weight, buf591, buf592, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf593 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf587, 2, int_array_1, int_array_2, 0L)); buf587.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_202], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf591, buf592, model_layers_28_feed_forward_w2_weight, buf593, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf595 = std::move(buf590);  // reuse
    // Topologically Sorted Source Nodes: [mul_307, add_111, mul_316, add_114, mul_318, add_115, rms_norm_58], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf569, buf573, buf588, buf593, model_layers_29_attention_norm_weight, buf595, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf596 = std::move(buf532);  // reuse
    // Topologically Sorted Source Nodes: [linear_203], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf595, model_layers_29_attention_wq_weight, buf596, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf597 = std::move(buf576);  // reuse
    // Topologically Sorted Source Nodes: [linear_204, linear_205, index_put__59], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf595, model_layers_29_attention_wk_weight, model_layers_29_attention_wv_weight, arg358_1, buf597, model_layers_29_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__58], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf597, model_freqs_cis, model_layers_29_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf601 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf555, 4, int_array_5, int_array_6, 0L)); buf555.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_29], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf596, arg358_1, model_freqs_cis, buf601, 4096L, stream, kernels, this->cubin_dir_);
    auto buf602 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf585, 3, int_array_7, int_array_8, 0L)); buf585.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_29], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf601, model_layers_29_attention_kv_cache_0_k_cache, buf602, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf606 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf602, 4, int_array_9, int_array_10, 0L)); buf602.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_29], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf606, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf607 = std::move(buf586);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_29], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf606, model_layers_29_attention_kv_cache_0_v_cache, buf607, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf608 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf601, 3, int_array_13, int_array_14, 0L)); buf601.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_29], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf607, buf608, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf610 = std::move(buf569);  // reuse
    // Topologically Sorted Source Nodes: [mul_307, add_111, mul_316, add_114, mul_318, add_115, linear_206, mul_327, add_118], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf610, buf608, model_layers_29_attention_wo_weight, buf573, buf588, buf593, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf611 = std::move(buf574);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_59], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf610, buf611, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf612 = std::move(buf592);  // reuse
    auto buf613 = std::move(buf591);  // reuse
    // Topologically Sorted Source Nodes: [linear_207, linear_208], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf610, buf611, model_layers_29_ffn_norm_weight, model_layers_29_feed_forward_w1_weight, model_layers_29_feed_forward_w3_weight, buf612, buf613, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf614 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf608, 2, int_array_1, int_array_2, 0L)); buf608.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_209], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf612, buf613, model_layers_29_feed_forward_w2_weight, buf614, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf615 = std::move(buf611);  // reuse
    // Topologically Sorted Source Nodes: [mul_329, add_119, rms_norm_60], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf610, buf614, buf615, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf616 = std::move(buf593);  // reuse
    // Topologically Sorted Source Nodes: [linear_210], Original ATen: [aten.mm]
    call_triton_red_fused_mm_20(buf610, buf614, buf615, model_layers_30_attention_norm_weight, model_layers_30_attention_wq_weight, buf616, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf617 = std::move(buf597);  // reuse
    // Topologically Sorted Source Nodes: [linear_211, linear_212, index_put__61], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_21(buf610, buf614, buf615, model_layers_30_attention_norm_weight, model_layers_30_attention_wk_weight, model_layers_30_attention_wv_weight, arg358_1, buf617, model_layers_30_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    // Topologically Sorted Source Nodes: [index_put__60], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf617, model_freqs_cis, model_layers_30_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    auto buf621 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf588, 4, int_array_5, int_array_6, 0L)); buf588.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_30], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf616, arg358_1, model_freqs_cis, buf621, 4096L, stream, kernels, this->cubin_dir_);
    auto buf622 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf606, 3, int_array_7, int_array_8, 0L)); buf606.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_30], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf621, model_layers_30_attention_kv_cache_0_k_cache, buf622, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf626 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf622, 4, int_array_9, int_array_10, 0L)); buf622.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_30], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf626, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    auto buf627 = std::move(buf607);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_30], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf626, model_layers_30_attention_kv_cache_0_v_cache, buf627, 12288L, 102L, stream, kernels, this->cubin_dir_);
    auto buf628 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf621, 3, int_array_13, int_array_14, 0L)); buf621.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_30], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf627, buf628, 4096L, 3L, stream, kernels, this->cubin_dir_);
    auto buf629 = std::move(buf616);  // reuse
    // Topologically Sorted Source Nodes: [linear_213], Original ATen: [aten.mm]
    call_triton_red_fused_mm_9(buf628, model_layers_30_attention_wo_weight, buf629, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf631 = std::move(buf595);  // reuse
    // Topologically Sorted Source Nodes: [mul_329, add_119, mul_338, add_122, rms_norm_61], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_22(buf610, buf614, buf629, model_layers_30_ffn_norm_weight, buf631, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf632 = std::move(buf613);  // reuse
    auto buf633 = std::move(buf612);  // reuse
    // Topologically Sorted Source Nodes: [linear_214, linear_215], Original ATen: [aten.mm]
    call_triton_red_fused_mm_11(buf631, model_layers_30_feed_forward_w1_weight, model_layers_30_feed_forward_w3_weight, buf632, buf633, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf634 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf628, 2, int_array_1, int_array_2, 0L)); buf628.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_216], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf632, buf633, model_layers_30_feed_forward_w2_weight, buf634, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    auto buf636 = std::move(buf631);  // reuse
    // Topologically Sorted Source Nodes: [mul_329, add_119, mul_338, add_122, mul_340, add_123, rms_norm_62], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
    call_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_23(buf610, buf614, buf629, buf634, model_layers_31_attention_norm_weight, buf636, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf637 = std::move(buf573);  // reuse
    // Topologically Sorted Source Nodes: [linear_217], Original ATen: [aten.mm]
    call_triton_red_fused_mm_14(buf636, model_layers_31_attention_wq_weight, buf637, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf638 = std::move(buf617);  // reuse
    // Topologically Sorted Source Nodes: [linear_218, linear_219, index_put__63], Original ATen: [aten.mm, aten.index_put]
    call_triton_red_fused_index_put_mm_15(buf636, model_layers_31_attention_wk_weight, model_layers_31_attention_wv_weight, arg358_1, buf638, model_layers_31_attention_kv_cache_0_v_cache, 1024L, 4096L, stream, kernels, this->cubin_dir_);
    buf636.reset();
    // Topologically Sorted Source Nodes: [index_put__62], Original ATen: [aten.index_put]
    call_triton_poi_fused_index_put_3(arg358_1, buf638, model_freqs_cis, model_layers_31_attention_kv_cache_0_k_cache, 1024L, stream, kernels, this->cubin_dir_);
    buf638.reset();
    auto buf642 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf596, 4, int_array_5, int_array_6, 0L)); buf596.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_31], Original ATen: [aten._to_copy, aten.mul]
    call_triton_poi_fused__to_copy_mul_4(buf637, arg358_1, model_freqs_cis, buf642, 4096L, stream, kernels, this->cubin_dir_);
    buf637.reset();
    auto buf643 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf626, 3, int_array_7, int_array_8, 0L)); buf626.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_31], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_5(buf642, model_layers_31_attention_kv_cache_0_k_cache, buf643, 9728L, 128L, stream, kernels, this->cubin_dir_);
    auto buf647 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf643, 4, int_array_9, int_array_10, 0L)); buf643.reset();  // reuse
    // Topologically Sorted Source Nodes: [index, scaled_dot_product_attention_31], Original ATen: [aten.index, aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
    call_triton_per_fused__safe_softmax_add_index_scalar_tensor_where_6(buf647, arg358_1, model_causal_mask, 32L, 304L, stream, kernels, this->cubin_dir_);
    arg358_1.reset();
    auto buf648 = std::move(buf627);  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_31], Original ATen: [aten.bmm]
    call_triton_red_fused_bmm_7(buf647, model_layers_31_attention_kv_cache_0_v_cache, buf648, 12288L, 102L, stream, kernels, this->cubin_dir_);
    buf647.reset();
    auto buf649 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf642, 3, int_array_13, int_array_14, 0L)); buf642.reset();  // reuse
    // Topologically Sorted Source Nodes: [scaled_dot_product_attention_31], Original ATen: [aten.bmm]
    call_triton_per_fused_bmm_8(buf648, buf649, 4096L, 3L, stream, kernels, this->cubin_dir_);
    buf648.reset();
    auto buf651 = std::move(buf610);  // reuse
    // Topologically Sorted Source Nodes: [mul_329, add_119, mul_338, add_122, mul_340, add_123, linear_220, mul_349, add_126], Original ATen: [aten.mul, aten.add, aten.mm]
    call_triton_red_fused_add_mm_mul_24(buf651, buf649, model_layers_31_attention_wo_weight, buf614, buf629, buf634, 4096L, 4096L, stream, kernels, this->cubin_dir_);
    buf614.reset();
    buf629.reset();
    buf634.reset();
    auto buf652 = std::move(buf615);  // reuse
    // Topologically Sorted Source Nodes: [rms_norm_63], Original ATen: [aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_mean_pow_17(buf651, buf652, 1L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf653 = std::move(buf633);  // reuse
    auto buf654 = std::move(buf632);  // reuse
    // Topologically Sorted Source Nodes: [linear_221, linear_222], Original ATen: [aten.mm]
    call_triton_red_fused_mm_18(buf651, buf652, model_layers_31_ffn_norm_weight, model_layers_31_feed_forward_w1_weight, model_layers_31_feed_forward_w3_weight, buf653, buf654, 14336L, 4096L, stream, kernels, this->cubin_dir_);
    auto buf655 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf649, 2, int_array_1, int_array_2, 0L)); buf649.reset();  // reuse
    // Topologically Sorted Source Nodes: [linear_223], Original ATen: [aten.mm]
    call_triton_red_fused_mm_12(buf653, buf654, model_layers_31_feed_forward_w2_weight, buf655, 4096L, 14336L, stream, kernels, this->cubin_dir_);
    buf653.reset();
    buf654.reset();
    auto buf656 = std::move(buf652);  // reuse
    // Topologically Sorted Source Nodes: [mul_351, add_127, rms_norm_64], Original ATen: [aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean]
    call_triton_red_fused__to_copy_add_mean_mul_pow_19(buf651, buf655, buf656, 1L, 4096L, stream, kernels, this->cubin_dir_);
    static constexpr int64_t int_array_19[] = {1L, 128256L};
    static constexpr int64_t int_array_20[] = {128256L, 1L};
    AtenTensorHandle buf658_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_19, int_array_20, cached_torch_dtype_bfloat16, cached_torch_device_type_cuda, this->device_idx_, &buf658_handle));
    RAIIAtenTensorHandle buf658(buf658_handle);
    // Topologically Sorted Source Nodes: [linear_224], Original ATen: [aten.mm]
    call_triton_red_fused_mm_25(buf651, buf655, buf656, model_norm_weight, model_output_weight, buf658, 128256L, 4096L, stream, kernels, this->cubin_dir_);
    buf651.reset();
    buf655.reset();
    buf656.reset();
    static constexpr int64_t int_array_21[] = {1L, 1L, 128256L};
    static constexpr int64_t int_array_22[] = {128256L, 128256L, 1L};
    auto var_0 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf658, 3, int_array_21, int_array_22, 0L));
    output_handles[0] = var_0.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor





// Compile cmd
// g++ /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cnyzukfltdxhtc327mswuv52zgb6xlcf5m74mgv57fovyva3p5gs.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D AOTI_LIBTORCH_FREE -D  C10_USING_CUSTOM_GENERATED_MACROS -D  USE_MMAP_SELF -D  USE_CUDA  -fPIC -O3 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fexcess-precision=fast -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fno-tree-loop-vectorize -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -fopenmp  -I/home/binbao/local/miniconda3/envs/pytorch-3.10/include/python3.10 -I/data/users/binbao/pytorch/torch/include -I/data/users/binbao/pytorch/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.6/include   -D_GLIBCXX_USE_CXX11_ABI=1  -c -o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cnyzukfltdxhtc327mswuv52zgb6xlcf5m74mgv57fovyva3p5gs.wrapper.o
// Link cmd
// g++ /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cnyzukfltdxhtc327mswuv52zgb6xlcf5m74mgv57fovyva3p5gs.wrapper.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/culsenbim4j3xtwlyhbtlrqtjs5cjdrrnxghkcdfuq5au2v53svk.kernel.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cnyzukfltdxhtc327mswuv52zgb6xlcf5m74mgv57fovyva3p5gs/cr4tcammsbazzjghn4cqndluwq5xvc7brjgs5gyg7p2vjohf4xmo.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cnvkbngwgvhxwmky27tiknylrvgfd7hrupwczdxythtx2iwmxmnw.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cefosdeer5eimhrl5y2wqxsmwkolmf64lwwulmzksr4pqsg6baft.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/crhflw4opepsg6lvvv5ga3oknnb4vm4bknuud6vbqayhrl4st654.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c24pugketupm3cgf7hh4pijiyvg4estagagxw6qi7uhfcqbz3sp6.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/czsemnym6pmqx2kapz52dyhwz6y3qsqv6yz3hmvrmmnytbcyem4x.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cu6iwiujymsulhca5uo4qwnzntn3x52lori62acz6nlpqetnsayd.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c546empi4r4rqktgs5mozszzg5e3xoxmu26rq4bnk3oj2frrmm2v.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c2r6h26gm3hj3wc3o5nlvvwft4lad6ngre76kyjuw63eyciegxmd.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c3zw26b3if5ad7na6oq4uihubi45imzjonjc5xkcs4kshoa7u3lu.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cbpmmgpymme5e5deysrtvqkpqxp3ak3jsgwnvqkt6jotvzojb3ci.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cuw3h6ozeixh5mqp54nt44sgky5dfdrdgknlbhw3dz6xgbmhpdxf.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/csbtmp7akkfebxa4cwwgpf4lnssooptn46my4274afg4g2babm6g.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c676gwke3qpj2yjj3akc74xndbe6jl7idikdvuyz2phsrwvcftim.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cmt5ppczxxa4q73xbsur3ao24cq4n5t4ryebe3lupu44jmcidjxq.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cqqsmtxtmfalmu5pua4v2omod4raauxko4i3rdboccvfadzf2ghp.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c3loerzgt2ftqga7igfaeulzkuuavytw2v35agpfalklisfzaogh.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/celbgvy2zwlgqwrliqbcc6xai2z6sowb56d4m2mbyhoqm6k2kjne.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cpevqxptowrcjwtnwjp2y6pwhhxuvu7n6y6wdg5cm5oqrmlqfjfa.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c2hxe4l2c2xbbly6ykpa5oq7kguqyld2l77xxoi3o2itxp7n7cki.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c6mocmn67pg5tylothqacas5pqdtsm74zvoviqugdhzc6webgcys.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c52awljwyyqhwgm5nyw4vwbfpw6xvbzcvozxp454a5xmmy4ysgib.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c7jjs5qgw52dafckch4n5xkwqlwe62q5lxoaoharlplk6wvfoiwq.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/c2d575tvqkplsnmwjpt56463tqsct66pojbc3z5snggco3yai3oh.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cymshjw44xdmodbmdst2t3ulmvfbmnrwk77q724ks2an7fkgricj.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/clcipnjoej3edprogdpq22gzne7exqytivisaric2yabqwutvrgh.cubin.o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cybdk2w7jmkzulymnppz6xvpxisfet6ed6tzlgpehgqfh6cspxsa.cubin.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D AOTI_LIBTORCH_FREE -D  C10_USING_CUSTOM_GENERATED_MACROS -D  USE_CUDA  -shared -fPIC -O3 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fexcess-precision=fast -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fno-tree-loop-vectorize -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -fopenmp  -I/home/binbao/local/miniconda3/envs/pytorch-3.10/include/python3.10 -I/data/users/binbao/pytorch/torch/include -I/data/users/binbao/pytorch/torch/include/torch/csrc/api/include -I/usr/local/cuda-12.6/include   -D_GLIBCXX_USE_CXX11_ABI=1  -o /tmp/torchinductor_binbao/cm2ckn3cielmjq5ziqaltkg7jbullpzotjaavzd2u6xusownecou/cnyzukfltdxhtc327mswuv52zgb6xlcf5m74mgv57fovyva3p5gs.wrapper.so  -lgomp -lopenblas -lsleef -lcuda -lcublas  -L/home/binbao/local/miniconda3/envs/pytorch-3.10/lib -L/data/users/binbao/pytorch/torch/lib -L/usr/local/cuda-12.6/lib64 
