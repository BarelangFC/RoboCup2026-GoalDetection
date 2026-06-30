#include "trt_infer.hpp"
#include "cuda_utils.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    // Only log warnings and errors
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << std::endl;
    }
}

TrtEngine::TrtEngine() {}
TrtEngine::~TrtEngine() { unload(); }

bool TrtEngine::load(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[TRT] Failed to open: " << engine_path << std::endl;
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    m_runtime = nvinfer1::createInferRuntime(m_logger);
    if (!m_runtime) return false;

    m_engine = m_runtime->deserializeCudaEngine(engine_data.data(), size);
    if (!m_engine) return false;

    m_context = m_engine->createExecutionContext();
    if (!m_context) return false;

    // Get bindings
    m_num_outputs = 0;
    for (int i = 0; i < m_engine->getNbBindings(); i++) {
        if (m_engine->bindingIsInput(i)) {
            m_input_idx = i;
            m_input_dims = m_engine->getBindingDimensions(i);
            m_input_size = 1;
            for (int j = 0; j < m_input_dims.nbDims; j++)
                m_input_size *= m_input_dims.d[j];
        } else {
            m_output_idx.push_back(i);
            m_num_outputs++;
            auto dims = m_engine->getBindingDimensions(i);
            m_output_dims.push_back(dims);
            int64_t sz = 1;
            for (int j = 0; j < dims.nbDims; j++)
                sz *= dims.d[j];
            m_output_sizes.push_back(sz);
        }
    }

    std::cout << "[TRT] Engine loaded: " << engine_path << std::endl;
    std::cout << "       Input: " << m_input_dims.d[1] << "x" << m_input_dims.d[2]
              << "x" << m_input_dims.d[3] << "  size=" << m_input_size << std::endl;
    std::cout << "       Outputs: " << m_num_outputs;
    for (int i = 0; i < m_num_outputs; i++)
        std::cout << " [" << i << "] size=" << m_output_sizes[i];
    std::cout << std::endl;
    return true;
}

bool TrtEngine::infer(float* input, int batch_size, std::vector<float*>& outputs) {
    if (!m_context) return false;

    // Allocate device buffers (first call)
    if (m_buffers[0] == nullptr) {
        CUDA_CHECK(cudaMalloc(&m_buffers[0], m_input_size * sizeof(float)));
    }
    if (m_buffers[1] == nullptr) {
        CUDA_CHECK(cudaMalloc(&m_buffers[1], m_output_sizes[0] * sizeof(float)));
    }

    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(m_buffers[0], input, m_input_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Run inference
    void* bindings[] = {m_buffers[0], m_buffers[1]};
    if (!m_context->executeV2(bindings)) {
        return false;
    }

    // Copy output back
    CUDA_CHECK(cudaMemcpy(outputs[0], m_buffers[1],
                          m_output_sizes[0] * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return true;
}

void TrtEngine::unload() {
    if (m_buffers[0]) { cudaFree(m_buffers[0]); m_buffers[0] = nullptr; }
    if (m_buffers[1]) { cudaFree(m_buffers[1]); m_buffers[1] = nullptr; }
    if (m_context) { delete m_context; m_context = nullptr; }
    if (m_engine) { delete m_engine; m_engine = nullptr; }
    if (m_runtime) { delete m_runtime; m_runtime = nullptr; }
}

int64_t TrtEngine::get_output_size(int i) const {
    if (i < (int)m_output_sizes.size()) return m_output_sizes[i];
    return 0;
}
