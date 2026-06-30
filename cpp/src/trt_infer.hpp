#ifndef TRT_INFER_H
#define TRT_INFER_H

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    bool load(const std::string& engine_path);
    bool infer(float* input, int batch_size, std::vector<float*>& outputs);
    void unload();

    int get_input_index() const { return m_input_idx; }
    int get_output_index(int i) const { return i < (int)m_output_idx.size() ? m_output_idx[i] : -1; }
    nvinfer1::Dims get_input_dims() const { return m_input_dims; }
    nvinfer1::Dims get_output_dims(int i) const { return i < (int)m_output_dims.size() ? m_output_dims[i] : nvinfer1::Dims{}; }
    int num_outputs() const { return m_num_outputs; }
    int64_t get_input_size() const { return m_input_size; }
    int64_t get_output_size(int i) const;
    bool is_loaded() const { return m_engine != nullptr; }

private:
    TrtLogger m_logger;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;

    int m_input_idx = -1;
    std::vector<int> m_output_idx;
    nvinfer1::Dims m_input_dims;
    std::vector<nvinfer1::Dims> m_output_dims;
    int m_num_outputs = 0;
    int64_t m_input_size = 0;
    std::vector<int64_t> m_output_sizes;

    void* m_buffers[2] = {nullptr, nullptr};  // input, output
};

#endif
