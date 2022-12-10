#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <chrono>

#include "onnxruntime_cxx_api.h"
#include "wav.h"

class VadModel
{
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

public:
    void init_engine_threads(int inter_threads, int intra_threads)
    {   
        // The method should be called in each thread/proc in multi-thread/proc work
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    void init_onnx_model(const std::string &model_path)
    {   
        // Init threads = 1 for 
        init_engine_threads(1, 1);
        // Load model
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    }

    void reset_states()
    {
        // Call reset before each audio start
        std::memset(_h.data(), 0.0f, _h.size() * sizeof(float));
        std::memset(_c.data(), 0.0f, _c.size() * sizeof(float));
        triggerd = false;
        temp_end = 0;
        current_sample = 0;
    }

    // Call it in predict func. if you prefer raw bytes input.
    void bytes_to_float_tensor(const char *pcm_bytes) 
    {
        std::memcpy(input.data(), pcm_bytes, window_size_samples * sizeof(int16_t));
        for (int i = 0; i < window_size_samples; i++)
        {
            input[i] = static_cast<float>(input[i]) / 32768; // int16_t normalized to float
        }
    }


    void predict(const std::vector<float> &data) // const char *data
    {
        // bytes_to_float_tensor(data); 
        
        // Infer
        // Inputs
        input.assign(data.begin(), data.end());
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        // std::cout << "input size:" << input.size() << std::endl;
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        Ort::Value h_ort = Ort::Value::CreateTensor<float>(
            memory_info, _h.data(), _h.size(), hc_node_dims, 3);
        Ort::Value c_ort = Ort::Value::CreateTensor<float>(
            memory_info, _c.data(), _c.size(), hc_node_dims, 3);

        ort_inputs.clear(); // clear inputs
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(sr_ort));
        ort_inputs.emplace_back(std::move(h_ort));
        ort_inputs.emplace_back(std::move(c_ort));

        // Infer
        ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        // out put Probability & update h,c recursively
        float output = ort_outputs[0].GetTensorMutableData<float>()[0];
        float *hn = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_h.data(), hn, size_hc * sizeof(float));
        float *cn = ort_outputs[2].GetTensorMutableData<float>();
        std::memcpy(_c.data(), cn, size_hc * sizeof(float));

        // Push forward sample index
        current_sample += window_size_samples;
        
        // 1) Reset temp_end when > threshold 
        if ((output >= threshold) && (temp_end != 0))
        {
            temp_end = 0;
        }
        // 2) Trigger and start sentence
        if ((output >= threshold) && (triggerd == false))
        {
            triggerd = true;
            speech_start = current_sample - speech_pad_samples;
            printf("{ start: %.3f s }\n", 1.0 * current_sample / sample_rate);
        }
        // 3) Speaking 
        if ((output >= (threshold - 0.15)) && (triggerd == true))
        {
            printf("{ speaking: %.3f s }\n", 1.0 * current_sample / sample_rate);
        }
        // 4) End 
        if ((output < (threshold - 0.15)) && (triggerd == true))
        {

            if (temp_end != 0)
            {
                temp_end = current_sample;
            }
            // a. silence < min_slience_samples, continue speaking 
            if ((current_sample - temp_end) < min_silence_samples)
            {
                printf("{ speaking: %.3f s }\n", 1.0 * current_sample / sample_rate);
            }
            // b. silence >= min_slience_samples, end speaking
            else
            {
                speech_end = temp_end + speech_pad_samples;
                temp_end = 0;
                triggerd = false;
                printf("{ end: %.3f s }\n", 1.0 * current_sample / sample_rate);
            }
        }
        // 5) Silence
        if ((output < threshold) && (triggerd == false))
        {
            printf("{ silence: %.3f s }\n", 1.0 * current_sample / sample_rate);
        }

    }

    // Print input output shape of the model
    void GetInputOutputInfo(
        const std::shared_ptr<Ort::Session> &session,
        std::vector<const char *> *in_names, std::vector<const char *> *out_names)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        // Input info
        int num_nodes = session->GetInputCount();
        in_names->resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            char *name = session->GetInputName(i, allocator);
            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::vector<int64_t> node_dims = tensor_info.GetShape();
            std::stringstream shape;
            for (auto j : node_dims)
            {
                shape << j;
                shape << " ";
            }
            std::cout << "\tInput " << i << " : name=" << name << " type=" << type
                      << " dims=" << shape.str() << std::endl;
            (*in_names)[i] = name;
        }
        // Output info
        num_nodes = session->GetOutputCount();
        out_names->resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            char *name = session->GetOutputName(i, allocator);
            Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            std::vector<int64_t> node_dims = tensor_info.GetShape();
            std::stringstream shape;
            for (auto j : node_dims)
            {
                shape << j;
                shape << " ";
            }
            std::cout << "\tOutput " << i << " : name=" << name << " type=" << type
                      << " dims=" << shape.str() << std::endl;
            ;
            (*out_names)[i] = name;
        }
    }

private:
    // model config
    int64_t window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sample_rate;
    int sr_per_ms;  // Assign when init, support 8 or 16
    float threshold = 0.5;
    int min_silence_samples; // sr_per_ms * #ms
    int speech_pad_samples = 0; // Can be used in offline infer to get as much speech as possible

    // model states
    bool triggerd = false;
    unsigned int speech_start = 0; 
    unsigned int speech_end = 0;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;    
    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes  
    float output;

    // Onnx model
    // Inputs
    std::vector<Ort::Value> ort_inputs;
    
    std::vector<const char *> input_node_names = {"input", "sr", "h", "c"};
    std::vector<float> input;
    std::vector<int64_t> sr;
    unsigned int size_hc = 2 * 1 * 64; // It's FIXED.
    std::vector<float> _h;
    std::vector<float> _c;

    int64_t input_node_dims[2] = {}; 
    const int64_t sr_node_dims[1] = {1};
    const int64_t hc_node_dims[3] = {2, 1, 64};

    // Outputs
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char *> output_node_names = {"output", "hn", "cn"};
    

public:
    // Construct init
    VadModel(const std::string ModelPath, 
             int sample_rate, int frame_size, 
             float threshold, int min_silence_duration_ms, int speech_pad_ms) 
    {
        init_onnx_model(ModelPath);
        sr_per_ms = sample_rate / 1000;
        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        speech_pad_samples = sr_per_ms * speech_pad_ms;
        window_size_samples = frame_size * sr_per_ms; // Input 64ms/frame * 8ms = 512 samples/frame
        input.resize(window_size_samples);
        input_node_dims[0] = 1;
        input_node_dims[1] = window_size_samples;
        // std::cout << "== Input size" << input.size() << std::endl;
        _h.resize(size_hc);
        _c.resize(size_hc);
        sr.resize(1);
    }

};

int main()
{

    // Read wav
    wav::WavReader wav_reader("silero-vad-master/test_audios/test0_for_vad.wav");

    std::vector<int16_t> data(wav_reader.num_samples());
    std::vector<float> input_wav(wav_reader.num_samples());

    for (int i = 0; i < wav_reader.num_samples(); i++)
    {
        data[i] = static_cast<int16_t>(*(wav_reader.data() + i));
    }

    for (int i = 0; i < wav_reader.num_samples(); i++)
    {
        input_wav[i] = static_cast<float>(data[i]) / 32768;
    }

    std::string path = "silero-vad-master/files/silero_vad.onnx";
    int test_sr = 8000;
    int test_frame_ms = 64;
    int test_window_samples = test_frame_ms * (test_sr/1000);
    VadModel vad(path, test_sr, test_frame_ms);
    // std::cout << "== 3" << std::endl;
    // std::cout << vad.window_size_samples1() << std::endl;

    for (int j = 0; j < wav_reader.num_samples(); j += test_window_samples)
    {
        std::vector<float> r{&input_wav[0] + j, &input_wav[0] + j + test_window_samples};
        auto start = std::chrono::high_resolution_clock::now();
        // Predict and print throughout process time
        vad.predict(r);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
        std::cout << "== Elapsed time: " << elapsed_time.count() << "ns" << " ==" <<std::endl;

    }
}
