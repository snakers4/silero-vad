#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <chrono>

#include "onnxruntime_cxx_api.h"
#include "wav.h"

class VadIterator
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


    void predict(const std::vector<float> &data)
    {
        // bytes_to_float_tensor(data); 
        
        // Infer
        // Create ort tensors
        input.assign(data.begin(), data.end());
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        Ort::Value h_ort = Ort::Value::CreateTensor<float>(
            memory_info, _h.data(), _h.size(), hc_node_dims, 3);
        Ort::Value c_ort = Ort::Value::CreateTensor<float>(
            memory_info, _c.data(), _c.size(), hc_node_dims, 3);

        // Clear and add inputs
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(sr_ort));
        ort_inputs.emplace_back(std::move(h_ort));
        ort_inputs.emplace_back(std::move(c_ort));

        // Infer
        ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        // Output probability & update h,c recursively
        float output = ort_outputs[0].GetTensorMutableData<float>()[0];
        float *hn = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_h.data(), hn, size_hc * sizeof(float));
        float *cn = ort_outputs[2].GetTensorMutableData<float>();
        std::memcpy(_c.data(), cn, size_hc * sizeof(float));

        // Push forward sample index
        current_sample += window_size_samples;
        
        // Reset temp_end when > threshold 
        if ((output >= threshold) && (temp_end != 0))
        {
            temp_end = 0;
        }
        // 1) Silence
        if ((output < threshold) && (triggerd == false))
        {
            // printf("{ silence: %.3f s }\n", 1.0 * current_sample / sample_rate);
        }
        // 2) Speaking 
        if ((output >= (threshold - 0.15)) && (triggerd == true))
        {
            // printf("{ speaking_2: %.3f s }\n", 1.0 * current_sample / sample_rate);
        }

        // 3) Start
        if ((output >= threshold) && (triggerd == false))
        {
            triggerd = true;
            speech_start = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
            printf("{ start: %.3f s }\n", 1.0 * speech_start / sample_rate);
        }

        // 4) End 
        if ((output < (threshold - 0.15)) && (triggerd == true))
        {

            if (temp_end == 0)
            {
                temp_end = current_sample;
            }
            // a. silence < min_slience_samples, continue speaking 
            if ((current_sample - temp_end) < min_silence_samples)
            {
                // printf("{ speaking_4: %.3f s }\n", 1.0 * current_sample / sample_rate);
                // printf("");
            }
            // b. silence >= min_slience_samples, end speaking
            else
            {
                speech_end = temp_end ? temp_end + speech_pad_samples : current_sample + speech_pad_samples;
                temp_end = 0;
                triggerd = false;
                printf("{ end: %.3f s }\n", 1.0 * speech_end / sample_rate);
            }
        }


    }

private:
    // model config
    int64_t window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sample_rate;
    int sr_per_ms;  // Assign when init, support 8 or 16
    float threshold;
    int min_silence_samples; // sr_per_ms * #ms
    int speech_pad_samples; // usually a 

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
    // Construction
    VadIterator(const std::string ModelPath, 
             int Sample_rate, int frame_size, 
             float Threshold, int min_silence_duration_ms, int speech_pad_ms) 
    {
        init_onnx_model(ModelPath);
        sample_rate = Sample_rate;
        sr_per_ms = sample_rate / 1000;
        threshold = Threshold;
        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        speech_pad_samples = sr_per_ms * speech_pad_ms;
        window_size_samples = frame_size * sr_per_ms;
        
        input.resize(window_size_samples);
        input_node_dims[0] = 1;
        input_node_dims[1] = window_size_samples;
        // std::cout << "== Input size" << input.size() << std::endl;
        _h.resize(size_hc);
        _c.resize(size_hc);
        sr.resize(1);
        sr[0] = sample_rate;
    }

};

int main()
{

    // Read wav
    wav::WavReader wav_reader("./test_for_vad.wav");
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

    // ===== Test configs =====
    std::string path = "../files/silero_vad.onnx";
    int test_sr = 8000;
    int test_frame_ms = 64;
    float test_threshold = 0.5f;
    int test_min_silence_duration_ms = 0;
    int test_speech_pad_ms = 0;
    int test_window_samples = test_frame_ms * (test_sr/1000);

    VadIterator vad(
        path, test_sr, test_frame_ms, test_threshold,
        test_min_silence_duration_ms, test_speech_pad_ms);

    for (int j = 0; j < wav_reader.num_samples(); j += test_window_samples)
    {
        // std::cout << "== 4" << std::endl;
        std::vector<float> r{&input_wav[0] + j, &input_wav[0] + j + test_window_samples};
        auto start = std::chrono::high_resolution_clock::now();
        // Predict and print throughout process time
        vad.predict(r);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
        // std::cout << "== Elapsed time: " << 1.0*elapsed_time.count()/1000000 << "ms" << " ==" <<std::endl;

    }
}
