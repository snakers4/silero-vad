# Stream example in C++

Here's a simple example of the vad model in c++ onnxruntime.



## Requirements

The example has been tested in the environment below; other environments may work as well.

- WSL2 + Debian-bullseye (docker)  
- gcc 12.2.0
- onnxruntime-linux-x64-1.12.1



## Usage

1. Install gcc 12.2.0, or just pull the docker image with `docker pull gcc:12.2.0-bullseye`

2. Install onnxruntime-linux-x64-1.12.1

   - Download lib onnxruntime: 

     `wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz`

   - Unzip. Assume the path is `/root/onnxruntime-linux-x64-1.12.1`

3. Modify the WAV path and test configuration in the main function

   `wav::WavReader wav_reader("${path_to_your_wav_file}");`

   The v5 ONNX model expects 8 kHz or 16 kHz audio. The example uses the corresponding
   256- or 512-sample window and passes the sample rate as the `sr` input. Keep the
   model, WAV sample rate, and window size consistent when changing the configuration.

   Adjust the threshold and silence settings as needed for your audio.

4. Build with gcc and run

   ```bash
   # Build
   g++ silero-vad-onnx.cpp -I /root/onnxruntime-linux-x64-1.12.1/include/ -L /root/onnxruntime-linux-x64-1.12.1/lib/ -lonnxruntime  -Wl,-rpath,/root/onnxruntime-linux-x64-1.12.1/lib/ -o test
   
   # Run
   ./test
   ```