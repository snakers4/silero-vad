This is the source code for Silero-VAD V5 in C++, based on LibTorch.
The primary implementation is the CPU version, and you should compare its results with the Python version.
I only checked the 16kHz results.

In addition, Batch and CUDA inference options are also available if you want to explore further. 
Note that when using batch inference, the speech probabilities might slightly differ from the standard version, likely due to differences in caching. 
Unlike processing individual inputs, batch inference may not be able to use the cache from previous chunks. 
Nevertheless, batch inference provides significantly faster processing. 
For optimal performance, carefully adjust the threshold when using batch inference.

#Requirements:
GCC 11.4.0 (GCC >= 5.1)
LibTorch 1.13.0(Other versions are also acceptable)

#Download Libtorch:
   *cpu
   $wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.0%2Bcpu.zip
   $unzip libtorch-shared-with-deps-1.13.0+cpu.zip 

   *cuda
   $wget https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.13.0%2Bcu116.zip
   $unzip libtorch-shared-with-deps-1.13.0+cu116.zip

#complie:
   *cpu
   $g++ main.cc silero_torch.cc -I ./libtorch/include/ -I ./libtorch/include/torch/csrc/api/include -L ./libtorch/lib/ -ltorch -ltorch_cpu -lc10 -Wl,-rpath,./libtorch/lib/ -o silero -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0
   *cuda
   $g++ main.cc silero_torch.cc -I ./libtorch/include/ -I ./libtorch/include/torch/csrc/api/include -L ./libtorch/lib/ -ltorch -ltorch_cuda -ltorch_cpu -lc10 -Wl,-rpath,./libtorch/lib/ -o silero -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -DUSE_GPU

   *option to add
   -DUSE_BATCH
   -DUSE_GPU

# Run:
./silero aepyx.wav 16000 0.5	#The sample file 'aepyx.wav' is part of the Voxconverse dataset.
				#aepyx.wav : 16kHz, 16-bit
