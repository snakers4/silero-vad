# Silero VAD in OpenVINO

The stock `silero_vad.onnx` does not load in OpenVINO. `read_model` fails with:

```
OpConversionFailure: Model wasn't fully converted.
-- Conv-16: While validating ONNX node
   '<Node(Conv): If_0_then_branch__Inline_0__/decoder/decoder/2/Conv_output_0>':
   Check 'data.get_partial_shape().rank().is_static()' failed
   The input data tensor's rank has to be known (static)
```

The ONNX frontend cannot infer static ranks for the Conv ops (the STFT) inside the If subgraphs. There is one top level If that switches between the 8 kHz and 16 kHz paths, plus 7 nested If nodes per branch left behind by TorchScript tracing, so folding the sample rate constant alone is not enough.

None of those conditions depend on the audio itself, only on the sample rate and tensor shapes, so for a fixed configuration they can all be resolved offline. `convert.py` inlines every If by evaluating its condition empirically with onnxruntime, removes the dead branch including its weights (the file drops from 2.27 MB to 1.23 MB), folds Identity nodes (recent OpenVINO serializes them as opset16::Identity, which older CPU plugins cannot execute) and fixes static shapes. The result loads and compiles in OpenVINO directly, as ONNX or as exported IR.

## Usage

```
pip install numpy onnx onnxruntime openvino

python convert.py                  # writes silero_vad_16k.onnx
python convert.py --sr 8000        # writes silero_vad_8k.onnx
python convert.py --save-ir        # also exports OpenVINO IR

python verify.py silero_vad_16k.onnx ../../tests/data/test.wav "../c++/aepyx.wav"
python verify.py silero_vad_8k.onnx --sr 8000 "../c++/aepyx_8k.wav"
```

`convert.py` defaults to the model shipped in this repo. `verify.py` compares the converted model against the stock one with chained state, on synthetic audio and on any 16 bit mono wav files you pass, and checks that the resulting speech segmentation is identical. It exits nonzero if anything is off.

## Set inference precision to f32

On CPUs with bf16 support (AMX or AVX512 BF16) the OpenVINO CPU plugin defaults to bf16 inference. For this model that is not a harmless accuracy tradeoff: the per chunk error compounds through the recurrent state until the speech segmentation actually changes. On real audio the default gave max abs diffs up to 3e-1 against the stock model, while f32 gives 2e-6. Always compile with:

```python
compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
```

or `ov::hint::inference_precision(ov::element::f32)` in C++. The model is tiny, f32 costs nothing.

## Inference

I/O contract of the converted model: `input` f32 [1, 576] at 16 kHz (64 context samples followed by 512 new ones; [1, 288] as 32 plus 256 at 8 kHz), `state` f32 [2, 1, 128], outputs `output` f32 [1, 1] and `stateN` f32 [2, 1, 128]. Chain `stateN` back into `state`, keep the last 64 (or 32) input samples as the next context, and reset both to zeros when a new stream starts. Same protocol as the OnnxWrapper in this repo.

```python
import numpy as np
import openvino as ov

req = ov.Core().compile_model("silero_vad_16k.onnx", "CPU",
    {"INFERENCE_PRECISION_HINT": "f32"}).create_infer_request()
state = np.zeros((2, 1, 128), np.float32)
ctx = np.zeros((1, 64), np.float32)
for chunk in stream_of_512_sample_chunks:
    x = np.concatenate([ctx, chunk[None]], axis=1)
    res = req.infer({"input": x, "state": state})
    prob, state, ctx = res["output"][0, 0], res["stateN"], x[:, -64:]
```

## Validation

Converted vs stock, chained state throughout. In onnxruntime the converted model is bit exact (max abs diff 0.0) for both sample rates, so the transformation is the mathematical identity for the chosen rate. In OpenVINO with f32: at 16 kHz, max abs diff 2.3e-06 over `tests/data/test.wav` plus `examples/c++/aepyx.wav` (229 s, 7161 chunks) with identical segmentation (29 of 29 and 65 of 65 segments). At 8 kHz, max abs diff 4.5e-06 over `examples/c++/aepyx_8k.wav` with identical segmentation (79 of 79). Reproduced on Linux and Windows across two OpenVINO versions. Source model sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

## Caveats

Each converted model is single sample rate (the `sr` input is removed) with batch fixed to 1. Some of the inlined guards may be batch dependent, so batch 1 is validated exhaustively and larger batches are not. IR files are version sensitive: regenerate the IR with your target OpenVINO version, or load the cleaned ONNX directly with `read_model`.
