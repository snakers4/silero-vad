from .utils_vad import init_jit_model, OnnxWrapper
import torch
torch.set_num_threads(1)

def load_silero_vad(onnx=False):
    model_name = 'silero_vad.onnx' if onnx else 'silero_vad.jit'
    package_path = "silero_vad.data"
    
    try:
        import importlib_resources as impresources
        model_file_path = str(impresources.files(package_path).joinpath(model_name))
    except:
        from importlib import resources as impresources
        try:
            with impresources.path(package_path, model_name) as f:
                model_file_path = f
        except:
            model_file_path = str(impresources.files(package_path).joinpath(model_name))

    if onnx:
        model = OnnxWrapper(model_file_path, force_onnx_cpu=True)
    else:
        model = init_jit_model(model_file_path)
    
    return model
