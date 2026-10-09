"""TFLite landmark model -> PyTorch -> ONNX -> TensorRT engine (true fp32: TF32 off), cached.

An engine is specific to the GPU and the TensorRT version, so the cache file name
carries both, plus the model's sha256. All engines of a process share one
activation buffer: the stages run one after another, never at the same time.
"""
import hashlib
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from .tflite_torch import TFLiteModule

LOG = trt.Logger(trt.Logger.ERROR)


class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return tuple(self.m(x))


def engine_path(cache_dir, tflite, max_batch):
    sha = hashlib.sha256(Path(tflite).read_bytes()).hexdigest()[:16]
    gpu = torch.cuda.get_device_name().replace(' ', '_')
    return Path(cache_dir) / f'{Path(tflite).stem}_{sha}_fp32_b{max_batch}_trt{trt.__version__}_{gpu}.engine'


def build(tflite, size, cache_dir, drop=(), max_batch=64, opt_batch=32):
    """The engine for `tflite` (input size x size x 3), built on first use."""
    path = engine_path(cache_dir, tflite, max_batch)
    mod = TFLiteModule(tflite, drop_outputs=drop)
    names = [mod.names[i] for i in mod.outputs]
    if not path.exists():
        print(f'building the TensorRT engine for {Path(tflite).name} (once, a few minutes) ...', flush=True)
        dev = torch.device('cuda')
        mod = mod.to(dev).eval()
        onnx_path = path.with_suffix('.onnx')
        with torch.inference_mode():
            torch.onnx.export(_Wrap(mod), (torch.zeros(2, size, size, 3, device=dev),), str(onnx_path), input_names=['x'],
                              output_names=names, dynamic_axes={'x': {0: 'n'}, **{n: {0: 'n'} for n in names}},
                              opset_version=17, dynamo=False)
        b = trt.Builder(LOG)
        net = b.create_network(0)
        parser = trt.OnnxParser(net, LOG)
        if not parser.parse(onnx_path.read_bytes()):
            raise RuntimeError(f'ONNX parse failed: {[parser.get_error(i) for i in range(parser.num_errors)]}')
        cfg = b.create_builder_config()
        cfg.clear_flag(trt.BuilderFlag.TF32)                     # true fp32
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        prof = b.create_optimization_profile()
        prof.set_shape('x', (1, size, size, 3), (opt_batch, size, size, 3), (max_batch, size, size, 3))
        cfg.add_optimization_profile(prof)
        blob = b.build_serialized_network(net, cfg)
        if blob is None:
            raise RuntimeError(f'TensorRT could not build {tflite}')
        tmp = path.with_suffix('.part')
        tmp.write_bytes(bytes(blob))
        tmp.replace(path)
        onnx_path.unlink(missing_ok=True)
    blob = path.read_bytes()
    engine = trt.Runtime(LOG).deserialize_cuda_engine(blob)
    return Engine(engine, names, max_batch)


class SharedScratch:
    """One activation buffer for every engine of the process."""
    buf = None
    ctxs = []

    @classmethod
    def attach(cls, ctx, size):
        if cls.buf is None or cls.buf.numel() < size:
            cls.buf = torch.empty(size, dtype=torch.uint8, device='cuda')
            for c in cls.ctxs:
                c.set_device_memory(cls.buf.data_ptr(), cls.buf.numel())
        ctx.set_device_memory(cls.buf.data_ptr(), cls.buf.numel())
        cls.ctxs.append(ctx)


class Engine:
    """A TensorRT engine: __call__(x) -> {output name: tensor}."""
    def __init__(self, engine, names, max_batch):
        self.engine, self.names, self.max_batch = engine, names, max_batch
        self.ctx = engine.create_execution_context_without_device_memory()
        if self.ctx is None:
            raise RuntimeError('TensorRT could not create an execution context (GPU memory?)')
        SharedScratch.attach(self.ctx, engine.device_memory_size_v2)

    def __call__(self, x):
        """x: (N, H, W, 3) float32 on the GPU, N <= max_batch -> {output name: tensor}."""
        x = x.contiguous()
        assert x.shape[0] <= self.max_batch, (x.shape, self.max_batch)
        self.ctx.set_input_shape('x', tuple(x.shape))
        self.ctx.set_tensor_address('x', x.data_ptr())
        outs = {}
        for n in self.names:
            outs[n] = torch.empty(tuple(self.ctx.get_tensor_shape(n)), device=x.device)
            self.ctx.set_tensor_address(n, outs[n].data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return outs


class LiteRTEngine:
    """The same interface on the CPU via LiteRT, one crop at a time. LiteRT is what MediaPipe
    runs its models with, so the results are MediaPipe's to the bit; for checking exactness."""
    def __init__(self, tflite):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(tflite), num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]['index']
        self.outs = {d['name']: d['index'] for d in self.it.get_output_details()}

    def __call__(self, x):
        res = {k: [] for k in self.outs}
        for img in x.cpu().numpy():
            self.it.set_tensor(self.inp, img[None])
            self.it.invoke()
            for k, i in self.outs.items():
                res[k].append(self.it.get_tensor(i).copy())
        return {k: torch.from_numpy(np.concatenate(v)).cuda() for k, v in res.items()}
