"""Run a TFLite landmark model in PyTorch (CUDA, batched).

Covers the ops of MediaPipe's pose, face and hand landmark models. Activations
are kept NCHW; outputs come back in TFLite's layout (NHWC for 4-D tensors).
"""
import numpy as np
import torch
import torch.nn.functional as F

from ai_edge_litert import schema_py_generated as S

OPS = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith('_')}
DTYPES = {S.TensorType.FLOAT32: np.float32, S.TensorType.FLOAT16: np.float16,
          S.TensorType.INT32: np.int32, S.TensorType.INT64: np.int64}
SAME = S.Padding.SAME
ACT = {S.ActivationFunctionType.NONE: lambda x: x,
       S.ActivationFunctionType.RELU: F.relu,
       S.ActivationFunctionType.RELU6: lambda x: torch.clamp(x, 0, 6)}


def same_pad(size, k, s, d=1):
    """TensorFlow SAME padding for one spatial dim -> (before, after)."""
    out = -(-size // s)
    total = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return total // 2, total - total // 2


class TFLiteModule(torch.nn.Module):
    def __init__(self, path, drop_outputs=()):
        super().__init__()
        m = S.Model.GetRootAsModel(open(path, 'rb').read(), 0)
        g = m.Subgraphs(0)
        self.shapes = {i: tuple(int(v) for v in g.Tensors(i).ShapeAsNumpy()) if g.Tensors(i).ShapeLength() else ()
                       for i in range(g.TensorsLength())}
        self.names = {i: g.Tensors(i).Name().decode() for i in range(g.TensorsLength())}
        consts = {}
        for i in range(g.TensorsLength()):
            t = g.Tensors(i)
            b = m.Buffers(t.Buffer())
            if b is not None and b.DataLength():
                consts[i] = np.frombuffer(b.DataAsNumpy().tobytes(), DTYPES[t.Type()]).reshape(self.shapes[i])
        self.inputs = [g.Inputs(j) for j in range(g.InputsLength())]
        self.outputs = [g.Outputs(j) for j in range(g.OutputsLength())]
        self.steps, self.params = [], torch.nn.ParameterList()

        def param(a):
            self.params.append(torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(a, np.float32).copy()), requires_grad=False))
            return len(self.params) - 1

        for k in range(g.OperatorsLength()):
            op = g.Operators(k)
            code = m.OperatorCodes(op.OpcodeIndex())
            name = OPS[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            out = op.Outputs(0)
            o = op.BuiltinOptions()
            if name == 'DEQUANTIZE':  # fp16 constant weights: fold now
                consts[out] = consts[ins[0]].astype(np.float32)
                continue
            if name in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
                x = (S.Conv2DOptions if name == 'CONV_2D' else S.DepthwiseConv2DOptions)()
                x.Init(o.Bytes, o.Pos)
                w = consts[ins[1]]
                if name == 'CONV_2D':
                    w, groups = w.transpose(0, 3, 1, 2), 1                      # O,kh,kw,I -> O,I,kh,kw
                else:
                    w = w.transpose(3, 0, 1, 2)                                 # 1,kh,kw,C -> C,1,kh,kw
                    groups = self.shapes[ins[0]][3]
                b = consts[ins[2]] if len(ins) > 2 and ins[2] >= 0 else np.zeros(w.shape[0], np.float32)
                _, h, wd, _ = self.shapes[ins[0]]
                kh, kw = w.shape[2:]
                pad = (0, 0, 0, 0)
                if x.Padding() == SAME:
                    t, bt = same_pad(h, kh, x.StrideH(), x.DilationHFactor())
                    l, r = same_pad(wd, kw, x.StrideW(), x.DilationWFactor())
                    pad = (l, r, t, bt)
                self.steps.append(('conv', ins[0], out, param(w), param(b), pad, (x.StrideH(), x.StrideW()),
                                   (x.DilationHFactor(), x.DilationWFactor()), groups, x.FusedActivationFunction()))
            elif name == 'ADD':
                x = S.AddOptions(); x.Init(o.Bytes, o.Pos)
                srcs = [('const', param(self._nchw_const(consts[i], self.shapes[out]))) if i in consts else ('t', i) for i in ins]
                self.steps.append(('add', srcs, out, x.FusedActivationFunction()))
            elif name == 'PAD':
                p = consts[ins[1]]                                              # NHWC [[n],[h],[w],[c]]
                self.steps.append(('pad', ins[0], out, (int(p[2, 0]), int(p[2, 1]), int(p[1, 0]), int(p[1, 1]), int(p[3, 0]), int(p[3, 1]))))
            elif name == 'RESIZE_BILINEAR':
                x = S.ResizeBilinearOptions(); x.Init(o.Bytes, o.Pos)
                assert x.HalfPixelCenters() and not x.AlignCorners()
                self.steps.append(('resize', ins[0], out, tuple(int(v) for v in consts[ins[1]])))
            elif name == 'MAX_POOL_2D':
                x = S.Pool2DOptions(); x.Init(o.Bytes, o.Pos)
                assert x.Padding() == S.Padding.VALID and x.FusedActivationFunction() == 0
                self.steps.append(('maxpool', ins[0], out, (x.FilterHeight(), x.FilterWidth()), (x.StrideH(), x.StrideW())))
            elif name == 'RESHAPE':
                self.steps.append(('reshape', ins[0], out, self.shapes[out]))
            elif name == 'LOGISTIC':
                self.steps.append(('sigmoid', ins[0], out))
            elif name == 'PRELU':
                self.steps.append(('prelu', ins[0], out, param(self._nchw_const(consts[ins[1]], self.shapes[out]))))
            elif name == 'FULLY_CONNECTED':
                x = S.FullyConnectedOptions(); x.Init(o.Bytes, o.Pos)
                b = consts[ins[2]] if len(ins) > 2 and ins[2] >= 0 else None
                self.steps.append(('fc', ins[0], out, param(consts[ins[1]]), None if b is None else param(b), x.FusedActivationFunction()))
            elif name == 'MEAN':
                axes = tuple(int(a) for a in np.atleast_1d(consts[ins[1]]))
                x = S.ReducerOptions(); x.Init(o.Bytes, o.Pos)
                self.steps.append(('mean', ins[0], out, axes, x.KeepDims()))
            else:
                raise NotImplementedError(name)
        # drop unwanted outputs and every step that only feeds them
        if drop_outputs:
            self.outputs = [i for i in self.outputs if self.names[i] not in drop_outputs]
            need, kept = set(self.outputs), []
            for st in reversed(self.steps):
                if st[2] in need:
                    kept.append(st)
                    need |= {j for t, j in st[1] if t == 't'} if st[0] == 'add' else {st[1]}
            self.steps = kept[::-1]
        # free each activation after its last use
        last = {}
        for k, st in enumerate(self.steps):
            used = [j for t, j in st[1] if t == 't'] if st[0] == 'add' else [st[1]]
            for j in used:
                last[j] = k
        self.free_after = [[] for _ in self.steps]
        for j, k in last.items():
            if j not in self.outputs:
                self.free_after[k].append(j)

    @staticmethod
    def _nchw_const(a, out_shape):
        a = np.asarray(a, np.float32)
        if len(out_shape) == 4:                                                 # broadcast over NHWC -> NCHW
            a = np.broadcast_to(a, (1,) * (4 - a.ndim) + a.shape) if a.ndim < 4 else a
            return a.transpose(0, 3, 1, 2)
        return a

    def forward(self, x):
        """x: (N, H, W, C) as the TFLite input. Returns the outputs, in TFLite's layout."""
        v = {self.inputs[0]: x.permute(0, 3, 1, 2)}
        P = self.params
        for k, s in enumerate(self.steps):
            kind = s[0]
            if kind == 'conv':
                _, i, o, w, b, pad, st, dil, groups, act = s
                y = F.pad(v[i], pad) if any(pad) else v[i]
                v[o] = ACT[act](F.conv2d(y, P[w], P[b], st, 0, dil, groups))
            elif kind == 'add':
                _, srcs, o, act = s
                a, b = (P[j] if t == 'const' else v[j] for t, j in srcs)
                v[o] = ACT[act](a + b)
            elif kind == 'pad':
                v[s[2]] = F.pad(v[s[1]], s[3])
            elif kind == 'resize':
                v[s[2]] = F.interpolate(v[s[1]], size=s[3], mode='bilinear', align_corners=False)
            elif kind == 'maxpool':
                v[s[2]] = F.max_pool2d(v[s[1]], s[3], s[4])
            elif kind == 'reshape':
                y = v[s[1]]
                if y.dim() == 4:
                    y = y.permute(0, 2, 3, 1)
                v[s[2]] = y.reshape(y.shape[0], *s[3][1:])
            elif kind == 'sigmoid':
                v[s[2]] = torch.sigmoid(v[s[1]])
            elif kind == 'prelu':
                y = v[s[1]]
                v[s[2]] = torch.where(y >= 0, y, y * P[s[3]])
            elif kind == 'fc':
                _, i, o, w, b, act = s
                y = v[i]
                if y.dim() == 4:
                    y = y.permute(0, 2, 3, 1)
                y = y.reshape(y.shape[0], -1)
                v[o] = ACT[act](F.linear(y, P[w], None if b is None else P[b]))
            elif kind == 'mean':
                _, i, o, axes, keep = s
                y = v[i]
                if y.dim() == 4:
                    axes = tuple({0: 0, 1: 2, 2: 3, 3: 1}[a] for a in axes)
                y = y.mean(dim=axes, keepdim=bool(keep))
                v[o] = y
            for j in self.free_after[k]:
                del v[j]
        outs = []
        for i in self.outputs:
            y = v[i]
            outs.append(y.permute(0, 2, 3, 1) if y.dim() == 4 else y)
        return outs
