"""The image geometry of MediaPipe's CPU path, reproduced bit for bit.

MediaPipe crops a rotated box out of the frame with OpenCV's warpPerspective.
- roi_matrix: the map warpPerspective gets, built as MediaPipe and OpenCV build it
  (cv::RotatedRect::points, cv::getPerspectiveTransform by LU, the 3x3 inverse).
- warp_gpu: OpenCV's INTER_LINEAR warpPerspective on 8-bit RGB, in torch on the GPU:
  coordinates in float64 in OpenCV's blocks, quantized to 1/32 px, then integer
  bilinear weights and rounding. Integer arithmetic, so the GPU gives the same bytes.
- to_float: the crop to float32, as MediaPipe converts it.
- expf, sigmoid_f, cosf, sinf, atan2f: float32 functions, computed in float64 and
  rounded, which gives glibc's correctly rounded results.
"""
import math

import numpy as np
import torch

f32 = np.float32


def expf(v):
    return f32(math.exp(float(v)))


def sigmoid_f(v):
    """1.0f / (1.0f + std::exp(-v)) on arrays, with a correctly rounded expf."""
    v = np.asarray(v, f32)
    with np.errstate(over='ignore'):           # exp overflows to inf, as expf does: sigmoid 0
        e = np.exp(-v.astype(np.float64)).astype(f32)
    return (f32(1.0) / (f32(1.0) + e)).astype(f32)


def cosf(a):
    return f32(math.cos(float(a)))


def sinf(a):
    return f32(math.sin(float(a)))


def atan2f(y, x):
    return f32(math.atan2(float(y), float(x)))


def norm_rad(a):
    a = float(a)
    return f32(a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi)))


def box_points(cx, cy, w, h, rot):
    """cv::RotatedRect::points of a box in pixels (rotation in radians)."""
    ang = f32(float(f32(f32(rot) * f32(180.0))) / math.pi)      # rotation*180.f (float) / M_PI (double)
    _a = float(ang) * math.pi / 180.0
    b = f32(f32(math.cos(_a)) * f32(0.5)); a = f32(f32(math.sin(_a)) * f32(0.5))
    cx, cy, w, h = f32(cx), f32(cy), f32(w), f32(h)
    ah, aw, bh, bw = a * h, a * w, b * h, b * w
    return [(cx - ah - bw, cy + bh - aw), (cx + ah - bw, cy - bh - aw), (cx + ah + bw, cy - bh + aw), (cx - ah + bw, cy + bh + aw)]


def _lu_solve(A, b):
    A = [row[:] for row in A]; b = b[:]; m = len(A)
    for i in range(m):
        k = i
        for j in range(i + 1, m):
            if abs(A[j][i]) > abs(A[k][i]):
                k = j
        if k != i:
            A[i], A[k] = A[k], A[i]; b[i], b[k] = b[k], b[i]
        d = -1.0 / A[i][i]
        for j in range(i + 1, m):
            alpha = A[j][i] * d
            for kk in range(i + 1, m):
                A[j][kk] += alpha * A[i][kk]
            b[j] += alpha * b[i]
    for i in range(m - 1, -1, -1):
        s = b[i]
        for kk in range(i + 1, m):
            s -= A[i][kk] * b[kk]
        b[i] = s / A[i][i]
    return b


def perspective(src, dst):
    A = [[0.0] * 8 for _ in range(8)]; B = [0.0] * 8
    for i in range(4):
        x, y = float(src[i][0]), float(src[i][1]); u, v = float(dst[i][0]), float(dst[i][1])
        A[i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u]; A[i + 4] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v]
        B[i] = u; B[i + 4] = v
    X = _lu_solve(A, B)
    return np.array(X + [1.0]).reshape(3, 3)


def inv3(S):
    S = [[float(S[i][j]) for j in range(3)] for i in range(3)]
    d = (S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
         + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]))
    d = 1.0 / d
    t = [(S[1][1] * S[2][2] - S[1][2] * S[2][1]) * d, (S[0][2] * S[2][1] - S[0][1] * S[2][2]) * d, (S[0][1] * S[1][2] - S[0][2] * S[1][1]) * d,
         (S[1][2] * S[2][0] - S[1][0] * S[2][2]) * d, (S[0][0] * S[2][2] - S[0][2] * S[2][0]) * d, (S[0][2] * S[1][0] - S[0][0] * S[1][2]) * d,
         (S[1][0] * S[2][1] - S[1][1] * S[2][0]) * d, (S[0][1] * S[2][0] - S[0][0] * S[2][1]) * d, (S[0][0] * S[1][1] - S[0][1] * S[1][0]) * d]
    return np.array(t)


def roi_matrix(roi_px, ow, oh):
    """Pixel-space RotatedRect (cx, cy, w, h, rot) -> the inverse map (9 doubles) warpPerspective applies."""
    src = box_points(*roi_px)
    dst = [(0.0, float(oh)), (0.0, 0.0), (float(ow), 0.0), (float(ow), float(oh))]
    return inv3(perspective(src, dst))


def _itab(device):
    t = torch.zeros((32, 32, 4), dtype=torch.int32)
    for fy in range(32):
        for fx in range(32):
            t[fy, fx] = torch.tensor([(32 - fx) * (32 - fy), fx * (32 - fy), (32 - fx) * fy, fx * fy], dtype=torch.int32) * 32
    t[0, 0] = torch.tensor([32767, 0, 0, 1], dtype=torch.int32)
    return t.reshape(1024, 4).to(device)


_ITAB = {}


def warp_gpu(canvas, src_hw, Ms, ow, oh, border):
    """canvas: (N, Hmax, Wmax, 3) uint8 on the GPU, one source image per crop (top-left aligned);
    src_hw: (N, 2) int (H, W) of each source; Ms: (N, 9) float64 inverse maps.
    Returns (N, oh, ow, 3) uint8, identical to cv::warpPerspective(INTER_LINEAR, border).
    Coordinates in float64 (OpenCV's block scheme), everything after in int32: the bilinear
    sum is at most 255 * 32768, so it is exact."""
    dev = canvas.device
    if dev not in _ITAB:
        _ITAB[dev] = _itab(dev)
    N, Hm, Wm, _ = canvas.shape
    M = torch.as_tensor(np.asarray(Ms, np.float64), device=dev)
    xs = torch.arange(ow, device=dev); ys = torch.arange(oh, device=dev)
    bw0 = min(1024 // min(16, oh), ow)
    xb = (xs // bw0) * bw0; x1 = xs - xb
    Y = ys[None, :, None].double(); XB = xb[None, None, :].double(); X1 = x1[None, None, :].double()
    m = [M[:, k][:, None, None] for k in range(9)]
    X0 = m[0] * XB + m[1] * Y + m[2]; Y0 = m[3] * XB + m[4] * Y + m[5]; W0 = m[6] * XB + m[7] * Y + m[8]
    Wd = W0 + m[6] * X1
    nz = Wd != 0
    Wd = torch.where(nz, 32.0 / torch.where(nz, Wd, torch.ones_like(Wd)), torch.zeros_like(Wd))
    fX = torch.clamp((X0 + m[0] * X1) * Wd, -2 ** 31, 2 ** 31 - 1); fY = torch.clamp((Y0 + m[3] * X1) * Wd, -2 ** 31, 2 ** 31 - 1)
    Xi = torch.round(fX).to(torch.int64); Yi = torch.round(fY).to(torch.int64)          # half-to-even, like rint
    sx = torch.clamp(Xi >> 5, -32768, 32767); sy = torch.clamp(Yi >> 5, -32768, 32767)
    wt = _ITAB[dev][((Yi & 31) * 32 + (Xi & 31))]                                        # (N, oh, ow, 4) int32
    Hs = torch.as_tensor(np.asarray(src_hw)[:, 0], device=dev)[:, None, None]
    Ws = torch.as_tensor(np.asarray(src_hw)[:, 1], device=dev)[:, None, None]
    base = (torch.arange(N, device=dev) * (Hm * Wm))[:, None, None]
    flat = canvas.reshape(-1, 3)

    def fetch(xx, yy):
        xc = torch.minimum(torch.clamp(xx, min=0), Ws - 1); yc = torch.minimum(torch.clamp(yy, min=0), Hs - 1)
        v = flat[base + yc * Wm + xc].to(torch.int32)
        if border == 'replicate':
            return v
        ok = (xx >= 0) & (xx < Ws) & (yy >= 0) & (yy < Hs)
        return v * ok[..., None]

    acc = (fetch(sx, sy) * wt[..., 0:1] + fetch(sx + 1, sy) * wt[..., 1:2]
           + fetch(sx, sy + 1) * wt[..., 2:3] + fetch(sx + 1, sy + 1) * wt[..., 3:4])
    out = torch.clamp((acc + 16384) >> 15, 0, 255)
    if border == 'const':
        allout = (sx >= Ws) | (sx + 1 < 0) | (sy >= Hs) | (sy + 1 < 0)
        out = out * ~allout[..., None]
    return out.to(torch.uint8)


def to_float(u8, lo, hi):
    """convertTo(scale, offset) to float32, as two roundings (u8*scale, then +offset)."""
    scale = f32(f32(hi - lo) / f32(255.0))
    return u8.float() * float(scale) + float(f32(lo))
