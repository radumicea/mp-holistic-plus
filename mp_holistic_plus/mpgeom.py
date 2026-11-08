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

The face and hand crops are not MediaPipe's but cv2.warpAffine's (OpenCV 5), which
warp_affine reproduces on the GPU, also bit for bit.
"""
import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

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


@triton.jit
def _warp_affine(src, dst, frame, hw, mat, Hm, Wm, S: tl.constexpr, BLOCK: tl.constexpr):
    """BLOCK pixels of crop program_id(1). Every float step is explicitly rounded (*_rn), so
    the result does not depend on the compiler fusing multiplies and adds."""
    c = tl.program_id(1)
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = p < S * S
    y = p // S
    x = p - y * S
    k = tl.load(frame + c).to(tl.int64)
    H = tl.load(hw + 2 * c)
    W = tl.load(hw + 2 * c + 1)
    m0 = tl.load(mat + 6 * c)
    m1 = tl.load(mat + 6 * c + 1)
    m2 = tl.load(mat + 6 * c + 2)
    m3 = tl.load(mat + 6 * c + 3)
    m4 = tl.load(mat + 6 * c + 4)
    m5 = tl.load(mat + 6 * c + 5)
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    sx = libdevice.fma_rn(m0, xf, libdevice.add_rn(libdevice.mul_rn(yf, m1), m2))
    sy = libdevice.fma_rn(m3, xf, libdevice.add_rn(libdevice.mul_rn(yf, m4), m5))
    flx = libdevice.floor(sx)
    fly = libdevice.floor(sy)
    a = libdevice.sub_rn(sx, flx)
    b = libdevice.sub_rn(sy, fly)
    ix = tl.minimum(tl.maximum(flx, -1048576.0), 1048576.0).to(tl.int32)   # beyond: outside anyway
    iy = tl.minimum(tl.maximum(fly, -1048576.0), 1048576.0).to(tl.int32)
    in_x0 = (ix >= 0) & (ix < W)
    in_x1 = (ix + 1 >= 0) & (ix + 1 < W)
    in_y0 = (iy >= 0) & (iy < H) & valid
    in_y1 = (iy + 1 >= 0) & (iy + 1 < H) & valid
    o00 = ((k * Hm + iy) * Wm + ix) * 3
    o10 = o00 + Wm * 3
    out = ((c.to(tl.int64) * S + y) * S + x) * 3
    for ch in tl.static_range(3):
        p00 = tl.load(src + o00 + ch, mask=in_y0 & in_x0, other=0).to(tl.float32)
        p01 = tl.load(src + o00 + 3 + ch, mask=in_y0 & in_x1, other=0).to(tl.float32)
        p10 = tl.load(src + o10 + ch, mask=in_y1 & in_x0, other=0).to(tl.float32)
        p11 = tl.load(src + o10 + 3 + ch, mask=in_y1 & in_x1, other=0).to(tl.float32)
        f0 = libdevice.fma_rn(a, libdevice.sub_rn(p01, p00), p00)
        f1 = libdevice.fma_rn(a, libdevice.sub_rn(p11, p10), p10)
        r = libdevice.fma_rn(b, libdevice.sub_rn(f1, f0), f0)
        v = tl.minimum(tl.maximum(libdevice.float2int_rn(r), 0), 255)
        tl.store(dst + out + ch, v.to(tl.uint8), mask=valid)


def affine_inverse_f32(M):
    """cv::warpAffine's inverse of a 2x3 map (float64, in its order of operations), as float32."""
    M = [float(v) for v in np.asarray(M, np.float64).reshape(6)]
    D = M[0] * M[4] - M[1] * M[3]
    D = 1. / D if D != 0 else 0.
    A11, A22 = M[4] * D, M[0] * D
    M[0] = A11; M[1] *= -D
    M[3] *= -D; M[4] = A22
    b1 = -M[0] * M[2] - M[1] * M[5]
    b2 = -M[3] * M[2] - M[4] * M[5]
    M[2] = b1; M[5] = b2
    return np.array(M, f32)


def warp_affine(canvas, frames, src_hw, Ms, size):
    """cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR) of OpenCV 5, for many
    crops at once on the GPU. canvas: (N, Hmax, Wmax, 3) uint8 on the GPU, the frames top-left
    aligned; per crop: its frame's index in canvas, that frame's (H, W), and M (2x3, float64).
    Returns (n, size, size, 3) uint8 on the GPU.

    OpenCV inverts M in float64 and runs warpAffineLinearInvoker_8UC3 (warp_kernels.simd.hpp,
    ALGO_HINT_ACCURATE) on float32 copies of it: per pixel, sx = fma(M0, x, y * M1 + M2)
    (y * M1 + M2 rounded twice), sy likewise; the 2 x 2 source pixels from (floor(sx),
    floor(sy)), 0 outside the frame; a = sx - floor(sx), b likewise; f0 = fma(a, p01 - p00,
    p00), f1 = fma(a, p11 - p10, p10), then fma(b, f1 - f0, f0), rounded half to even."""
    n = len(Ms)
    dev = canvas.device
    out = torch.empty((n, size, size, 3), dtype=torch.uint8, device=dev)
    if n == 0:
        return out
    mat = torch.from_numpy(np.stack([affine_inverse_f32(M) for M in Ms])).to(dev)
    fr = torch.from_numpy(np.asarray(frames, np.int32)).to(dev)
    hw = torch.from_numpy(np.asarray(src_hw, np.int32).reshape(n, 2)).to(dev)
    canvas = canvas.contiguous()
    BLOCK = 1024
    _warp_affine[(triton.cdiv(size * size, BLOCK), n)](canvas, out, fr, hw, mat, canvas.shape[1], canvas.shape[2],
                                                      S=size, BLOCK=BLOCK)
    return out
