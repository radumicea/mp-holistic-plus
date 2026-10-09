"""The pose: MediaPipe's PoseLandmarker (heavy model, VIDEO mode, one pose) for many
videos at once.

Per video: the pose detector (LiteRT, CPU) when there is no track, the landmark model
on a 256 x 256 crop of the pose box (batched on the GPU), heatmap refinement, the
mapping back to the frame, the presence check, the next frame's box from the
landmarks, and the One Euro and low-pass filters.

Written from MediaPipe's C++ source: the crop is OpenCV's fixed-point warp redone on
the GPU (mpgeom.warp_gpu), and every scalar step keeps MediaPipe's float32/float64
mix. With the model run by LiteRT the output was checked byte-identical to mediapipe
1.0.1's PoseLandmarker on the CPU; with TensorRT only the model's rounding differs.
"""
import numpy as np
import torch

from . import mpgeom as G

f32 = np.float32
TARGET_90 = f32(np.pi * 90 / f32(180.0))         # MediaPipe's float target_angle_ = M_PI * 90 / 180.f


# --------------------------------------------------------------------------- #
# Rects
# --------------------------------------------------------------------------- #

def points_to_rect(p0, p1, W, H):
    """AlignmentPointsRectsCalculator: centre p0, size 2*|p1-p0| (pixels), rotation to p1 at 90 deg."""
    x0, y0 = f32(p0[0]) * f32(W), f32(p0[1]) * f32(H)
    x1, y1 = f32(p1[0]) * f32(W), f32(p1[1]) * f32(H)
    box = f32(float(np.sqrt(f32((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)))) * 2.0)
    rot = G.norm_rad(f32(TARGET_90 - G.atan2f(-(y1 - y0), x1 - x0)))
    return (x0 / f32(W), y0 / f32(H), box / f32(W), box / f32(H), rot)


def expand(r, W, H, scale=1.25):
    """RectTransformationCalculator: square_long, scale 1.25, no shift; rotation kept."""
    xc, yc, w, h, rot = r
    long_side = max(f32(w * f32(W)), f32(h * f32(H)))
    w, h = long_side / f32(W), long_side / f32(H)
    return (xc, yc, f32(w * f32(scale)), f32(h * f32(scale)), rot)


def iou_rect(a, b):
    """Axis-aligned IoU of two NormalizedRects (rotation ignored)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(f32(0), min(ax1, bx1) - max(ax0, bx0)); iy = max(f32(0), min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else f32(0)


def roi_px(roi, W, H):
    """GetRoi: NormalizedRect -> pixel RotatedRect (float32)."""
    xc, yc, w, h, rot = roi
    return (f32(xc) * f32(W), f32(yc) * f32(H), f32(w) * f32(W), f32(h) * f32(H), f32(rot))


def letterbox_full(W, H, size):
    """GetRoi(no rect) + PadRoi(keep_aspect_ratio) for a size x size tensor -> (roi_px, [l, t, r, b])."""
    cx, cy, w, h = f32(0.5) * f32(W), f32(0.5) * f32(H), f32(W), f32(H)
    r_t = f32(size) / f32(size); r_roi = h / w
    if r_t > r_roi:
        h = w * r_t; pv = (f32(1.0) - r_roi / r_t) / f32(2.0); pad = (f32(0), pv, f32(0), pv)
    else:
        w = h / r_t; ph = (f32(1.0) - r_t / r_roi) / f32(2.0); pad = (ph, f32(0), ph, f32(0))
    return (cx, cy, w, h, f32(0)), pad


# --------------------------------------------------------------------------- #
# Filters (MediaPipe's util/filtering, with its float/double mix)
# --------------------------------------------------------------------------- #

class LowPass:
    """Vector of LowPassFilter (float members)."""
    def __init__(self, n, alpha):
        self.alpha = np.full(n, f32(alpha), f32)
        self.raw = np.zeros(n, f32); self.stored = np.zeros(n, f32); self.init = False

    def apply_with_alpha(self, value, alpha):
        alpha = np.asarray(alpha, f32)
        ok = (alpha >= 0) & (alpha <= 1)
        self.alpha = np.where(ok, alpha, self.alpha).astype(f32)
        return self.apply(value)

    def apply(self, value):
        value = np.asarray(value, f32)
        if self.init:
            res = ((self.alpha * value).astype(f32).astype(np.float64)
                   + (1.0 - self.alpha.astype(np.float64)) * self.stored.astype(np.float64)).astype(f32)
        else:
            res = value.copy(); self.init = True
        self.raw = value.copy(); self.stored = res
        return res


class OneEuro:
    """Vector of OneEuroFilter sharing one timestamp (double members, float LPFs)."""
    def __init__(self, n, min_cutoff, beta, derivate_cutoff, frequency=30.0):
        self.n = n
        self.min_cutoff, self.beta, self.dc = float(f32(min_cutoff)), float(f32(beta)), float(f32(derivate_cutoff))
        self.freq0 = float(f32(frequency))
        self.reset()

    def reset(self):
        self.freq = self.freq0
        self.last_time = -1
        self.x = LowPass(self.n, self._alpha(self.min_cutoff))
        self.dx = LowPass(self.n, self._alpha(self.dc))

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * np.asarray(cutoff, np.float64))
        return 1.0 / (1.0 + tau / te)

    def apply(self, t_ns, value, value_scale):
        value = np.asarray(value, f32)
        if self.last_time >= t_ns:
            return value
        if self.last_time != 0 and t_ns != 0:
            self.freq = 1.0 / ((t_ns - self.last_time) * 1e-9)
        self.last_time = t_ns
        v64 = value.astype(np.float64)
        dvalue = (v64 - self.x.raw.astype(np.float64)) * float(value_scale) * self.freq if self.x.init else np.zeros(self.n)
        edvalue = self.dx.apply_with_alpha(dvalue.astype(f32), f32(self._alpha(self.dc)))
        cutoff = self.min_cutoff + self.beta * np.abs(edvalue.astype(np.float64))
        return self.x.apply_with_alpha(value, self._alpha(cutoff).astype(f32))


# --------------------------------------------------------------------------- #
# Detector (only on frames with nothing to track; LiteRT, CPU)
# --------------------------------------------------------------------------- #

def ssd_anchors():
    """Pose detector anchors: 224x224, strides 8,16,32,32,32, 2 per layer, fixed size 1."""
    anchors, strides, n, layer = [], [8, 16, 32, 32, 32], 5, 0
    while layer < n:
        last, count = layer, 0
        while last < n and strides[last] == strides[layer]:
            count += 2
            last += 1
        fm = int(np.ceil(f32(224) / f32(strides[layer])))
        for y in range(fm):
            for x in range(fm):
                anchors.extend([(f32((x + f32(0.5)) / f32(fm)), f32((y + f32(0.5)) / f32(fm)))] * count)
        layer = last
    return np.array(anchors, f32)


class PoseDetector:
    """BlazePose's detector (224 x 224, LiteRT, CPU) -> the pose box tracking starts from."""
    def __init__(self, tflite_path, min_det=0.5, nms=0.5, track_iou=0.5):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(tflite_path), num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]['index']
        outs = self.it.get_output_details()
        self.o_box = [d['index'] for d in outs if tuple(d['shape']) == (1, 2254, 12)][0]
        self.o_score = [d['index'] for d in outs if tuple(d['shape']) == (1, 2254, 1)][0]
        self.anchors = ssd_anchors()
        assert len(self.anchors) == 2254
        self.min_det, self.nms, self.track_iou = min_det, nms, track_iou

    def __call__(self, canvas, i, H, W):
        """Detect on source i of the GPU canvas -> the ROI MediaPipe starts tracking from, or None."""
        rp, pad = letterbox_full(W, H, 224)
        u8 = G.warp_gpu(canvas[i:i + 1], [(H, W)], [G.roi_matrix(rp, 224, 224)], 224, 224, 'const')
        x = G.to_float(u8, -1.0, 1.0).cpu().numpy()
        self.it.set_tensor(self.inp, x); self.it.invoke()
        raw = self.it.get_tensor(self.o_box)[0].astype(f32)
        score = G.sigmoid_f(np.clip(self.it.get_tensor(self.o_score)[0, :, 0], f32(-100), f32(100)))
        ax, ay = self.anchors[:, 0], self.anchors[:, 1]
        s224 = f32(224)
        xc = raw[:, 0] / s224 + ax; yc = raw[:, 1] / s224 + ay
        w = raw[:, 2] / s224; h = raw[:, 3] / s224
        ymin, xmin, ymax, xmax = yc - h / f32(2), xc - w / f32(2), yc + h / f32(2), xc + w / f32(2)
        bw, bh = xmax - xmin, ymax - ymin
        kx = raw[:, 4:12:2] / s224 + ax[:, None]; ky = raw[:, 5:12:2] / s224 + ay[:, None]
        keep = ~(score < f32(self.min_det)) & (bw >= 0) & (bh >= 0) & ~np.isnan(bw) & ~np.isnan(bh)
        idx = np.where(keep)[0]
        if len(idx) == 0:
            return None
        dets = self._weighted_nms(idx, score, xmin, ymin, bw, bh, kx, ky)
        l, t, r, b = pad
        rects = []
        for (_, _, _, _, _, kpx, kpy) in dets:
            kpx = (kpx - l) / (f32(1.0) - (l + r)); kpy = (kpy - t) / (f32(1.0) - (t + b))
            rects.append(expand(points_to_rect((kpx[0], kpy[0]), (kpx[1], kpy[1]), W, H), W, H))
        L = [rects[0]]
        for rj in rects[1:]:
            L = [e for e in L if not (iou_rect(rj, e) > f32(self.track_iou))] + [rj]
        return L[0]

    def _weighted_nms(self, idx, score, xmin, ymin, bw, bh, kx, ky):
        S = sorted(idx.tolist(), key=lambda i: -score[i])
        out = []

        def iou(i, j):
            x0, y0 = max(xmin[i], xmin[j]), max(ymin[i], ymin[j])
            x1, y1 = min(xmin[i] + bw[i], xmin[j] + bw[j]), min(ymin[i] + bh[i], ymin[j] + bh[j])
            if x1 < x0 or y1 < y0:
                return f32(0)
            inter = (x1 - x0) * (y1 - y0)
            den = bw[i] * bh[i] + bw[j] * bh[j] - inter
            return inter / den if den > 0 else f32(0)

        while S:
            d = S[0]
            cand = [j for j in S if iou(j, d) > f32(self.nms)]
            rest = [j for j in S if not (iou(j, d) > f32(self.nms))]
            T = f32(0); sx0 = sy0 = sx1 = sy1 = f32(0); skx = np.zeros(4, f32); sky = np.zeros(4, f32)
            for j in cand:
                s = score[j]; T = f32(T + s)
                sx0 = f32(sx0 + xmin[j] * s); sy0 = f32(sy0 + ymin[j] * s)
                sx1 = f32(sx1 + (xmin[j] + bw[j]) * s); sy1 = f32(sy1 + (ymin[j] + bh[j]) * s)
                skx = (skx + kx[j] * s).astype(f32); sky = (sky + ky[j] * s).astype(f32)
            x0, y0 = sx0 / T, sy0 / T
            out.append((score[d], x0, y0, sx1 / T - x0, sy1 / T - y0, (skx / T).astype(f32), (sky / T).astype(f32)))
            if len(rest) == len(S):
                break
            S = rest
        return out


# --------------------------------------------------------------------------- #
# Heatmap refinement (exact summation order)
# --------------------------------------------------------------------------- #

def refine_from_heatmap(xn, yn, windows, cols, rows, hm=64, min_conf=0.5):
    """xn, yn: (N, 39) float32 (crop-normalized); windows: (N, 39, 7, 7) raw heatmap values around
    (rows, cols) = the truncated centres, with NaN outside the heatmap. Row-major float32 sums."""
    N, L = xn.shape
    inside = (cols >= 0) & (cols < hm) & (rows >= 0) & (rows < hm)
    s = np.zeros((N, L), f32); mx = np.zeros((N, L), f32); wc = np.zeros((N, L), f32); wr = np.zeros((N, L), f32)
    for dr in range(7):
        for dc in range(7):
            v = windows[:, :, dr, dc]
            ok = ~np.isnan(v)
            conf = np.where(ok, G.sigmoid_f(np.where(ok, v, f32(0))), f32(0)).astype(f32)
            r = (rows + (dr - 3)).astype(f32); c = (cols + (dc - 3)).astype(f32)
            s = np.where(ok, s + conf, s).astype(f32)
            mx = np.where(ok, np.maximum(mx, conf), mx).astype(f32)
            wc = np.where(ok, wc + (c * conf).astype(f32), wc).astype(f32)
            wr = np.where(ok, wr + (r * conf).astype(f32), wr).astype(f32)
    ok = inside & (mx >= f32(min_conf)) & (s > 0)
    xr = np.where(ok, (wc / f32(hm) / s).astype(f32), xn).astype(f32)
    yr = np.where(ok, (wr / f32(hm) / s).astype(f32), yn).astype(f32)
    return xr, yr


def heat_windows(heat, xn, yn, hm=64, k=7):
    """Gather the k x k windows of each landmark's heatmap channel on the GPU (exact copies)."""
    xs = torch.as_tensor(xn, device=heat.device); ys = torch.as_tensor(yn, device=heat.device)
    cols = torch.trunc(xs * hm).to(torch.int64); rows = torch.trunc(ys * hm).to(torch.int64)  # float->int truncation
    N, L = cols.shape
    d = torch.arange(-(k // 2), k // 2 + 1, device=heat.device)
    rr = rows[..., None, None] + d[:, None]; cc = cols[..., None, None] + d[None, :]
    valid = (rr >= 0) & (rr < hm) & (cc >= 0) & (cc < hm)
    n = torch.arange(N, device=heat.device)[:, None, None, None]; l = torch.arange(L, device=heat.device)[None, :, None, None]
    v = heat[n, rr.clamp(0, hm - 1), cc.clamp(0, hm - 1), l]
    v = torch.where(valid, v, torch.full_like(v, float('nan')))
    return v.cpu().numpy(), cols.cpu().numpy(), rows.cpu().numpy()


# --------------------------------------------------------------------------- #
# Per-video tracking state, and the batched step
# --------------------------------------------------------------------------- #

class PoseTrack:
    """One video's pose tracking state."""
    def __init__(self):
        self.prev_roi = None                   # the next frame's pose box; None: detect
        self.vis_norm = LowPass(33, 0.1)
        self.euro_norm = OneEuro(33 * 3, 0.05, 80.0, 1.0)
        self.euro_world = OneEuro(33 * 3, 0.1, 40.0, 1.0)


class PoseStage:
    def __init__(self, engine, detector, min_presence=0.5):
        self.engine, self.detector, self.min_presence = engine, detector, min_presence

    def run(self, tracks, canvas, sizes, t_ns):
        """One frame of each video (canvas: the frames on the GPU; sizes: (H, W) each; t_ns:
        timestamps). Returns per video None (no pose) or a dict of pose_xyz (normalized),
        pose_vis, pose_pres and pose_world (metres), as float32."""
        n = len(tracks)
        rois = [None] * n
        for i, tr in enumerate(tracks):
            if tr.prev_roi is not None:
                rois[i] = tr.prev_roi
            else:
                rois[i] = self.detector(canvas, i, *sizes[i])
                if rois[i] is None:
                    tr.euro_norm.reset()
        idx = [i for i in range(n) if rois[i] is not None]
        out = [None] * n
        if not idx:
            return out
        Ms = [G.roi_matrix(roi_px(rois[i], sizes[i][1], sizes[i][0]), 256, 256) for i in idx]
        u8 = G.warp_gpu(canvas[idx], [sizes[i] for i in idx], Ms, 256, 256, 'replicate')
        x = G.to_float(u8, 0.0, 1.0)
        res = {k: [] for k in ('Identity', 'Identity_1', 'Identity_3', 'Identity_4')}
        for s in range(0, len(idx), 64):
            r = self.engine(x[s:s + 64])
            for k in res:
                res[k].append(r[k])
        lm = torch.cat(res['Identity']).reshape(-1, 39, 5).cpu().numpy().astype(f32)
        xn, yn = (lm[..., 0] / f32(256)).astype(f32), (lm[..., 1] / f32(256)).astype(f32)
        win, cols, rows = heat_windows(torch.cat(res['Identity_3']), xn, yn)
        xr, yr = refine_from_heatmap(xn, yn, win, cols, rows)
        pres_flag = torch.cat(res['Identity_1']).reshape(-1).cpu().numpy()
        world_raw = torch.cat(res['Identity_4']).reshape(-1, 39, 3).cpu().numpy().astype(f32)
        for j, i in enumerate(idx):
            tr, (H, W) = tracks[i], sizes[i]
            if not (float(pres_flag[j]) > float(self.min_presence)):
                tr.euro_norm.reset(); tr.prev_roi = None
                continue
            raw = lm[j]
            x_n, y_n = xr[j], yr[j]
            z_n = (raw[:, 2] / f32(256)).astype(f32)
            vis, pres = G.sigmoid_f(raw[:, 3]), G.sigmoid_f(raw[:, 4])
            xc, yc, wn, hn, rot = rois[i]
            c, s = G.cosf(rot), G.sinf(rot)
            ux, uy = (x_n - f32(0.5)).astype(f32), (y_n - f32(0.5)).astype(f32)
            px = (((c * ux).astype(f32) - (s * uy).astype(f32)).astype(f32) * wn + xc).astype(f32)
            py = (((s * ux).astype(f32) + (c * uy).astype(f32)).astype(f32) * hn + yc).astype(f32)
            pz = (z_n * wn).astype(f32)
            wr = world_raw[j, :33]
            wx = ((c * wr[:, 0]).astype(f32) - (s * wr[:, 1]).astype(f32)).astype(f32)
            wy = ((s * wr[:, 0]).astype(f32) + (c * wr[:, 1]).astype(f32)).astype(f32)
            world = np.stack([wx, wy, wr[:, 2]], 1)
            nxt = expand(points_to_rect((px[33], py[33]), (px[34], py[34]), W, H), W, H)
            tr.prev_roi = nxt
            vis_s = tr.vis_norm.apply(vis[:33])
            obj = f32((f32(nxt[2] * f32(W)) + f32(nxt[3] * f32(H))) / f32(2.0))
            P = np.stack([px[:33] * f32(W), py[:33] * f32(H), pz[:33] * f32(W)], 1).astype(f32)
            sm = P if obj < f32(1e-6) else tr.euro_norm.apply(t_ns[i], P.reshape(-1), f32(f32(1.0) / obj)).reshape(33, 3)
            xyz = np.stack([sm[:, 0] / f32(W), sm[:, 1] / f32(H), sm[:, 2] / f32(W)], 1).astype(f32)
            wsm = tr.euro_world.apply(t_ns[i], world.reshape(-1), f32(1.0)).reshape(33, 3)
            out[i] = dict(pose_xyz=xyz, pose_vis=vis_s, pose_pres=pres[:33].astype(f32), pose_world=wsm.astype(f32))
        return out
