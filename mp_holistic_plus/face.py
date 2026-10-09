"""The face: MediaPipe's FaceLandmarker (VIDEO mode, one face) run on a crop around the
pose's face points, for many videos at once.

Per video: a 256 x 256 crop around pose points 0-10 (common.face_roi_from_pose, made
with OpenCV on the CPU; no pose, no face). On it, as FaceLandmarker does: the face
detector (short-range BlazeFace, LiteRT, CPU) when there is no track, the landmark
model on the face box (batched on the GPU), the presence check, the next frame's box
from the landmarks, and One Euro smoothing. The landmarks are then mapped from the
crop back to the frame.

Written from MediaPipe's C++ source: the face box crop is OpenCV's fixed-point warp
redone on the GPU (mpgeom.warp_gpu), and every scalar step keeps MediaPipe's
float32/float64 mix. With the model run by LiteRT the output was checked byte-identical
to mediapipe 1.0.1's FaceLandmarker on the CPU, given the same crop.
"""
import math

import cv2
import numpy as np
import torch

from . import mpgeom as G
from . import common as hp
from .pose import OneEuro

f32 = np.float32
FACE = hp.FACE_INPUT_SIZE                                   # side of the face crop and of the model's input


def face_anchors():
    """Short-range BlazeFace anchors: 128 x 128 input, strides 8 and 16, 2 and 6 per cell."""
    anchors = []
    for stride, n in ((8, 2), (16, 6)):
        fm = math.ceil(128 / stride)
        for y in range(fm):
            for x in range(fm):
                for _ in range(n):
                    anchors.append((f32((x + f32(0.5)) / fm), f32((y + f32(0.5)) / fm)))
    a = np.array(anchors, f32)
    assert len(a) == 896
    return a


class FaceDetector:
    """Short-range BlazeFace on the 256 x 256 crop (LiteRT, CPU) -> the face box tracking
    starts from (crop-normalized), or None."""
    def __init__(self, tflite_path):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(tflite_path), num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]['index']
        self.outs = self.it.get_output_details()
        self.anchors = face_anchors()

    def __call__(self, canvas, i):
        W = H = FACE
        roi = (f32(0.5 * W), f32(0.5 * H), f32(W), f32(H), 0.0)    # full image, square: no padding
        u8 = G.warp_gpu(canvas[i:i + 1], [(H, W)], [G.roi_matrix(roi, 128, 128)], 128, 128, 'const')
        t = G.to_float(u8, -1.0, 1.0).cpu().numpy()
        self.it.set_tensor(self.inp, t); self.it.invoke()
        outs = [self.it.get_tensor(o['index']) for o in self.outs]
        raw = [o for o in outs if o.shape[-1] == 16][0][0]
        sc = G.sigmoid_f(np.clip([o for o in outs if o.shape[-1] == 1][0][0, :, 0], f32(-100), f32(100)))
        dets = []
        for k in np.where(~(sc < f32(0.5)))[0]:
            ax, ay = self.anchors[k]
            xc = raw[k, 0] / f32(128) + ax; yc = raw[k, 1] / f32(128) + ay
            w = raw[k, 2] / f32(128); h = raw[k, 3] / f32(128)
            ymin = yc - h / f32(2); xmin = xc - w / f32(2); ymax = yc + h / f32(2); xmax = xc + w / f32(2)
            kps = [(raw[k, 4 + 2 * q] / f32(128) + ax, raw[k, 5 + 2 * q] / f32(128) + ay) for q in range(6)]
            bw = xmax - xmin; bh = ymax - ymin
            if bw < 0 or bh < 0:
                continue
            dets.append((sc[k], f32(xmin), f32(ymin), f32(bw), f32(bh), np.array(kps, f32)))
        if not dets:
            return None
        order = sorted(range(len(dets)), key=lambda q: -dets[q][0])

        def iou(p, q):
            ax0, ay0, ax1, ay1 = p[1], p[2], p[1] + p[3], p[2] + p[4]; bx0, by0, bx1, by1 = q[1], q[2], q[1] + q[3], q[2] + q[4]
            ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
            if ix1 < ix0 or iy1 < iy0:
                return 0.0
            inter = (ix1 - ix0) * (iy1 - iy0); un = p[3] * p[4] + q[3] * q[4] - inter
            return inter / un if un > 0 else 0.0

        out, rem = [], order
        while rem:
            top = dets[rem[0]]
            cand = [q for q in rem if iou(dets[q], top) > 0.5]; rest = [q for q in rem if q not in cand]
            tot = f32(0); wx0 = wy0 = wx1 = wy1 = f32(0); kp = np.zeros((6, 2), f32)
            for q in cand:
                s = dets[q][0]; tot += s; d = dets[q]
                wx0 += d[1] * s; wy0 += d[2] * s; wx1 += (d[1] + d[3]) * s; wy1 += (d[2] + d[4]) * s; kp += d[5] * s
            xmin = wx0 / tot; ymin = wy0 / tot
            out.append((top[0], xmin, ymin, wx1 / tot - xmin, wy1 / tot - ymin, kp / tot))
            if len(rest) == len(rem):
                break
            rem = rest
        d = out[0]
        xmin, ymin, bw, bh, kp = d[1], d[2], d[3], d[4], d[5]
        xc = xmin + bw / f32(2); yc = ymin + bh / f32(2)
        x0, y0 = kp[0, 0] * f32(W), kp[0, 1] * f32(H); x1, y1 = kp[1, 0] * f32(W), kp[1, 1] * f32(H)
        rot = G.norm_rad(f32(0) - G.atan2f(-(y1 - y0), x1 - x0))
        return (f32(xc), f32(yc), f32(bw * f32(1.5)), f32(bh * f32(1.5)), rot)


def rot_matrix(cx, cy, w, h, rot, W, H):
    """MediaPipe's GetRotatedSubRectToRectTransformMatrix in float32: maps box-normalized
    points into the image. Only the 6 entries the landmark projection uses."""
    a, b = f32(w), f32(h); c = G.cosf(f32(rot)); d = G.sinf(f32(rot)); e, f = f32(cx), f32(cy)
    g = f32(1.0) / f32(W); hh = f32(1.0) / f32(H)
    m0 = a * c * g; m1 = -b * d * g; m3 = (f32(-0.5) * a * c + f32(0.5) * b * d + e) * g
    m4 = a * d * hh; m5 = b * c * hh; m7 = (f32(-0.5) * b * c - f32(0.5) * a * d + f) * hh
    return m0, m1, m3, m4, m5, m7


class FaceTrack:
    """One video's face tracking state."""
    def __init__(self):
        self.prev = None                      # the next frame's face box (crop-normalized); None: detect
        self.smoother = None                  # OneEuro over 478 x 3, or None after a reset


class FaceStage:
    def __init__(self, engine, detector):
        self.engine, self.detector = engine, detector

    def run(self, tracks, frames, poses, ts_ms):
        """frames: RGB uint8 per video; poses: PoseStage.run's output; ts_ms: timestamps.
        Returns per video the (478, 3) landmarks normalized to the frame, or None."""
        n = len(tracks)
        out = [None] * n
        jobs = []                                                   # (video, Minv, size_px)
        crops = []
        for i in range(n):
            if poses[i] is None:
                continue
            H, W = frames[i].shape[:2]
            roi = hp.face_roi_from_pose(poses[i]['pose_xyz'], W, H)
            if roi is None:
                continue
            cx_px, cy_px, size_px = roi
            M, Minv = hp.crop_affine(cx_px, cy_px, size_px, 0.0, FACE)
            crops.append(np.ascontiguousarray(cv2.warpAffine(frames[i], M, (FACE, FACE), flags=cv2.INTER_LINEAR)))
            jobs.append((i, Minv, size_px))
        if not jobs:
            return out
        canvas = torch.from_numpy(np.stack(crops)).cuda()
        rects = []
        for k, (i, _, _) in enumerate(jobs):
            tr = tracks[i]
            r = tr.prev if tr.prev is not None else self.detector(canvas, k)
            if r is None:
                tr.prev = None; tr.smoother = None
            rects.append(r)
        sel = [k for k, r in enumerate(rects) if r is not None]
        if not sel:
            return out
        rois = [(f32(rects[k][0] * f32(FACE)), f32(rects[k][1] * f32(FACE)), f32(rects[k][2] * f32(FACE)),
                 f32(rects[k][3] * f32(FACE)), rects[k][4]) for k in sel]
        u8 = G.warp_gpu(canvas[sel], [(FACE, FACE)] * len(sel), [G.roi_matrix(r, 256, 256) for r in rois], 256, 256, 'replicate')
        x = G.to_float(u8, 0.0, 1.0)
        res = {'Identity': [], 'Identity_1': []}
        for s in range(0, len(sel), 64):
            r = self.engine(x[s:s + 64])
            for kk in res:
                res[kk].append(r[kk])
        raw_all = torch.cat(res['Identity']).reshape(-1, 478, 3).cpu().numpy().astype(f32)
        pres_all = torch.cat(res['Identity_1']).reshape(-1).cpu().numpy().astype(f32)
        W = H = FACE
        for q, k in enumerate(sel):
            i, Minv, size_px = jobs[k]
            tr = tracks[i]
            pres = f32(f32(1) / (f32(1) + G.expf(-pres_all[q])))
            if not (float(pres) > 0.5):
                tr.prev = None; tr.smoother = None
                continue
            raw = raw_all[q]
            nx = raw[:, 0] / f32(256); ny = raw[:, 1] / f32(256); nz = raw[:, 2] / f32(256)
            roi = rois[q]
            m0, m1, m3, m4, m5, m7 = rot_matrix(roi[0], roi[1], roi[2], roi[3], roi[4], W, H)
            px = nx * m0 + ny * m1 + nz * f32(0) + m3; py = nx * m4 + ny * m5 + nz * f32(0) + m7
            dxz = (m0 + m3) - m3; dyz = (m4 + m7) - m7
            zs = f32(math.sqrt(float(dxz) ** 2 + float(dyz) ** 2))
            L = np.stack([px, py, zs * nz], 1).astype(f32)
            # next-frame rect from the raw landmarks
            xmn, xmx = L[:, 0].min(), L[:, 0].max(); ymn, ymx = L[:, 1].min(), L[:, 1].max()
            bw, bh = xmx - xmn, ymx - ymn; cx, cy = xmn + bw / f32(2), ymn + bh / f32(2)
            x0, y0 = L[33, 0] * f32(W), L[33, 1] * f32(H); x1, y1 = L[263, 0] * f32(W), L[263, 1] * f32(H)
            rr = G.norm_rad(f32(0) - G.atan2f(-(y1 - y0), x1 - x0))
            long_side = max(bw * f32(W), bh * f32(H)); nw = long_side / f32(W); nh = long_side / f32(H)
            tr.prev = (f32(cx), f32(cy), f32(nw * f32(1.5)), f32(nh * f32(1.5)), rr)
            # OneEuro on the output, object scale from the landmarks (crop pixels)
            Lp = np.stack([L[:, 0] * f32(W), L[:, 1] * f32(H), L[:, 2] * f32(W)], 1).astype(f32)
            osc = f32((f32(Lp[:, 0].max() - Lp[:, 0].min()) + f32(Lp[:, 1].max() - Lp[:, 1].min())) / f32(2))
            if osc < f32(1e-6):
                S = Lp
            else:
                if tr.smoother is None:
                    tr.smoother = OneEuro(478 * 3, 0.05, 80.0, 1.0)
                S = tr.smoother.apply(int(ts_ms[i]) * 1000 * 1000, Lp.reshape(-1), f32(f32(1) / osc)).reshape(478, 3)
            lm = np.stack([S[:, 0] / f32(W), S[:, 1] / f32(H), S[:, 2] / f32(W)], 1).astype(f32)
            H0, W0 = frames[i].shape[:2]
            out[i] = hp.unmap_crop(lm, Minv, FACE, size_px, W0, H0)
        return out
