"""The hands: each arm's hand in every frame, for the frames of many videos at once.

For each arm with a pose wrist of visibility >= 0.1:

1. HandCropMLP predicts the hand's box from the arm's pose points; the rotation is
   MediaPipe's estimate from the wrist, index and pinky (common.heuristic_rotation_deg).
   The hand landmark model runs on that crop.
2. The hand is re-cropped around its own landmarks and run again
   (Options.hand_refine_passes, hand_refine_scale).
3. If the hand is on the other arm (common.side_ok) or its presence is below
   Options.hand_presence_threshold, MediaPipe's palm detector searches around the
   arm's pose hand points for up to 2 hands. Each is run through the hand model and
   refined as in 2. The best one that is on this arm and is not the other arm's hand
   replaces the result if its presence is higher.
4. A hand below Options.hand_reject_threshold is dropped. If both slots hold one
   physical hand (common.hands_coincide), it is kept on the arm whose pose wrist is
   nearer.

All hand-model calls of one step are one batched call.
"""
import math

import cv2
import numpy as np
import torch

from . import common as hp

SIDES = hp.HAND_SIDES


class Hand:
    """The hand model's result on one box: landmarks (normalized to the frame), world
    landmarks (metres, rotated back to the image's axes), presence, and the box
    (cx, cy, side, rotation in degrees; normalized, side by max(W, H))."""
    __slots__ = ('lm', 'world', 'score', 'roi')

    def __init__(self, lm, world, score, roi):
        self.lm, self.world, self.score, self.roi = lm, world, score, roi


class HandStage:
    def __init__(self, engine, opts: hp.Options, fallback, device):
        self.engine, self.opts, self.fallback, self.device = engine, opts, fallback, device
        self.mlp = hp.HandCropMLP()
        # The model takes uint8 / 255 in float32, correctly rounded. CUDA computes x / 255
        # as x * (1 / 255), which is 1 ulp off for half the values, so look the quotients up.
        self.lut = torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0).to(device)

    def at_rois(self, reqs):
        """Run the hand model on many boxes at once. reqs: (rgb, cx, cy, side, rot_deg, W, H)
        per box, normalized as in Hand.roi. Returns per box a Hand, or None if the box is
        not finite or under 4 px."""
        out, crops, keep = [None] * len(reqs), [], []
        for i, (rgb, cx, cy, size, rot_deg, W, H) in enumerate(reqs):
            size_px = size * max(W, H)
            if not (np.isfinite(size_px) and size_px > 4.0 and np.isfinite(cx) and np.isfinite(cy)):
                continue
            M, Minv = hp.crop_affine(cx * W, cy * H, size_px, rot_deg, hp.HAND_INPUT_SIZE)
            crops.append(np.ascontiguousarray(cv2.warpAffine(rgb, M, (hp.HAND_INPUT_SIZE, hp.HAND_INPUT_SIZE), flags=cv2.INTER_LINEAR)))
            keep.append((i, Minv, size_px))
        if not crops:
            return out
        x = self.lut[torch.from_numpy(np.stack(crops)).to(self.device).long()]
        res = {}
        for s in range(0, len(crops), 64):
            r = self.engine(x[s:s + 64])
            for k in ('Identity', 'Identity_1', 'Identity_3'):
                res.setdefault(k, []).append(r[k])
        lm_all = torch.cat(res['Identity']).cpu().numpy().reshape(-1, hp.N_HAND, 3)
        pres_all = torch.cat(res['Identity_1']).cpu().numpy().reshape(-1)
        world_all = torch.cat(res['Identity_3']).cpu().numpy().reshape(-1, hp.N_HAND, 3)
        for j, (i, Minv, size_px) in enumerate(keep):
            rgb, cx, cy, size, rot_deg, W, H = reqs[i]
            lm_crop, world_crop, presence = lm_all[j], world_all[j], float(pres_all[j])
            lm_norm = hp.decode_hand(lm_crop, Minv, size_px, W, H)
            a = math.radians(rot_deg)
            ca, sa = math.cos(a), math.sin(a)
            world = world_crop.copy()
            world[:, 0] = ca * world_crop[:, 0] - sa * world_crop[:, 1]
            world[:, 1] = sa * world_crop[:, 0] + ca * world_crop[:, 1]
            out[i] = Hand(lm_norm, world, presence, (cx, cy, size, rot_deg))
        return out

    def refine(self, items):
        """items: [rgb, W, H, Hand or None] lists. Each hand is replaced in place by the
        model's result on the box around its own landmarks, up to hand_refine_passes
        times; a result with lower presence is discarded and ends that hand's passes."""
        active = list(range(len(items)))
        for _ in range(self.opts.hand_refine_passes):
            reqs, idx = [], []
            for i in active:
                rgb, W, H, got = items[i]
                if got is None or not np.all(np.isfinite(got.lm[:, :2])):
                    continue
                cx, cy, size_abs, rot = hp.landmarks_to_roi(got.lm[:, :2] * np.array([W, H], np.float32),
                                                            scale=self.opts.hand_refine_scale)
                reqs.append((rgb, cx / W, cy / H, size_abs / max(W, H), rot, W, H))
                idx.append(i)
            nxt = self.at_rois(reqs)
            active = []
            for i, n in zip(idx, nxt):
                if n is None or n.score < items[i][3].score:
                    continue
                items[i][3] = n
                active.append(i)
            if not active:
                break

    def run(self, frames):
        """frames: one dict per video with rgb, W, H, pose_ok, pose_xyz, pose_vis, pose_world.
        Returns per frame a dict of hand_xyz, hand_world, hand_score, hand_roi, hand_ok."""
        jobs = []                                             # (frame index, side index, side)
        for f, fr in enumerate(frames):
            if fr['pose_ok']:
                for hi, side in enumerate(SIDES):
                    if fr['pose_vis'][hp.POSE_WRIST_IDX[side]] < 0.1:
                        continue
                    jobs.append((f, hi, side))
        # 1. the box from the pose
        reqs, req_jobs, got = [], [], {}
        for j in jobs:
            f, hi, side = j
            fr = frames[f]
            W, H = fr['W'], fr['H']
            arm = fr['pose_xyz'][hp.POSE_ARM_IDX[side]]
            if not np.all(np.isfinite(arm)):
                got[j] = None
                continue
            cx, cy, size = self.mlp.predict(arm, W / H, side)
            rot_deg = hp.heuristic_rotation_deg(arm[2, :2] * (W, H), arm[4, :2] * (W, H), arm[5, :2] * (W, H))
            reqs.append((fr['rgb'], cx, cy, size, rot_deg, W, H))
            req_jobs.append(j)
        for j, h in zip(req_jobs, self.at_rois(reqs)):
            got[j] = h
        # 2. re-crop around the landmarks
        items = [[frames[j[0]]['rgb'], frames[j[0]]['W'], frames[j[0]]['H'], got[j]] for j in jobs]
        self.refine(items)
        for j, it in zip(jobs, items):
            got[j] = it[3]
        # 3. on the other arm or unsure: the palm detector
        fb_jobs = []
        for j in jobs:
            f, hi, side = j
            fr = frames[f]
            if got[j] is not None and not hp.side_ok(got[j].lm, fr['pose_xyz'], side, fr['W'], fr['H']):
                got[j] = None
            if got[j] is None or got[j].score < self.opts.hand_presence_threshold:
                fb_jobs.append(j)
        rois = list(self.fallback.pool.map(lambda j: self.fallback_roi(frames[j[0]], j[2]), fb_jobs))
        fb_items, fb_of = [], []
        for j, rs in zip(fb_jobs, rois):
            for r in rs or ():
                fb_items.append(r)
                fb_of.append(j)
        fb_hands = self.at_rois(fb_items) if fb_items else []
        items = [[frames[j[0]]['rgb'], frames[j[0]]['W'], frames[j[0]]['H'], h] for j, h in zip(fb_of, fb_hands)]
        self.refine(items)
        cands = {}
        for j, it in zip(fb_of, items):
            cands.setdefault(j, []).append(it[3])
        primary = {j: got[j] for j in jobs}
        for j in fb_jobs:
            f, hi, side = j
            fr = frames[f]
            other = primary.get((f, 1 - hi, SIDES[1 - hi]))
            best = None
            for b in cands.get(j, ()):
                if b is None or not hp.side_ok(b.lm, fr['pose_xyz'], side, fr['W'], fr['H']):
                    continue
                if other is not None and hp.hands_coincide(b.lm, other.lm, fr['W'], fr['H']):
                    continue                                   # the other arm's hand
                if best is None or b.score > best.score:
                    best = b
            if best is not None and (got[j] is None or best.score > got[j].score):
                got[j] = best
        # 4. the rows, with the hand's world landmarks moved onto the pose's world wrist
        rows = []
        for f, fr in enumerate(frames):
            row = dict(hand_xyz=np.full((2, hp.N_HAND, 3), np.nan, np.float32), hand_world=np.full((2, hp.N_HAND, 3), np.nan, np.float32),
                       hand_score=np.zeros(2, np.float32), hand_roi=np.full((2, 4), np.nan, np.float32),
                       hand_ok=np.zeros(2, bool))
            rows.append(row)
        for j in jobs:
            f, hi, side = j
            fr, h = frames[f], got[j]
            if h is None or h.score < self.opts.hand_reject_threshold:
                continue
            world = h.world
            if fr['pose_world'] is not None and np.all(np.isfinite(world)):
                world = world - world[0] + fr['pose_world'][hp.POSE_WRIST_IDX[side]]
            row = rows[f]
            row['hand_xyz'][hi] = h.lm
            row['hand_world'][hi] = world
            row['hand_score'][hi] = h.score
            row['hand_roi'][hi] = h.roi
            row['hand_ok'][hi] = True
        for f, fr in enumerate(frames):                       # one hand in both slots
            row, W, H = rows[f], fr['W'], fr['H']
            if row['hand_ok'].all() and hp.hands_coincide(row['hand_xyz'][0], row['hand_xyz'][1], W, H):
                wl, wr = row['hand_xyz'][0, 0, :2] * (W, H), row['hand_xyz'][1, 0, :2] * (W, H)
                dl = np.linalg.norm(wl - fr['pose_xyz'][hp.POSE_WRIST_IDX['left'], :2] * (W, H))
                dr = np.linalg.norm(wr - fr['pose_xyz'][hp.POSE_WRIST_IDX['right'], :2] * (W, H))
                drop = 1 if dl <= dr else 0
                row['hand_xyz'][drop] = np.nan; row['hand_world'][drop] = np.nan
                row['hand_score'][drop] = 0.0; row['hand_roi'][drop] = np.nan
                row['hand_ok'][drop] = False
        return rows

    def fallback_roi(self, fr, side):
        """Run the palm detector (MediaPipe HandLandmarker) on a square around the arm's
        wrist, thumb, index and pinky pose points (Options.fallback_roi_scale). Returns, per
        hand it finds, the box around that hand's landmarks as at_rois takes it; or None."""
        import mediapipe as mp
        W, H, rgb = fr['W'], fr['H'], fr['rgb']
        arm = fr['pose_xyz'][hp.POSE_ARM_IDX[side]]
        if not np.all(np.isfinite(arm)):
            return None
        pts = arm[2:, :2] * np.array([W, H], np.float32)
        c = pts.mean(0)
        half = max(float(np.abs(pts - c).max()) * self.opts.fallback_roi_scale, 0.04 * max(W, H))
        size_px = 2 * half
        if size_px < 8:
            return None
        M, Minv = hp.crop_affine(float(c[0]), float(c[1]), size_px, 0.0, hp.HAND_FALLBACK_SIZE)
        crop = np.ascontiguousarray(cv2.warpAffine(rgb, M, (hp.HAND_FALLBACK_SIZE, hp.HAND_FALLBACK_SIZE), flags=cv2.INTER_LINEAR))
        res = self.fallback.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        if not res.hand_landmarks:
            return None
        out = []
        for hand in res.hand_landmarks:
            lm_crop = np.array([[p.x, p.y, p.z] for p in hand], np.float32)
            lm = hp.unmap_crop(lm_crop, Minv, hp.HAND_FALLBACK_SIZE, size_px, W, H)
            cx, cy, size_abs, rot_deg = hp.landmarks_to_roi(lm[:, :2] * np.array([W, H], np.float32), scale=self.opts.hand_refine_scale)
            out.append((rgb, cx / W, cy / H, size_abs / max(W, H), rot_deg, W, H))
        return out
