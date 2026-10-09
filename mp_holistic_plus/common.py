"""The CPU side of mp_holistic_plus (numpy and OpenCV): the pinned MediaPipe model
files, the options, the network that predicts a hand's box from its arm, the crop
geometry, the checks that reject wrong hands, and the output file."""
import dataclasses
import hashlib
import math
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"   # shipped with the package: hand_crop_mlp.npz
# The MediaPipe bundles (downloaded on first use), the models unpacked from them, the TensorRT engines.
CACHE = Path(os.environ.get("MP_HOLISTIC_PLUS_CACHE", Path.home() / ".cache" / "mp_holistic_plus"))

# Versioned URLs (the "latest" ones can change) and the sha256 of every asset.
_BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {name: f"{_BASE}/{folder}/float16/1/{name}" for name, folder in [
    ("pose_landmarker_heavy.task", "pose_landmarker/pose_landmarker_heavy"),
    ("face_landmarker.task", "face_landmarker/face_landmarker"),
    ("hand_landmarker.task", "hand_landmarker/hand_landmarker")]}
MODEL_SHA256 = {
    "pose_landmarker_heavy.task": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "hand_crop_mlp.npz": "8a89d24fbbcc16b94c112bbd62f6a3755f8ed203d3a9ec350dc5904eedf9fec8",
}

# Pose points of each arm, in the order HandCropMLP takes them:
# shoulder, elbow, wrist, thumb, index, pinky. Left and right are the person's.
POSE_ARM_IDX = {"left": [11, 13, 15, 21, 19, 17], "right": [12, 14, 16, 22, 20, 18]}
POSE_WRIST_IDX = {"left": 15, "right": 16}
HAND_SIDES = ("left", "right")       # index 0 and 1 of every hand_* array
N_POSE, N_FACE, N_HAND = 33, 478, 21

HAND_INPUT_SIZE = 224                # input side of the hand landmark model
HAND_NORMALIZE_Z = 0.4               # MediaPipe's z scale for that model (kLandmarksNormalizeZ)
FACE_INPUT_SIZE = 256                # input side of the face landmark model
HAND_FALLBACK_SIZE = 384             # side of the crop the palm detector searches


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_models():
    """Download the MediaPipe bundles to CACHE on first use, then check every model file's sha256."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        if not (CACHE / name).exists():
            print(f"[mp_holistic_plus] downloading {name} ...", file=sys.stderr)
            urllib.request.urlretrieve(url, CACHE / name)
    for name, want in MODEL_SHA256.items():
        path = (CACHE if name in MODEL_URLS else ASSETS) / name
        got = _sha256(path)
        if got != want:
            raise RuntimeError(f"{path}: sha256 {got}, expected {want}; "
                               f"delete it and run again to download the pinned version")


class HandCropMLP:
    """
    Predicts a hand's box from its arm: the network of
    github.com/sign-language-processing/mediapipe-hand-crop-fix (Moryossef 2024),
    evaluated in numpy. Only its centre and size heads are used; the rotation comes
    from heuristic_rotation_deg.

    Each head is an MLP 19 -> 10 -> 10 -> out with ReLU between the layers. Input:
    the frame's aspect ratio W / H, then x, y, z of shoulder, elbow, wrist, thumb,
    index and pinky, normalized as MediaPipe gives them. Output: the box centre
    (normalized by W and H) and its side (normalized by max(W, H)). The box is the
    one the hand landmark model runs on, used as is.

    The network was trained on right hands: a left arm is mirrored (x -> 1 - x)
    on the way in, and the centre mirrored back.
    """

    def __init__(self, npz_path: Path = ASSETS / "hand_crop_mlp.npz"):
        w = np.load(npz_path)
        self._w = {
            head: [(w[f"{head}.layers.{i}.weight"], w[f"{head}.layers.{i}.bias"])
                   for i in (0, 2, 4)]
            for head in ("center", "size")
        }

    @staticmethod
    def _forward(layers, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(layers):
            x = x @ W.T + b
            if i < len(layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def predict(self, arm_xyz: np.ndarray, aspect_ratio: float, side: str):
        """arm_xyz: (6, 3) pose points in POSE_ARM_IDX order -> (cx, cy, side)."""
        pts = arm_xyz.copy()
        if side == "left":
            pts[:, 0] = 1.0 - pts[:, 0]
        feat = np.concatenate(([aspect_ratio], pts.reshape(-1)))[None].astype(np.float32)
        cx, cy = self._forward(self._w["center"], feat)[0]
        size = float(self._forward(self._w["size"], feat)[0, 0])
        if side == "left":
            cx = 1.0 - cx
        return float(cx), float(cy), size


def heuristic_rotation_deg(wrist_px, index_px, pinky_px) -> float:
    """
    The hand's rotation in degrees, estimated from the pose as MediaPipe Holistic
    does: the hand points from the wrist to (2 * index + pinky) / 3, roughly the
    middle finger's knuckle. The crop-fix paper found its network's rotation head
    less accurate than this.
    """
    centre = (2.0 * np.asarray(index_px) + np.asarray(pinky_px)) / 3.0
    d = centre - np.asarray(wrist_px)
    return 90.0 + math.degrees(math.atan2(d[1], d[0]))


def crop_affine(cx_px: float, cy_px: float, size_px: float, rot_deg: float, out: int):
    """The 2x3 affine that maps the image onto an `out` x `out` crop of the square of
    side size_px centred at (cx_px, cy_px) and rotated by rot_deg, and its inverse."""
    M = cv2.getRotationMatrix2D((cx_px, cy_px), rot_deg, out / size_px)
    M[0, 2] += out / 2.0 - cx_px
    M[1, 2] += out / 2.0 - cy_px
    return M, cv2.invertAffineTransform(M)


def landmarks_to_roi(lm_xy: np.ndarray, scale: float = 2.0, shift_y: float = -0.1):
    """
    The box around a hand's 21 landmarks (pixels). It is rotated so that the hand
    points up (from the wrist to the middle of the index, middle and ring knuckles),
    fitted tightly in that rotated frame, made square on its longer side, moved by
    shift_y x side along the hand (negative: towards the fingers) and scaled by
    `scale`.

    Returns (cx, cy, side, rotation in degrees), in the units of lm_xy.
    """
    wrist, index, middle, ring = lm_xy[0], lm_xy[5], lm_xy[9], lm_xy[13]
    fingers = ((index + ring) / 2.0 + middle) / 2.0
    rot = math.pi / 2 - math.atan2(-(fingers[1] - wrist[1]), fingers[0] - wrist[0])
    rot = rot - 2 * math.pi * math.floor((rot + math.pi) / (2 * math.pi))

    centre = (lm_xy.max(0) + lm_xy.min(0)) / 2.0
    R = np.array([[math.cos(-rot), -math.sin(-rot)],
                  [math.sin(-rot),  math.cos(-rot)]])
    rotated = (lm_xy - centre) @ R.T
    rc = (rotated.max(0) + rotated.min(0)) / 2.0
    Rf = np.array([[math.cos(rot), -math.sin(rot)],
                   [math.sin(rot),  math.cos(rot)]])
    final = rc @ Rf.T + centre
    w, h = rotated.max(0) - rotated.min(0)

    size = max(w, h)
    cx = final[0] + shift_y * size * math.sin(rot)
    cy = final[1] + shift_y * size * math.cos(rot)
    return float(cx), float(cy), float(size * scale), math.degrees(rot)


def hands_coincide(a: np.ndarray, b: np.ndarray, W: int, H: int, tol: float = 0.25) -> bool:
    """Whether two (21, 3) hands (normalized coordinates) are one physical hand: the
    mean distance between their landmarks is below `tol` x the hand's size in pixels.
    On 19 broadcast clips, one hand found twice differed by at most 0.1, two real
    hands held close by at least 0.3."""
    a, b = a[:, :2] * (W, H), b[:, :2] * (W, H)
    size = max(float(np.ptp(a, 0).max()), float(np.ptp(b, 0).max()), 1.0)
    return float(np.linalg.norm(a - b, axis=1).mean()) < tol * size


def unmap_crop(lm_norm_in_crop: np.ndarray, Minv, out: int,
               size_px: float, W: int, H: int) -> np.ndarray:
    """Landmarks normalized to an `out` x `out` crop made with crop_affine -> normalized
    to the W x H frame. z is on the scale of x, as in MediaPipe, so it is multiplied
    by size_px / W."""
    n = lm_norm_in_crop.shape[0]
    xy_px = lm_norm_in_crop[:, :2] * out
    xy = np.c_[xy_px, np.ones(n)] @ Minv.T
    res = np.empty((n, 3), np.float32)
    res[:, 0] = xy[:, 0] / W
    res[:, 1] = xy[:, 1] / H
    res[:, 2] = lm_norm_in_crop[:, 2] * (size_px / W)
    return res


@dataclasses.dataclass
class Options:
    #: no pose below this pose presence (MediaPipe's min_pose_presence_confidence)
    min_pose_presence_confidence: float = 0.5
    #: below this hand presence, the palm detector searches for the hand
    hand_presence_threshold: float = 0.5
    #: a hand below this presence is dropped, even after that search
    hand_reject_threshold: float = 0.2
    #: the palm detector searches the square around the pose's wrist, thumb, index
    #: and pinky points whose half-side is this many times their largest x or y
    #: distance from their mean (and at least 4% of the frame's longer side)
    fallback_roi_scale: float = 3.0
    #: times the hand is re-cropped around its own landmarks and run again; a pass
    #: whose presence is lower than the previous one is discarded and ends the passes
    hand_refine_passes: int = 2
    #: scale of the box around the landmarks for those passes (see landmarks_to_roi).
    #: 1.6 makes the box the size MediaPipe's own hand tracking uses (0.96 of it),
    #: and hand z depends strongly on the box size. Against stock MediaPipe Holistic
    #: on one 15-frame clip:
    #:
    #:   passes  scale | z corr | xy error | box side / Holistic's
    #:      0     --   | -0.446 |  9.42 px |  1.52
    #:      2     2.0  | -0.448 |  5.14 px |  1.34
    #:      2     1.8  | +0.333 |  3.29 px |  1.15
    #:      2     1.6  | +0.898 |  1.42 px |  0.96
    #:      2     1.4  | +0.977 |  0.69 px |  0.81
    hand_refine_scale: float = 1.6


def decode_hand(lm_crop, Minv, size_px, W, H):
    """
    The hand model's raw landmarks on a 224 x 224 crop (made with crop_affine) ->
    normalized frame coordinates, decoded as MediaPipe decodes them: x and y through
    the inverse crop; z = raw / 224 / 0.4 (TensorsToLandmarksCalculator) x size_px / W
    (LandmarkProjectionCalculator scales z like x).
    """
    xy = np.c_[lm_crop[:, :2], np.ones(N_HAND)] @ Minv.T
    out = np.empty((N_HAND, 3), np.float32)
    out[:, 0] = xy[:, 0] / W
    out[:, 1] = xy[:, 1] / H
    out[:, 2] = lm_crop[:, 2] / HAND_INPUT_SIZE / HAND_NORMALIZE_Z * (size_px / W)
    return out


def side_ok(lm, pose_xyz, side, W, H) -> bool:
    """False if the hand's wrist is nearer the other arm's pose wrist than its own.
    When both hands are in one crop the hand model may return either of them.
    True if a pose wrist is missing."""
    pw = lm[0, :2] * (W, H)
    own = pose_xyz[POSE_WRIST_IDX[side], :2] * (W, H)
    other = pose_xyz[POSE_WRIST_IDX["right" if side == "left" else "left"], :2] * (W, H)
    if not (np.all(np.isfinite(own)) and np.all(np.isfinite(other))):
        return True
    return float(np.linalg.norm(pw - own)) <= float(np.linalg.norm(pw - other))


def face_roi_from_pose(pose_xyz: np.ndarray, W: int, H: int):
    """
    The face crop, from the pose as in MediaPipe Holistic but not rotated: the square
    (in pixels) around pose points 0-10 (nose, eyes, ears, mouth), 3 times their
    extent. The face detector only finds faces that fill much of its input, so it
    runs on this crop rather than on the frame.

    Returns (cx_px, cy_px, size_px), or None.
    """
    pts = pose_xyz[0:11, :2]
    if not np.all(np.isfinite(pts)):
        return None
    px = pts * np.array([W, H], np.float32)
    lo, hi = px.min(0), px.max(0)
    centre = (lo + hi) / 2.0
    size_px = float(np.max(hi - lo)) * 3.0
    if not (np.isfinite(size_px) and size_px >= 8.0):
        return None
    return float(centre[0]), float(centre[1]), size_px


class FrameRow:
    """One frame of output. The defaults mean "not found"."""
    __slots__ = ("pose_xyz", "pose_vis", "pose_pres", "pose_world", "pose_ok",
                 "face_xyz", "face_ok", "hand_xyz", "hand_world", "hand_score",
                 "hand_ok", "hand_roi")

    def __init__(self):
        nan = np.nan
        self.pose_xyz = np.full((N_POSE, 3), nan, np.float32)
        self.pose_vis = np.zeros(N_POSE, np.float32)
        self.pose_pres = np.zeros(N_POSE, np.float32)
        self.pose_world = np.full((N_POSE, 3), nan, np.float32)
        self.pose_ok = False
        self.face_xyz = np.full((N_FACE, 3), nan, np.float32)
        self.face_ok = False
        self.hand_xyz = np.full((2, N_HAND, 3), nan, np.float32)
        self.hand_world = np.full((2, N_HAND, 3), nan, np.float32)
        self.hand_score = np.zeros(2, np.float32)
        self.hand_ok = np.zeros(2, bool)
        self.hand_roi = np.full((2, 4), nan, np.float32)


class Recorder:
    """The rows of one video, written as one .npz (format in README.md)."""
    def __init__(self):
        self.rows = {k: [] for k in FrameRow.__slots__}

    def append(self, row: FrameRow):
        for k, v in self.rows.items():
            v.append(getattr(row, k))

    def save(self, path: Path):
        T = len(self.rows["pose_ok"])

        def stack(k, shape):   # a video with no frames still gets arrays of the right rank
            return (np.asarray(self.rows[k], np.float32) if T else
                    np.zeros((0,) + shape, np.float32))

        def f16(k, shape):
            return stack(k, shape).astype(np.float16)

        def u8(k, shape):      # [0, 1] -> 0..255
            return np.clip(np.rint(stack(k, shape) * 255.0), 0, 255).astype(np.uint8)

        def flags(k, shape):
            return np.asarray(self.rows[k], bool).reshape((T,) + shape)

        np.savez(
            path,
            pose_xyz=f16("pose_xyz", (N_POSE, 3)),
            pose_vis=u8("pose_vis", (N_POSE,)),
            pose_pres=u8("pose_pres", (N_POSE,)),
            pose_world=f16("pose_world", (N_POSE, 3)),
            pose_ok=flags("pose_ok", ()),
            face_xyz=f16("face_xyz", (N_FACE, 3)),
            face_ok=flags("face_ok", ()),
            hand_xyz=f16("hand_xyz", (2, N_HAND, 3)),
            hand_world=f16("hand_world", (2, N_HAND, 3)),
            hand_score=u8("hand_score", (2,)),
            hand_ok=flags("hand_ok", (2,)),
            hand_roi=stack("hand_roi", (2, 4)),
        )
