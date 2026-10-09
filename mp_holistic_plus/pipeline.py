"""HolisticPlus: pose, face and hand landmarks for many videos at once.

Each video is decoded on its own thread. A step takes the next frame of every
video, and the pose, face and hand stages each run once on all of them, so every
model call is batched. Each video's frames still go through in order, which the
pose and face tracking need.
"""
import queue
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from . import face as F
from . import hands as Hs
from . import pose as P
from . import common as hp
from .trt_engine import LiteRTEngine, build

CACHE = hp.CACHE   # the bundles, the models unpacked from them, the TensorRT engines
BUNDLED = {'pose_detector.tflite': 'pose_landmarker_heavy.task', 'pose_landmarks_detector.tflite': 'pose_landmarker_heavy.task',
           'face_detector.tflite': 'face_landmarker.task', 'face_landmarks_detector.tflite': 'face_landmarker.task',
           'hand_landmarks_detector.tflite': 'hand_landmarker.task'}


def unpack_models():
    """The .tflite models inside the MediaPipe bundles, unpacked to the cache: {name: path}."""
    hp.verified_models()
    CACHE.mkdir(exist_ok=True)
    out = {}
    for name, bundle in BUNDLED.items():
        path = CACHE / name
        data = zipfile.ZipFile(CACHE / bundle).read(name)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        out[name] = path
    return out


class Decoder:
    """Reads one video on its own thread: RGB frames, then None."""
    def __init__(self, path, depth=32):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f'cannot open {path}')
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if (np.isfinite(fps) and fps > 0) else 25.0
        self.q = queue.Queue(depth)
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            while not self.stop:
                ok, bgr = self.cap.read()
                if not ok:
                    break
                self.q.put(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            self.cap.release()
            self.q.put(None)

    def close(self):
        self.stop = True
        while self.q.get() is not None:
            pass


class Slot:
    """One video in progress."""
    def __init__(self, video, output):
        self.video, self.output = Path(video), Path(output)
        self.dec = Decoder(video)
        self.pose, self.face = P.PoseTrack(), F.FaceTrack()
        self.rec = hp.Recorder()
        self.idx, self.prev_ts, self.W, self.H = 0, -1, None, None


class FallbackPool:
    """MediaPipe's HandLandmarker (palm detector and hand model; CPU, IMAGE mode, up to 2
    hands), one per thread, for the hand search of hands.HandStage."""
    def __init__(self, n):
        from mediapipe.tasks.python import BaseOptions, vision
        self.local, self.made, self.lock = threading.local(), [], threading.Lock()
        self.make = lambda: vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(CACHE / 'hand_landmarker.task'), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE, num_hands=2))
        self.pool = ThreadPoolExecutor(n)

    def detect(self, image):
        lm = getattr(self.local, 'lm', None)
        if lm is None:
            lm = self.local.lm = self.make()
            with self.lock:
                self.made.append(lm)
        return lm.detect(image)

    def close(self):
        self.pool.shutdown(wait=True)
        for lm in self.made:
            lm.close()
        self.made.clear()


class HolisticPlus:
    def __init__(self, batch=16, fallback_threads=4, backend='tensorrt', opts=None):
        """batch: videos processed together. fallback_threads: CPU threads for the palm
        detector. backend: 'tensorrt' (the landmark models in fp32 on the GPU) or 'litert'
        (on the CPU with MediaPipe's own arithmetic; slow, for checking exactness).
        opts: common.Options."""
        self.opts = opts or hp.Options()
        self.batch = batch
        models = unpack_models()
        if backend == 'tensorrt':
            engines = {'pose': build(models['pose_landmarks_detector.tflite'], 256, CACHE, drop=('Identity_2',)),   # no segmentation
                       'face': build(models['face_landmarks_detector.tflite'], 256, CACHE),
                       'hand': build(models['hand_landmarks_detector.tflite'], 224, CACHE)}
        elif backend == 'litert':
            engines = {k: LiteRTEngine(models[f'{k}_landmarks_detector.tflite']) for k in ('pose', 'face', 'hand')}
        else:
            raise ValueError(backend)
        self.pose = P.PoseStage(engines['pose'], P.PoseDetector(models['pose_detector.tflite']), self.opts.min_pose_presence_confidence)
        self.face = F.FaceStage(engines['face'], F.FaceDetector(models['face_detector.tflite']))
        self.fallback = FallbackPool(fallback_threads)
        self.hands = Hs.HandStage(engines['hand'], self.opts, self.fallback, torch.device('cuda'))

    def close(self):
        self.fallback.close()

    def run(self, jobs, on_done=None):
        """jobs: iterable of (video, output .npz), taken as slots free up; up to `batch`
        videos advance one frame per step. on_done(video, output, error or None, frames)
        is called once per video."""
        jobs, active = iter(jobs), []
        more = True
        while more or active:
            while more and len(active) < self.batch:
                job = next(jobs, None)
                if job is None:
                    more = False
                    break
                try:
                    active.append(Slot(*job))
                except Exception as exc:
                    if on_done:
                        on_done(job[0], job[1], f'{type(exc).__name__}: {exc}', 0)
            imgs, live = [], []
            for s in active:
                img = s.dec.q.get()
                if img is None:
                    s.output.parent.mkdir(parents=True, exist_ok=True)
                    s.rec.save(s.output)
                    if on_done:
                        on_done(s.video, s.output, None, s.idx)
                    continue
                if s.W is None:
                    s.H, s.W = img.shape[:2]
                elif img.shape[:2] != (s.H, s.W):
                    s.dec.close()
                    if on_done:
                        on_done(s.video, s.output, f'frame {s.idx} is {img.shape[1]}x{img.shape[0]}, expected {s.W}x{s.H}', s.idx)
                    continue
                imgs.append(img)
                live.append(s)
            active = live
            if not active:
                continue
            ts = []
            for s in active:                         # MediaPipe's VIDEO-mode timestamps (ms), strictly increasing
                t = max(int(s.idx * 1000.0 / s.dec.fps), s.prev_ts + 1)
                s.prev_ts = t
                ts.append(t)
            sizes = [img.shape[:2] for img in imgs]
            Hm, Wm = max(h for h, _ in sizes), max(w for _, w in sizes)
            canvas = torch.zeros((len(imgs), Hm, Wm, 3), dtype=torch.uint8).pin_memory()
            for k, img in enumerate(imgs):
                canvas[k, :img.shape[0], :img.shape[1]] = torch.from_numpy(img)
            canvas = canvas.cuda(non_blocking=True)
            poses = self.pose.run([s.pose for s in active], canvas, sizes, [t * 1_000_000 for t in ts])
            faces = self.face.run([s.face for s in active], imgs, poses, ts)
            frames = [dict(rgb=img, W=s.W, H=s.H, pose_ok=p is not None,
                           pose_xyz=p['pose_xyz'] if p else np.full((33, 3), np.nan, np.float32),
                           pose_vis=p['pose_vis'] if p else np.zeros(33, np.float32),
                           pose_world=p['pose_world'] if p else None)
                      for s, img, p in zip(active, imgs, poses)]
            hand_rows = self.hands.run(frames)
            for s, p, fc, hr in zip(active, poses, faces, hand_rows):
                row = hp.FrameRow()
                if p is not None:
                    row.pose_xyz, row.pose_vis, row.pose_pres, row.pose_world = p['pose_xyz'], p['pose_vis'], p['pose_pres'], p['pose_world']
                    row.pose_ok = True
                if fc is not None:
                    row.face_xyz, row.face_ok = fc, True
                for k, v in hr.items():
                    setattr(row, k, v)
                s.rec.append(row)
                s.idx += 1
