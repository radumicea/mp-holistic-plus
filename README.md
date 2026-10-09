# mp_holistic_plus

Pose, face and hand landmarks of one person per video, made with MediaPipe's own
models. It runs on an NVIDIA GPU and processes many videos at once. It was built
for sign language and is the extractor of the RSL-News landmarks. Nothing in it is
specific to sign language: it suits any video of one person whose hands matter.

## Why

MediaPipe Holistic tracks each hand from frame to frame. When it loses a hand, it
waits for its palm detector to find the hand again. Fast hands, motion blur, and
hands held against the face or against each other make it lose hands for many
frames in a row. Sometimes it also puts the same hand in both the left and the right
slot. Sign language does all of this all the time.

This library finds each hand from the pose instead, in every frame:

1. A small network predicts the hand's box from the arm's pose points
   ([mediapipe-hand-crop-fix](https://github.com/sign-language-processing/mediapipe-hand-crop-fix)).
   MediaPipe's hand landmark model runs on that box.
2. The box is redrawn around the hand's own landmarks, and the model runs again.
   This happens twice.
3. If the hand model is unsure, MediaPipe's palm detector searches around the arm's
   hand. It also searches if the hand found is nearer the other arm.
4. Left and right come from the arm, so a hand cannot swap sides. A hand nearer the
   other arm's wrist is not accepted. If one hand is found in both slots, it is kept
   only on the nearer arm.

The pose comes from MediaPipe's PoseLandmarker with the heavy model; Holistic uses
the lite one. The face comes from MediaPipe's FaceLandmarker with its 256 px model,
where Holistic's is 192 px. It runs on a crop around the pose's face points, as in
Holistic.

Holistic also cannot run on the GPU. In mediapipe 1.0.1, HolisticLandmarker runs on
the CPU, but with the GPU delegate it fails in its face-blendshapes step ("Batch size
mismatch"), and that step cannot be left out. MediaPipe's separate pose, face and
hand landmarkers do run on the GPU, and this library is built from their models. So
it runs on the GPU and is more accurate than Holistic (see Results).

## Results

Measured on 480 frames from the RSL-News test split: 48 windows of sign-language
news across three TV channels. The comparison is MediaPipe's HolisticLandmarker.
RTMW-x served as an independent reference. Every case where the systems disagreed
was rated by eye without knowing which output was which. Other footage was not
measured.

- Hands missed or wrong: 5 of 955 hand slots (0.5%), against 99 (10.4%) for
  Holistic. It never put a hand in the wrong place or on the other hand; Holistic
  did so 5 times.
- Where both find a hand, both are equally close to RTMW (median 2.6 px).
- Shoulders, elbows and wrists are closer to RTMW: median 2.7, 4.3 and 4.5 px,
  against 3.4, 4.9 and 6.0. The one weak spot is an elbow hidden behind ProTV's
  logo. It is placed about 15 px from where RTMW and Holistic put it (9 of 160
  ProTV frames).

On an RTX 3060 laptop GPU it does 183 frames/s. The same method built from
MediaPipe's separate landmarkers with their GPU delegate does 67 frames/s, in its
fastest setup found (12 processes). That is capped because MediaPipe runs the heavy
pose model on the GPU one frame at a time, about 15 ms each.

## How it runs on the GPU

Tracking makes each frame of a video depend on the previous one, so the frames of
one video cannot be batched. This library steps through many videos together. Each
step takes the next frame of every video, and each model runs once per step on all
of them, as a TensorRT fp32 engine.

MediaPipe's pose and face landmarkers are rewritten from MediaPipe's C++ source:
detection, tracking, crops, heatmap refinement and smoothing. Each step computes the
same numbers as MediaPipe, including OpenCV's fixed-point crop, which is redone on
the GPU. The pose and face detectors and the palm detector run on the CPU. The
pose and face detectors run only when there is nothing to track; the palm detector
runs only for the hand search in step 3 above.

With `backend='litert'` the models run on the CPU with MediaPipe's own arithmetic.
The output was then checked byte-identical to the same method built from
mediapipe 1.0.1's Python tasks. With TensorRT, only the models' floating-point
rounding differs. On 19 clips this changed the results about as much as changing
0.5% of the input pixels by ±1. fp16 and TF32 move pose points by 1 to 3 px, so
they are not used.

## Use

```python
from mp_holistic_plus.pipeline import HolisticPlus

h = HolisticPlus(batch=16)
h.run([('a.mp4', 'a.npz'), ('b.mp4', 'b.npz')])
h.close()
```

`run` takes any iterable of (video, output) pairs. It keeps up to `batch` videos
going and starts the next one as soon as one finishes. Pass
`on_done(video, output, error, frames)` to hear about each video; a video that
fails does not stop the others. On one GPU, 3 processes with 16 videos each is the
fastest setup.

## Settings

The settings are in `common.Options`; each has a comment there. The defaults are
what RSL-News uses. None was chosen for the best score on some data:

- The wrist must be visible (≥ 0.1) for its hand to be looked for, as in Holistic.
- `hand_refine_scale` (1.6) makes the refinement box the size MediaPipe's own hand
  tracking uses. Hand z depends strongly on that size. On the 15-frame clip it was
  measured on, 1.4 scored better, but it gives a box 19% smaller than MediaPipe's,
  so it was not chosen.
- Two hands count as one hand found twice (`common.hands_coincide`) when their
  landmarks lie within 0.25 hand sizes on average. On 19 broadcast clips, one hand
  found twice was within 0.1, and two real hands held close were 0.3 or more apart.
- `hand_presence_threshold` (0.5), `hand_reject_threshold` (0.2) and
  `fallback_roi_scale` (3.0) are round values; nothing records them being tuned.

## Output

One `.npz` per video, one row per frame. Image coordinates are normalized so that
the image spans [0, 1]; points outside the image fall outside it. z is as MediaPipe
gives it, on the scale of x. A part that was not found (pose, face or one hand) is NaN
in every point; a part that was found has all its points. Left and right are the
person's.

| array | shape, type | |
|---|---|---|
| `pose_xyz` | (T, 33, 3) float16 | MediaPipe's 33 pose points |
| `pose_vis`, `pose_pres` | (T, 33) uint8 | visibility and presence, 0 to 255 |
| `pose_world` | (T, 33, 3) float16 | in metres, centred between the hips |
| `pose_ok` | (T,) bool | pose found |
| `face_xyz` | (T, 478, 3) float16 | face mesh with irises |
| `face_ok` | (T,) bool | face found |
| `hand_xyz` | (T, 2, 21, 3) float16 | index 0 is the left hand, 1 the right |
| `hand_world` | (T, 2, 21, 3) float16 | in metres, wrist placed on the pose's world wrist |
| `hand_score` | (T, 2) uint8 | hand presence, 0 to 255; 0 where no hand was found |
| `hand_ok` | (T, 2) bool | hand found |
| `hand_roi` | (T, 2, 4) float32 | the hand's box: centre x, y, side (relative to the longer image side), rotation in degrees |

## Install

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    "mp-holistic-plus @ git+https://github.com/radumicea/mp-holistic-plus"
```

An NVIDIA GPU with a driver for CUDA 13. The package pins `tensorrt-cu13==10.15.1.29`,
`torch==2.11.0` (the CUDA 13 build, from the index above), `onnx==1.23.1`,
`mediapipe==1.0.1`, `ai-edge-litert==2.2.0`, `opencv-contrib-python==5.0.0.93` and
`numpy==2.2.6`. Keep the OpenCV version: the crops are made with OpenCV, and other
builds round them differently.

On first use, the MediaPipe bundles are downloaded to `~/.cache/mp_holistic_plus`
(or `$MP_HOLISTIC_PLUS_CACHE`) and checked against their sha256, and their models
are unpacked there. The TensorRT engines are then built for your GPU, which takes a
few minutes, and cached there too. Engines built by different TensorRT versions or
for different GPUs differ by rounding.

## Files

In `mp_holistic_plus/`:

- `pipeline.py`: `HolisticPlus`, which decodes the videos and runs the steps.
- `pose.py`, `face.py`, `hands.py`: the three stages.
- `common.py`: options, model files, hand-box network, crop geometry, hand checks,
  the output file (numpy and OpenCV only).
- `mpgeom.py`: MediaPipe's crop geometry, bit for bit, with the crop on the GPU.
- `tflite_torch.py`, `trt_engine.py`: a MediaPipe `.tflite` model turned into a
  TensorRT engine, through PyTorch and ONNX.
- `assets/hand_crop_mlp.npz`: the hand-box network's weights.

## License

Apache License 2.0 (`LICENSE`). It reimplements parts of MediaPipe (Apache 2.0), and the
hand-box network's weights come from mediapipe-hand-crop-fix (MIT); see `NOTICE`.
