import os
import cv2
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode
import mediapipe as mp

from .base import BaseVideoService

# ── Model path ────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'pose_landmarker_heavy.task'
)

# ── Landmark indices ──────────────────────────────────────────────────────────
_LEFT_HIP         = 23
_RIGHT_HIP        = 24
_LEFT_HEEL        = 29
_RIGHT_HEEL       = 30
_LEFT_FOOT_INDEX  = 31
_RIGHT_FOOT_INDEX = 32

_VIDEO_WIDTH_CM = 1000.0


class StepLengthService(BaseVideoService):
    """
    Measures step length and stride length from a side-view walking/running video.

    Two-pass algorithm
    ------------------
    Pass 1 — collect every heel Y value and X value for the whole video.
    Pass 2 — detect foot contacts using thresholds derived from the full
             signal statistics, then compute step lengths from the X positions
             at each detected contact.

    Why two passes?
    One-pass approaches fail because any threshold set during a warmup window
    can be thrown off by the person standing still (very high Y) or already
    walking (compressed range). With the full signal we can compute a stable
    percentile-based threshold that adapts to whatever the video contains.

    Contact detection
    -----------------
    A foot is "grounded" when its heel Y exceeds the 55th percentile of all
    heel Y values observed for that foot. This percentile sits naturally in
    the middle of the grounded cluster for normal walking gait.
    A contact event fires on every grounded→airborne→grounded transition,
    debounced to avoid double-firing.
    """

    def _run_analysis(self, video_path, filename, **kwargs):

        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_width == 0:
            cap.release()
            raise ValueError("Could not read frame dimensions from video.")

        px_per_cm = frame_width / _VIDEO_WIDTH_CM

        model_path = os.path.normpath(_MODEL_PATH)
        if not os.path.exists(model_path):
            raise ValueError(
                f"Pose model not found at {model_path}. "
                "Run: curl -L -o videos/models/pose_landmarker_heavy.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )

        options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        # ── Pass 1: collect raw landmark data ────────────────────────────────
        # frames_data[i] = {"frame": int, "left_y", "left_x", "right_y", "right_x"}
        frames_data = []

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts   = int((frame_count / fps) * 1000)
                res  = landmarker.detect_for_video(img, ts)

                if not res.pose_landmarks or len(res.pose_landmarks) == 0:
                    continue

                lm = res.pose_landmarks[0]
                frames_data.append({
                    "frame":   frame_count,
                    "left_y":  lm[_LEFT_HEEL].y,
                    "left_x":  lm[_LEFT_HEEL].x,
                    "right_y": lm[_RIGHT_HEEL].y,
                    "right_x": lm[_RIGHT_HEEL].x,
                })

        cap.release()

        if len(frames_data) < 20:
            raise ValueError(
                "Not enough pose detections in the video. "
                "Ensure the full body is visible and well-lit throughout."
            )

        # ── Compute per-foot thresholds from full signal ──────────────────────
        left_ys  = [d["left_y"]  for d in frames_data]
        right_ys = [d["right_y"] for d in frames_data]

        # 55th percentile: sits in the grounded cluster for normal gait.
        # If the person walks the whole video, ~50-60% of frames have a foot grounded.
        left_thresh  = self._percentile(left_ys,  55)
        right_thresh = self._percentile(right_ys, 55)

        # Sanity: if the signal has almost no variation the video is probably
        # a person standing still — bail early with a clear message.
        if (max(left_ys) - min(left_ys)) < 0.02:
            raise ValueError(
                "Heel position shows almost no movement. "
                "Ensure the person is actively walking or running in the video."
            )

        # ── Pass 2: detect contacts ───────────────────────────────────────────
        DEBOUNCE_FRAMES = max(4, int(fps * 0.18))   # ~0.18 s minimum between contacts
        MIN_STEP_CM     = 8.0
        MAX_STEP_CM     = 250.0

        contacts = []

        foot_cfg = {
            "left":  {"thresh": left_thresh,  "y_key": "left_y",  "x_key": "left_x"},
            "right": {"thresh": right_thresh, "y_key": "right_y", "x_key": "right_x"},
        }
        state = {
            "left":  {"was_airborne": False, "last_contact_f": -999},
            "right": {"was_airborne": False, "last_contact_f": -999},
        }

        for d in frames_data:
            f = d["frame"]
            for side, cfg in foot_cfg.items():
                y           = d[cfg["y_key"]]
                thresh      = cfg["thresh"]
                is_grounded = y >= thresh

                if not is_grounded:
                    state[side]["was_airborne"] = True

                if is_grounded and state[side]["was_airborne"]:
                    if f - state[side]["last_contact_f"] > DEBOUNCE_FRAMES:
                        state[side]["last_contact_f"] = f
                        state[side]["was_airborne"]   = False
                        contacts.append({
                            "foot":   side,
                            "x_norm": d[cfg["x_key"]],
                            "frame":  f,
                        })

        if len(contacts) < 2:
            raise ValueError(
                "Fewer than 2 foot contacts detected. "
                "Ensure the person walks or runs clearly across the frame for at least "
                "2–3 full steps, with the whole body (including feet) visible throughout."
            )

        contacts.sort(key=lambda c: c["frame"])

        # ── Direction of travel ───────────────────────────────────────────────
        direction = 1 if contacts[-1]["x_norm"] >= contacts[0]["x_norm"] else -1

        # ── Step lengths ──────────────────────────────────────────────────────
        steps = []
        for i in range(1, len(contacts)):
            prev = contacts[i - 1]
            curr = contacts[i]

            delta_norm = (curr["x_norm"] - prev["x_norm"]) * direction
            step_cm    = delta_norm * frame_width / px_per_cm

            if step_cm < MIN_STEP_CM or step_cm > MAX_STEP_CM:
                continue

            steps.append({
                "step_number":    len(steps) + 1,
                "foot":           curr["foot"],
                "contact_frame":  curr["frame"],
                "contact_time_s": round(curr["frame"] / fps, 3),
                "step_length_cm": round(step_cm, 2),
            })

        if not steps:
            raise ValueError(
                "Contacts were detected but no valid step lengths could be measured. "
                "Check that the person moves clearly in one direction across the frame."
            )

        # ── Stride lengths ────────────────────────────────────────────────────
        stride_lengths = []
        for i in range(1, len(steps)):
            if steps[i]["foot"] != steps[i - 1]["foot"]:
                stride_lengths.append(
                    steps[i - 1]["step_length_cm"] + steps[i]["step_length_cm"]
                )

        avg_step_length_cm   = round(sum(s["step_length_cm"] for s in steps) / len(steps), 2)
        avg_stride_length_cm = (
            round(sum(stride_lengths) / len(stride_lengths), 2)
            if stride_lengths else None
        )

        elapsed_s = steps[-1]["contact_time_s"] - steps[0]["contact_time_s"]
        avg_cadence_steps_min = (
            round((len(steps) - 1) / elapsed_s * 60, 1)
            if elapsed_s > 0 and len(steps) >= 2
            else None
        )

        return {
            "steps":                  steps,
            "step_count":             len(steps),
            "avg_step_length_cm":     avg_step_length_cm,
            "avg_stride_length_cm":   avg_stride_length_cm,
            "avg_cadence_steps_min":  avg_cadence_steps_min,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _percentile(values, pct):
        """Return the pct-th percentile of a list of floats (no numpy needed)."""
        sorted_vals = sorted(values)
        idx = (pct / 100) * (len(sorted_vals) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)