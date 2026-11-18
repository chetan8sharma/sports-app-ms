import os
import cv2
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
_LEFT_HIP        = 23
_RIGHT_HIP       = 24
_LEFT_ANKLE      = 27
_RIGHT_ANKLE     = 28
_LEFT_HEEL       = 29
_RIGHT_HEEL      = 30
_LEFT_SHOULDER   = 11
_RIGHT_SHOULDER  = 12

# ── Physical calibration ──────────────────────────────────────────────────────
_VIDEO_WIDTH_CM = 1000.0


class LateralShuffleService(BaseVideoService):
    """
    Measures performance metrics for a lateral shuffle (side-step) exercise
    from a front-facing video.

    Algorithm overview
    ------------------
    The athlete shuffles laterally (left <-> right). We track the horizontal (X)
    position of the mid-hip point across all frames to build a 1-D signal that
    oscillates between two boundary positions.

    Two-pass strategy
    -----------------
    Pass 1  -- collect mid-hip X, mid-shoulder X, and foot data for every
               frame where a pose is detected.
    Pass 2  -- smooth the hip-X signal, find direction reversals (peaks/troughs),
               classify each reversal as a left or right boundary touch, then
               derive all metrics.

    Metrics returned
    ----------------
    - shuffle_count            : total direction changes
    - shuffles_per_side        : {"left": n, "right": n}
    - avg_shuffle_width_cm     : average lateral distance between consecutive touches
    - left_avg_width_cm        : average width of shuffles ending on the left
    - right_avg_width_cm       : average width of shuffles ending on the right
    - symmetry_pct             : how symmetric left vs right widths are (100 = perfect)
    - avg_shuffle_speed_cm_s   : average lateral speed during active shuffle phases
    - avg_transition_time_s    : average time to complete one shuffle (touch to touch)
    - cadence_shuffles_min     : shuffles per minute over the active period
    - active_duration_s        : total active shuffle time
    - touches                  : list of individual boundary-touch events
    - shuffles                 : list of individual shuffle segments with metadata
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
        frames_data = []

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts  = int((frame_count / fps) * 1000)
                res = landmarker.detect_for_video(img, ts)

                if not res.pose_landmarks or len(res.pose_landmarks) == 0:
                    continue

                lm = res.pose_landmarks[0]

                # Mid-hip X is the primary lateral-position signal.
                mid_hip_x = (lm[_LEFT_HIP].x + lm[_RIGHT_HIP].x) / 2.0

                # Mid-shoulder X used to cross-validate body orientation.
                mid_shoulder_x = (lm[_LEFT_SHOULDER].x + lm[_RIGHT_SHOULDER].x) / 2.0

                frames_data.append({
                    "frame":          frame_count,
                    "time_s":         frame_count / fps,
                    "mid_hip_x":      mid_hip_x,
                    "mid_shoulder_x": mid_shoulder_x,
                    "left_heel_x":    lm[_LEFT_HEEL].x,
                    "right_heel_x":   lm[_RIGHT_HEEL].x,
                    "left_heel_y":    lm[_LEFT_HEEL].y,
                    "right_heel_y":   lm[_RIGHT_HEEL].y,
                    "left_ankle_x":   lm[_LEFT_ANKLE].x,
                    "right_ankle_x":  lm[_RIGHT_ANKLE].x,
                })

        cap.release()

        if len(frames_data) < 20:
            raise ValueError(
                "Not enough pose detections in the video. "
                "Ensure the full body is visible and well-lit throughout."
            )

        # ── Validate that meaningful lateral movement exists ──────────────────
        hip_xs    = [d["mid_hip_x"] for d in frames_data]
        hip_range = max(hip_xs) - min(hip_xs)

        if hip_range < 0.05:
            raise ValueError(
                "Insufficient lateral movement detected (hip X range < 5% of frame). "
                "Ensure the athlete shuffles clearly side-to-side within the camera frame."
            )

        # ── Smooth hip-X signal to suppress jitter ────────────────────────────
        # Box-kernel moving average over ~0.12 s removes per-frame noise while
        # preserving the macro-oscillation caused by direction changes.
        smooth_window  = max(3, int(fps * 0.12))
        hip_xs_smooth  = self._smooth(hip_xs, smooth_window)

        # ── Pass 2: detect direction reversals (boundary touches) ─────────────
        # A reversal is a local min/max in the smoothed hip-X signal.
        # MIN_REVERSAL_SEP prevents double-firing on very fast athletes.
        # A typical shuffle is 0.5-2 s, so 0.25 s is a safe lower bound.
        MIN_REVERSAL_SEP = max(5, int(fps * 0.25))
        MIN_SHUFFLE_CM   = 15.0
        MAX_SHUFFLE_CM   = 600.0

        reversal_indices = self._find_reversals(hip_xs_smooth, MIN_REVERSAL_SEP)

        if len(reversal_indices) < 2:
            raise ValueError(
                "Fewer than 2 direction reversals detected. "
                "Ensure the athlete completes at least 2 full lateral shuffles "
                "with their whole body visible throughout."
            )

        # ── Build touch events ────────────────────────────────────────────────
        touches = []
        for rev_idx in reversal_indices:
            fd       = frames_data[rev_idx]
            hip_x_n  = hip_xs_smooth[rev_idx]          # normalised [0, 1]
            hip_x_cm = hip_x_n * frame_width / px_per_cm

            touches.append({
                "frame":      fd["frame"],
                "time_s":     round(fd["time_s"], 3),
                "hip_x_norm": round(hip_x_n, 4),
                "hip_x_cm":   round(hip_x_cm, 2),
            })

        # Label each touch "left" or "right" relative to the median hip position.
        # Touches below the median are on the left side of the frame; above = right.
        # Note: in normalised image coords X=0 is the left edge, X=1 is the right.
        median_x = self._percentile(hip_xs_smooth, 50)
        for t in touches:
            t["side"] = "left" if t["hip_x_norm"] < median_x else "right"

        # ── Shuffle segments: each consecutive touch pair is one shuffle ───────
        shuffles = []
        for i in range(1, len(touches)):
            prev = touches[i - 1]
            curr = touches[i]

            delta_norm = abs(curr["hip_x_norm"] - prev["hip_x_norm"])
            width_cm   = delta_norm * frame_width / px_per_cm

            if width_cm < MIN_SHUFFLE_CM or width_cm > MAX_SHUFFLE_CM:
                continue

            dt         = curr["time_s"] - prev["time_s"]
            speed_cm_s = round(width_cm / dt, 2) if dt > 0 else None
            direction  = (
                "right_to_left"
                if curr["hip_x_norm"] < prev["hip_x_norm"]
                else "left_to_right"
            )

            shuffles.append({
                "shuffle_number":    len(shuffles) + 1,
                "direction":         direction,
                "from_side":         prev["side"],
                "to_side":           curr["side"],
                "from_time_s":       prev["time_s"],
                "to_time_s":         curr["time_s"],
                "width_cm":          round(width_cm, 2),
                "transition_time_s": round(dt, 3),
                "speed_cm_s":        speed_cm_s,
            })

        if not shuffles:
            raise ValueError(
                "Direction changes were detected but no valid shuffle widths could be "
                "measured. Ensure the athlete moves clearly and stays within the frame."
            )

        # ── Aggregate metrics ─────────────────────────────────────────────────
        all_widths  = [s["width_cm"]          for s in shuffles]
        all_times   = [s["transition_time_s"] for s in shuffles]
        all_speeds  = [s["speed_cm_s"]        for s in shuffles if s["speed_cm_s"]]

        left_widths  = [s["width_cm"] for s in shuffles if s["to_side"] == "left"]
        right_widths = [s["width_cm"] for s in shuffles if s["to_side"] == "right"]

        avg_shuffle_width_cm  = round(sum(all_widths) / len(all_widths), 2)
        avg_transition_time_s = round(sum(all_times)  / len(all_times),  3)
        avg_speed_cm_s        = (
            round(sum(all_speeds) / len(all_speeds), 2) if all_speeds else None
        )

        left_avg_width_cm = (
            round(sum(left_widths) / len(left_widths), 2) if left_widths else None
        )
        right_avg_width_cm = (
            round(sum(right_widths) / len(right_widths), 2) if right_widths else None
        )

        # Symmetry: 100% = perfectly equal left/right shuffle widths.
        # Formula: 100 x (1 - |L - R| / max(L, R))
        if left_avg_width_cm and right_avg_width_cm:
            max_w    = max(left_avg_width_cm, right_avg_width_cm)
            symmetry = round(
                100 * (1 - abs(left_avg_width_cm - right_avg_width_cm) / max_w), 1
            )
        else:
            symmetry = None

        # Cadence: shuffles per minute over the active window (first to last touch)
        active_start = shuffles[0]["from_time_s"]
        active_end   = shuffles[-1]["to_time_s"]
        active_dur_s = active_end - active_start
        cadence = (
            round(len(shuffles) / active_dur_s * 60, 1)
            if active_dur_s > 0 else None
        )

        shuffles_per_side = {
            "left":  sum(1 for s in shuffles if s["to_side"] == "left"),
            "right": sum(1 for s in shuffles if s["to_side"] == "right"),
        }

        return {
            "touches":                touches,
            "shuffles":               shuffles,
            "shuffle_count":          len(shuffles),
            "shuffles_per_side":      shuffles_per_side,
            "avg_shuffle_width_cm":   avg_shuffle_width_cm,
            "left_avg_width_cm":      left_avg_width_cm,
            "right_avg_width_cm":     right_avg_width_cm,
            "symmetry_pct":           symmetry,
            "avg_shuffle_speed_cm_s": avg_speed_cm_s,
            "avg_transition_time_s":  avg_transition_time_s,
            "cadence_shuffles_min":   cadence,
            "active_duration_s":      round(active_dur_s, 2),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _smooth(values, window):
        """
        Box-kernel moving average. Edge values use a shrinking kernel so that
        every position in the output is always defined.
        """
        n      = len(values)
        half   = window // 2
        result = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            result.append(sum(values[lo:hi]) / (hi - lo))
        return result

    @staticmethod
    def _find_reversals(signal, min_sep):
        """
        Return indices of alternating local maxima and minima in `signal`,
        enforcing a minimum frame separation of `min_sep` between successive
        reversals.

        Steps
        -----
        1. Scan for local extrema (a point strictly higher/lower than both
           immediate neighbours).
        2. Alternate between searching for a maximum and a minimum so every
           consecutive pair represents a genuine direction change.
        3. Debounce: after recording a reversal, skip all extrema within the
           next `min_sep` frames.
        """
        n         = len(signal)
        reversals = []
        last_idx  = -min_sep

        # Decide whether to look for a peak or trough first by comparing the
        # signal value at the very first and very last detected frame.
        looking_for_peak = signal[0] < signal[-1]

        for i in range(1, n - 1):
            if i - last_idx < min_sep:
                continue

            is_peak   = signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]
            is_trough = signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]

            if looking_for_peak and is_peak:
                reversals.append(i)
                last_idx         = i
                looking_for_peak = False
            elif not looking_for_peak and is_trough:
                reversals.append(i)
                last_idx         = i
                looking_for_peak = True

        return reversals

    @staticmethod
    def _percentile(values, pct):
        """Return the pct-th percentile of a list of floats (no numpy needed)."""
        sorted_vals = sorted(values)
        idx = (pct / 100) * (len(sorted_vals) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)