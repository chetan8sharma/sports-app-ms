import os
import cv2
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, RunningMode
import mediapipe as mp

from .base import BaseVideoService

# ── Model path ────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'pose_landmarker_heavy.task')

# ── Landmark indices (Tasks API uses plain integers, same as solutions API) ───
_NOSE        = 0
_LEFT_HIP    = 23
_RIGHT_HIP   = 24
_LEFT_ANKLE  = 27
_RIGHT_ANKLE = 28


class VerticalLeapService(BaseVideoService):

    def process(self, video_file, filename, **kwargs):
        """Override to thread person_height_cm through to _run_analysis."""
        import tempfile, hashlib

        suffix     = os.path.splitext(filename)[-1] or '.mp4'
        video_bytes = video_file.read()

        upload_md5 = hashlib.md5(video_bytes).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as f:
            written_md5 = hashlib.md5(f.read()).hexdigest()

        if upload_md5 != written_md5:
            os.remove(tmp_path)
            raise ValueError("File integrity check failed during upload. Please retry.")

        try:
            return self._run_analysis(tmp_path, filename, **kwargs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _run_analysis(self, video_path, filename, person_height_cm=None, **kwargs):
        if person_height_cm is None:
            raise ValueError("person_height_cm is required for vertical leap analysis.")

        # ── Open video ────────────────────────────────────────────────────────
        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── Build Tasks API landmarker (VIDEO mode = frame-by-frame, no callback) ─
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
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Thresholds ────────────────────────────────────────────────────────
        CALIBRATION_FRAMES = 30
        TAKEOFF_CM         = 3.0
        LANDING_CM         = 2.0
        MAX_FLIGHT_S       = 1.5
        TAKEOFF_STREAK     = 2
        LANDING_STREAK     = 3
        SMOOTH_N           = 2

        # ── State ─────────────────────────────────────────────────────────────
        px_per_cm             = None
        standing_hip_y        = None
        frame_count           = 0
        takeoff_frame         = None
        landing_frame         = None
        phase                 = "CALIBRATING"
        airborne_streak       = 0
        grounded_streak       = 0
        pending_takeoff_frame = None
        smoothed_hip_y        = None

        calib_hip_y   = []
        calib_head_y  = []
        calib_ankle_y = []
        hip_buffer    = deque(maxlen=SMOOTH_N)

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Tasks API requires RGB + a timestamp in milliseconds
                rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((frame_count / fps) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # No pose detected this frame — skip
                if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                    continue

                lm = result.pose_landmarks[0]  # first (only) pose

                raw_hip_y = (lm[_LEFT_HIP].y  + lm[_RIGHT_HIP].y)  / 2
                nose_y    =  lm[_NOSE].y
                avg_ankle = (lm[_LEFT_ANKLE].y + lm[_RIGHT_ANKLE].y) / 2

                # ── Calibration phase ──────────────────────────────────────────
                if frame_count <= CALIBRATION_FRAMES:
                    calib_hip_y.append(raw_hip_y)
                    calib_head_y.append(nose_y)
                    calib_ankle_y.append(avg_ankle)

                    if frame_count == CALIBRATION_FRAMES:
                        calib_hip_y.sort()
                        calib_head_y.sort()
                        calib_ankle_y.sort()
                        mid = CALIBRATION_FRAMES // 2

                        standing_hip_y   = calib_hip_y[mid]
                        person_px_height = (calib_ankle_y[mid] - calib_head_y[mid]) * height
                        px_per_cm        = person_px_height / person_height_cm
                        phase            = "READY"

                    continue  # never run detection logic during calibration

                # ── Detection phase ────────────────────────────────────────────
                if px_per_cm is None:
                    # All calibration frames missed — skip
                    continue

                hip_buffer.append(raw_hip_y)
                smoothed_hip_y = sum(hip_buffer) / len(hip_buffer)
                rise_cm = (standing_hip_y - smoothed_hip_y) * height / px_per_cm

                # Flight-time sanity cap
                if takeoff_frame and not landing_frame:
                    elapsed = (frame_count - takeoff_frame) / fps
                    if elapsed > MAX_FLIGHT_S:
                        takeoff_frame         = None
                        pending_takeoff_frame = None
                        airborne_streak       = 0
                        hip_buffer.clear()
                        phase = "READY"

                # Takeoff detection
                if not takeoff_frame:
                    if rise_cm >= TAKEOFF_CM:
                        airborne_streak += 1
                        if airborne_streak == 1:
                            pending_takeoff_frame = frame_count
                        if airborne_streak >= TAKEOFF_STREAK:
                            takeoff_frame = pending_takeoff_frame
                            phase         = "AIRBORNE"
                    else:
                        airborne_streak       = 0
                        pending_takeoff_frame = None

                # Landing detection
                elif not landing_frame:
                    if rise_cm < LANDING_CM:
                        grounded_streak += 1
                        if grounded_streak >= LANDING_STREAK:
                            landing_frame = frame_count - LANDING_STREAK + 1
                            phase         = "LANDED"
                    else:
                        grounded_streak = 0

        cap.release()

        if not takeoff_frame or not landing_frame:
            raise ValueError(
                "No jump detected in the video. "
                "Ensure the person is fully visible and standing still for the first second."
            )

        flight_time_s  = (landing_frame - takeoff_frame) / fps
        jump_height_cm = (9.81 * flight_time_s ** 2 / 8) * 100

        return {
            "jump_height_cm": round(jump_height_cm, 2),
            "flight_time_s":  round(flight_time_s, 3),
        }