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

# ── Landmark indices ──────────────────────────────────────────────────────────
_NOSE             = 0
_LEFT_HIP         = 23
_RIGHT_HIP        = 24
_LEFT_FOOT_INDEX  = 31   # tip of left foot (big-toe side)
_RIGHT_FOOT_INDEX = 32   # tip of right foot (big-toe side)

# ── Known video width ─────────────────────────────────────────────────────────
_VIDEO_WIDTH_CM = 1000.0   # camera is set up so the full frame width = 10 m


class HorizontalJumpService(BaseVideoService):
    """
    Measures standing-broad-jump / horizontal jump distance from a side-view video.

    The camera must be perpendicular to the jump direction and framed so the
    full frame width represents exactly 10 metres.  No extra form fields are
    required — calibration is derived from the frame width alone.

    Returned dict
    -------------
    jump_distance_cm : float  – horizontal distance from takeoff toe to landing toe
    flight_time_s    : float  – time between takeoff and landing frames
    """

    # BaseVideoService.process() handles file I/O and MD5 integrity check.
    # We do NOT override process() because we need no extra kwargs.

    def _run_analysis(self, video_path, filename, **kwargs):

        # ── Open video ────────────────────────────────────────────────────────
        cap         = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_width == 0:
            cap.release()
            raise ValueError("Could not read frame dimensions from video.")

        # ── Pixel-to-cm calibration (horizontal axis only) ────────────────────
        # Full frame width = 10 m = 1000 cm  →  1 px = 1000 / frame_width cm
        px_per_cm = frame_width / _VIDEO_WIDTH_CM   # px per cm (inverse of cm per px)
        # We'll use:  distance_cm = delta_px / px_per_cm

        # ── Build Tasks API landmarker ────────────────────────────────────────
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

        # ── Thresholds (mirroring vertical_leap.py) ───────────────────────────
        CALIBRATION_FRAMES = 30
        TAKEOFF_CM         = 3.0    # hip must rise this many cm to count as airborne
        LANDING_CM         = 2.0    # hip must drop back within this many cm of standing
        MAX_FLIGHT_S       = 2.0    # sanity cap on flight time
        TAKEOFF_STREAK     = 2      # consecutive frames above threshold → takeoff
        LANDING_STREAK     = 3      # consecutive frames below threshold → landing
        SMOOTH_N           = 2      # hip-Y smoothing window

        # ── State ─────────────────────────────────────────────────────────────
        standing_hip_y        = None
        frame_count           = 0
        takeoff_frame         = None
        landing_frame         = None
        phase                 = "CALIBRATING"
        airborne_streak       = 0
        grounded_streak       = 0
        pending_takeoff_frame = None
        smoothed_hip_y        = None

        # Toe-X positions recorded at key moments
        takeoff_toe_x = None   # normalised [0–1] x of frontmost foot at takeoff
        landing_toe_x = None   # normalised [0–1] x of frontmost foot at landing

        calib_hip_y  = []
        hip_buffer   = deque(maxlen=SMOOTH_N)

        def _frontmost_toe_x(lm):
            """
            Return the normalised X coordinate of whichever foot index landmark
            is further in the jump direction.  In a side-view left-to-right jump
            that is the larger X value; for right-to-left it is the smaller.
            We don't know the direction yet during calibration, so we store the
            direction-agnostic value and resolve it after the jump is detected.
            Both foot landmarks are returned so the caller can pick max or min.
            """
            lx = lm[_LEFT_FOOT_INDEX].x
            rx = lm[_RIGHT_FOOT_INDEX].x
            return lx, rx

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((frame_count / fps) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                    continue

                lm = result.pose_landmarks[0]

                raw_hip_y = (lm[_LEFT_HIP].y + lm[_RIGHT_HIP].y) / 2

                # ── Calibration phase ──────────────────────────────────────────
                if frame_count <= CALIBRATION_FRAMES:
                    calib_hip_y.append(raw_hip_y)

                    if frame_count == CALIBRATION_FRAMES:
                        calib_hip_y.sort()
                        mid = CALIBRATION_FRAMES // 2
                        standing_hip_y = calib_hip_y[mid]
                        phase          = "READY"

                    continue

                # ── Detection phase ────────────────────────────────────────────
                if standing_hip_y is None:
                    continue

                hip_buffer.append(raw_hip_y)
                smoothed_hip_y = sum(hip_buffer) / len(hip_buffer)

                # rise_cm: positive = hips moved up relative to standing position
                # MediaPipe Y is normalised 0→1 top-to-bottom, so a smaller Y
                # means higher in the frame.  We multiply by frame_height to get
                # pixels, then divide by px_per_cm.
                rise_cm = (standing_hip_y - smoothed_hip_y) * frame_height / px_per_cm

                # Flight-time sanity cap
                if takeoff_frame and not landing_frame:
                    elapsed = (frame_count - takeoff_frame) / fps
                    if elapsed > MAX_FLIGHT_S:
                        takeoff_frame         = None
                        takeoff_toe_x         = None
                        pending_takeoff_frame = None
                        airborne_streak       = 0
                        hip_buffer.clear()
                        phase = "READY"

                # ── Takeoff detection ──────────────────────────────────────────
                if not takeoff_frame:
                    if rise_cm >= TAKEOFF_CM:
                        airborne_streak += 1
                        if airborne_streak == 1:
                            pending_takeoff_frame = frame_count
                        if airborne_streak >= TAKEOFF_STREAK:
                            takeoff_frame = pending_takeoff_frame
                            lx, rx        = _frontmost_toe_x(lm)
                            # Store both; we'll resolve jump direction after landing
                            takeoff_toe_x = (lx, rx)
                            phase         = "AIRBORNE"
                    else:
                        airborne_streak       = 0
                        pending_takeoff_frame = None

                # ── Landing detection ──────────────────────────────────────────
                elif not landing_frame:
                    if rise_cm < LANDING_CM:
                        grounded_streak += 1
                        if grounded_streak >= LANDING_STREAK:
                            landing_frame = frame_count - LANDING_STREAK + 1
                            lx, rx        = _frontmost_toe_x(lm)
                            landing_toe_x = (lx, rx)
                            phase         = "LANDED"
                    else:
                        grounded_streak = 0

        cap.release()

        if not takeoff_frame or not landing_frame:
            raise ValueError(
                "No jump detected in the video. "
                "Ensure the person is fully visible and standing still for the first second."
            )

        # ── Resolve jump direction and pick frontmost toe ─────────────────────
        # The athlete jumps toward one side of the frame.  We determine direction
        # by comparing the average foot-X at takeoff vs landing.
        takeoff_avg_x = (takeoff_toe_x[0] + takeoff_toe_x[1]) / 2
        landing_avg_x = (landing_toe_x[0] + landing_toe_x[1]) / 2

        if landing_avg_x >= takeoff_avg_x:
            # Jumping left → right: larger X = further forward
            takeoff_x_norm = max(takeoff_toe_x)
            landing_x_norm = max(landing_toe_x)
        else:
            # Jumping right → left: smaller X = further forward
            takeoff_x_norm = min(takeoff_toe_x)
            landing_x_norm = min(landing_toe_x)

        # Convert normalised X difference to cm
        delta_px          = abs(landing_x_norm - takeoff_x_norm) * frame_width
        jump_distance_cm  = delta_px / px_per_cm

        flight_time_s = (landing_frame - takeoff_frame) / fps

        return {
            "jump_distance_cm": round(jump_distance_cm, 2),
            "flight_time_s":    round(flight_time_s, 3),
        }