import cv2
import numpy as np
import os
from scipy.signal import savgol_filter
from .base import BaseVideoService

class FlyRunService(BaseVideoService):

    def _run_analysis(self, video_path, filename):
        # 1. Initialize exactly as original
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Original logic used MOG2 with these specific defaults
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        raw_detections = [] 
        
        # 2. Pass 1: Tracking (Identical Loop)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Original logic: Build background for first 0.5s
            if i < fps/2:
                bg_subtractor.apply(frame, learningRate=0.1)
                continue
                
            pos = self._detect_runner_torso(frame, bg_subtractor)
            if pos:
                raw_detections.append((i, pos[0]))
                
        cap.release()
        
        if len(raw_detections) < 15:
            raise ValueError("Error: Could not track runner for a sufficient duration.")

        # 3. Pass 2: Calculation (Exact Math)
        frames, x_coords = zip(*raw_detections)
        
        # Calibration - This is where the discrepancy likely lived.
        # It uses the very first and very last successful detections.
        distance_meters = 17.0
        pixel_span = abs(x_coords[-1] - x_coords[0])
        
        # Avoid division by zero if runner didn't move
        if pixel_span == 0:
            raise ValueError("Error: No horizontal movement detected for calibration.")
            
        meters_per_pixel = distance_meters / pixel_span
        
        # Interpolation
        full_frame_range = np.arange(frames[0], frames[-1] + 1)
        full_x = np.interp(full_frame_range, frames, x_coords)
        
        # Smoothing
        window = int(fps * 0.5)
        if window % 2 == 0: window += 1
        if window >= len(full_x): window = len(full_x) // 2 * 2 - 1
        
        smooth_x = savgol_filter(full_x, window, polyorder=2)
        
        # Velocity (px/frame * fps = px/sec)
        velocity_px_sec = np.gradient(smooth_x) * fps
        
        speed_points = []
        for idx, frame_num in enumerate(full_frame_range):
            time_sec = frame_num / fps
            s_ms = abs(velocity_px_sec[idx] * meters_per_pixel)

            # Cap MUST happen before km/h conversion
            if s_ms > 13.0:
                s_ms = 13.0

            speed_points.append({
                "time_sec": round(float(time_sec), 4),
                "speed_ms": round(float(s_ms), 4),
                "speed_kmh": round(float(s_ms * 3.6), 4)   # ← uses capped s_ms
            })
        
        print(f"total detections: {len(raw_detections)}")
        print(f"first detection frame: {frames[0]}, x: {x_coords[0]}")
        print(f"last detection frame:  {frames[-1]}, x: {x_coords[-1]}")
        print(f"pixel_span: {pixel_span}")
        print(f"meters_per_pixel: {meters_per_pixel}")
        print(f"window: {window}")
        print(f"first speed: {speed_points[0]}")
        print(f"max speed: {max(p['speed_kmh'] for p in speed_points)}")

        return {filename: speed_points}

    def _detect_runner_torso(self, frame, bg_subtractor, min_area=1000):
        # Exact copy of the original detection logic
        fg_mask = bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < min_area:
            return None
        
        M = cv2.moments(largest_contour)
        if M["m00"] == 0: 
            return None
        
        cx = int(M["m10"] / M["m00"])
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (cx, y + int(h * 0.35))