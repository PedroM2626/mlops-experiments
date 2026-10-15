import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import vision_task_running_mode


POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), (24, 26), (26, 28),
    (26, 30), (26, 32), (28, 30), (0, 23), (0, 24), (1, 11), (2, 12), (3, 11), (4, 12)
])


class HolisticDetector:
    def __init__(self):
        """Initialize MediaPipe Holistic detector"""
        base_options = python.BaseOptions(model_asset_buffer=self._get_model_buffer())
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            output_segmentation_mask=False
        )
        self.holistic = vision.HolisticLandmarker.create_from_options(options)

    def _get_model_buffer(self):
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'holistic_landmarker.task')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return f.read()
        return None

    def detect(self, frame):
        """
        Detect holistic landmarks in a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            results: MediaPipe holistic results
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.holistic.detect(mp_image)

    def draw_landmarks(self, frame, results):
        """
        Draw landmarks on the frame

        Args:
            frame: Input frame
            results: MediaPipe holistic results

        Returns:
            frame: Frame with landmarks drawn
        """
        if results is None:
            return frame

        if results.pose_landmarks:
            self._draw_pose_landmarks(frame, results.pose_landmarks)

        return frame

    def _draw_pose_landmarks(self, frame, landmarks):
        if not landmarks:
            return

        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_x = int(start.x * frame.shape[1])
                start_y = int(start.y * frame.shape[0])
                end_x = int(end.x * frame.shape[1])
                end_y = int(end.y * frame.shape[0])

                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)

        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def get_landmark_data(self, results):
        """
        Extract landmark coordinates from results

        Args:
            results: MediaPipe holistic results

        Returns:
            dict: Dictionary containing landmark data
        """
        landmark_data = {
            'pose': None,
        }

        if results is not None and results.pose_landmarks:
            landmark_data['pose'] = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks])

        return landmark_data

    def release(self):
        """Release resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
