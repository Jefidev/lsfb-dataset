from cv2 import cv2
from os import path
from typing import Optional
import pandas as pd
from .features import (
    draw_holistic_landmarks,
    draw_holistic_boxes,
    get_holistic_features_img,
    get_annotations_img,
    draw_pose_landmarks,
    draw_hands_landmarks,
    draw_face_landmarks,
)


class VideoPlayer:
    def __init__(self, video_filepath: str):
        if not path.isfile(video_filepath):
            raise FileNotFoundError(f"Video file not found: {video_filepath}")

        self.video_path: str = video_filepath
        self.draw_landmarks: bool = True
        self.draw_boxes: bool = True
        self.draw_duration: bool = True

        self.holistic_features: Optional[pd.DataFrame] = None

        self.right_hand_annotations: Optional[pd.DataFrame] = None
        self.left_hand_annotations: Optional[pd.DataFrame] = None

        self.pose_features: Optional[pd.DataFrame] = None
        self.hands_features: Optional[pd.DataFrame] = None
        self.face_features: Optional[pd.DataFrame] = None

    def attach_holistic_features(self, holistic_filepath: str):
        if not path.isfile(holistic_filepath):
            raise FileNotFoundError(
                f"Holistic features file not found: {holistic_filepath}"
            )
        self.holistic_features = pd.read_csv(holistic_filepath)

    def attach_annotations(self, annotations_filepath: str, hand: str):
        if not path.isfile(annotations_filepath):
            raise FileNotFoundError(
                f"Annotations (CSV) file not found: {annotations_filepath}"
            )

        if hand == "right":
            self.right_hand_annotations = pd.read_csv(annotations_filepath)
        elif hand == "left":
            self.left_hand_annotations = pd.read_csv(annotations_filepath)
        else:
            raise ValueError(f"Unknown hand: {hand}")

    def attach_pose_landmarks(self, pose_filepath: str):
        if not path.isfile(pose_filepath):
            raise FileNotFoundError(f"Pose (CSV) file not found: {pose_filepath}")
        self.pose_features = pd.read_csv(pose_filepath)

    def attach_hands_landmarks(self, hands_filepath: str):
        if not path.isfile(hands_filepath):
            raise FileNotFoundError(f"Hands (CSV) file not found: {hands_filepath}")
        self.hands_features = pd.read_csv(hands_filepath)

    def attach_face_landmarks(self, face_landmarks: str):
        if not path.isfile(face_landmarks):
            raise FileNotFoundError(f"Face (CSV) file not found: {face_landmarks}")
        self.face_features = pd.read_csv(face_landmarks)

    def show_landmarks(self, value: bool):
        self.draw_landmarks = value

    def show_boxes(self, value: bool):
        self.draw_boxes = value

    def show_duration(self, value: bool):
        self.draw_duration = value

    def play(self):
        self._process_video()

    def save(self, filepath: str):
        self._process_video(filepath)

    def _process_video(self, save_path: str = None):

        cap = cv2.VideoCapture(self.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        current_frame = 0
        delay = int(1000 / frame_rate)
        total_duration = int(frame_count * 1000 / frame_rate)
        current_time = 0

        if save_path is not None:
            out = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame_rate,
                (int(frame_width), int(frame_height)),
            )

        while current_frame < frame_count and cap.isOpened():
            success, frame = cap.read()

            if not success:
                break
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

            try:
                processed_frame = self._process_frame(
                    frame, current_frame, current_time, total_duration
                )

                if save_path is not None:
                    out.write(processed_frame)
                else:
                    cv2.imshow("Video", processed_frame)

            except IndexError as e:
                # Index error could happen if the video is longer than the features dataframe
                print("Warning : The annotation are too short for the video.")
                break

            current_frame += 1
            current_time += delay

        cv2.destroyAllWindows()
        cap.release()

    def _process_frame(self, frame, current_frame, current_time, total_duration):
        if self.pose_features is not None:
            draw_pose_landmarks(
                frame, self.pose_features.iloc[current_frame].values.reshape((-1, 2))
            )
        if self.hands_features is not None:
            draw_hands_landmarks(
                frame, self.hands_features.iloc[current_frame].values.reshape((-1, 2))
            )
        if self.face_features is not None:
            draw_face_landmarks(
                frame, self.face_features.iloc[current_frame].values.reshape((-1, 2))
            )

        if self.holistic_features is not None:
            cv2.imshow(
                "Features",
                get_holistic_features_img(
                    frame, self.holistic_features.iloc[current_frame]
                ),
            )
            if self.draw_landmarks:
                draw_holistic_landmarks(
                    frame, self.holistic_features.iloc[current_frame]
                )
            if self.draw_boxes:
                draw_holistic_boxes(frame, self.holistic_features.iloc[current_frame])

        if self.right_hand_annotations is not None:
            cv2.imshow(
                "Right hand annotations",
                get_annotations_img(self.right_hand_annotations, current_time),
            )
        if self.left_hand_annotations is not None:
            cv2.imshow(
                "Left hand annotations",
                get_annotations_img(self.left_hand_annotations, current_time),
            )

        if self.draw_duration:
            self._show_duration(frame, current_time, total_duration)

        return frame

    def _show_duration(self, frame, current_time: int, total_duration: int):
        duration_info = f"{self._duration_to_str(current_time)} / {self._duration_to_str(total_duration)}"
        cv2.putText(
            frame,
            duration_info,
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            2,
        )

    def _duration_to_str(self, duration):
        milli = duration % 1000
        seconds = (duration // 1000) % 60
        minutes = duration // 60000
        return f"{minutes}min {seconds:2}s {milli:3}ms"
