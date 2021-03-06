from cv2 import cv2
from ..utils.holistic_features import absolute_position


def draw_line(img, start_pos, end_pos, color=(255, 255, 255), thickness=2):
    cv2.line(img, absolute_position(img, start_pos), absolute_position(img, end_pos), color, thickness)


def draw_connections(img, positions, connections, color=(255, 0, 0), thickness=2):
    for i, j in connections:
        x1, y1 = absolute_position(img, positions[i])
        x2, y2 = absolute_position(img, positions[j])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_points(img, positions, color=(0, 0, 255), thickness=cv2.FILLED):
    for pos in positions:
        x, y = absolute_position(img, pos)
        cv2.circle(img, (x, y), 3, color, thickness)


def draw_rect(img, pos1, pos2, color=(255, 255, 255), thickness=cv2.FILLED):
    cv2.rectangle(img, absolute_position(img, pos1), absolute_position(img, pos2), color=color, thickness=thickness)



