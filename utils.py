import os

import numpy as np
import cv2 as cv

from typing import NamedTuple


class BBox(NamedTuple):
    class_idx: int = None
    score: float = None
    # NOTE: we use float here to not loose
    # accuracy in math operations.
    # y-coord (row_idx) of the top-left corner
    top: float = 0
    # x-coord (col_idx) of the top-left corner
    left: float = 0
    # num_rows
    height: float = 0
    # num_cols
    width: float = 0

    def to_int(self):
        return BBox(
            class_idx=self.class_idx,
            score=self.score,
            top=int(self.top),
            left=int(self.left),
            height=int(self.height),
            width=int(self.width)
        )


def check_file_exist(file_path):
    if not os.path.isfile(file_path):
        raise Exception("ERROR: No such file: '%s'" % file_path)

def load_scaled_header(header_path, target_width):
    # Case 1: if we don't need header (i.e. header_path = None),
    # then returning header with zero height.
    if header_path is None:
        print("Warning: no header path!")
        return np.empty(shape=(0, target_width, 3), dtype=np.uint8)

    # Case 2: we need header.
    check_file_exist(header_path)
    header = cv.imread(header_path)
    scale = target_width / header.shape[1]
    scaled_header = cv.resize(header, None, fx=scale, fy=scale,
                              interpolation= cv.INTER_LINEAR)
    return scaled_header

def get_video_params(video_reader):
    num_frames = video_reader.get(cv.CAP_PROP_FRAME_COUNT)
    frame_width = video_reader.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = video_reader.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = video_reader.get(cv.CAP_PROP_FPS)
    return num_frames, frame_width, frame_height, fps

def build_video_reader(input_video_path):
    check_file_exist(input_video_path)
    return cv.VideoCapture(input_video_path)

def build_video_writer(frame_width, frame_height, fps, output_video_path):

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(output_video_path, fourcc, fps,
                                  (frame_width, frame_height))

    return video_writer

def choose_main_person_bbox(bboxes, score_threshold, dim_threshold):
    HEAD_CLASS_IDX = 1

    main_bbox = None
    max_square = -1
    for bbox in bboxes:

        if bbox.class_idx != HEAD_CLASS_IDX or bbox.score < score_threshold:
            continue

        if min(bbox.width, bbox.height) < dim_threshold:
            continue

        square = bbox.width * bbox.height
        if square > max_square:
            main_bbox = bbox
            max_square = square

    return main_bbox

def calc_max_possible_frame_size(
    video_frame_width,
    video_frame_height,
    frame_aspect_ratio
):
    max_frame_width_by_video_height = frame_aspect_ratio * video_frame_height
    max_frame_width = min(video_frame_width, max_frame_width_by_video_height)
    max_frame_height = max_frame_width / frame_aspect_ratio
    return max_frame_width, max_frame_height

def build_frame_bbox(
    head_bbox,
    frame_aspect_ratio,
    width_scale,
    top_offset_ratio,
    min_frame_width,
    video_frame_width,
    video_frame_height,
):

    # NOTE: assuming that min_frame_width <= video_frame_width
    # and corresponding frame_height <= video_frame_height.

    # 1. Calculating frame size.

    # 1.1 Desirable frame width.
    frame_width = head_bbox.width * width_scale
    # print("DES", frame_width)

    # 1.2 Calculating max possible frame width for specified
    # frame aspect ratio and video frame width and height.
    max_frame_width_by_video_height = frame_aspect_ratio * video_frame_height
    max_frame_width, _ = calc_max_possible_frame_size(video_frame_width,
                                                      video_frame_height,
                                                      frame_aspect_ratio)

    # 1.3 Calculating actual frame width.
    actual_width = max(min_frame_width, frame_width)
    actual_width = min(actual_width, max_frame_width)

    # 1.4 Calculating actual frame height.
    actual_height = actual_width / frame_aspect_ratio
    # print("ACTUAL SIZE", actual_width, actual_height)

    # 2. Calculating frame position.

    # 2.1 Desirable frame position.
    top_offset = top_offset_ratio * head_bbox.height
    frame_top = head_bbox.top - top_offset

    side_offset = (actual_width - head_bbox.width) / 2
    frame_left = head_bbox.left - side_offset

    # 2.2 Calculating actual frame position.
    min_top, min_left = 0, 0
    actual_top = max(min_top, frame_top)
    actual_left = max(min_left, frame_left)

    max_top = video_frame_height - actual_height
    max_left = video_frame_width - actual_width
    actual_top = min(max_top, actual_top)
    actual_left = min(max_left, actual_left)

    return BBox(top=actual_top, left=actual_left,
                height=actual_height, width=actual_width)


def build_output_frame(
    image,
    frame_bbox,
    output_frame_width,
    output_frame_height,
):
    # NOTE: Assuming that frame_bbox is completely inside image.

    # 1. Extracting subimage from frame_bbox.
    int_bbox = frame_bbox.to_int()
    frame_image = image[int_bbox.top:int_bbox.top + int_bbox.height,
                        int_bbox.left:int_bbox.left + int_bbox.width]
    # cv.imwrite("assets/test_crop.jpg", frame_image.astype(np.uint8))

    # 2. Resizing to fit target width.
    # NOTE: in perfect case we can use only width OR height
    # for scaling (because frame_bbox has aspect_ratio that we need).
    # But because of integer rounding (caused by pixels)
    # we need to use both dimensions to scale exactly to them.
    scaled_image = cv.resize(frame_image, (output_frame_width, output_frame_height),
                             interpolation= cv.INTER_LINEAR)
    # cv.imwrite("assets/test_scaled.jpg", scaled_image.astype(np.uint8))

    return scaled_image.astype(np.uint8)

def apply_lin_interp(x0, x1, k):
    return x0 + k * (x1 - x0)

def apply_bbox_lin_interp(bbox_0, bbox_1, k):
    return BBox(
        class_idx=bbox_1.class_idx,
        score=bbox_1.score,
        top=apply_lin_interp(bbox_0.top, bbox_1.top, k),
        left=apply_lin_interp(bbox_0.left, bbox_1.left, k),
        height=apply_lin_interp(bbox_0.height, bbox_1.height, k),
        width=apply_lin_interp(bbox_0.width, bbox_1.width, k),
    )

def render_bbox(image, bbox, color, show_score=False):
    int_bbox = bbox.to_int()
    border_thickness = 4
    cv.rectangle(image, (int_bbox.left, int_bbox.top),
                 (int_bbox.left + int_bbox.width, int_bbox.top + int_bbox.height),
                 color, border_thickness)
    left_offset = 5
    top_offset = 25
    font_scale = 0.75
    font_thickness = 2
    cv.putText(image, "size: %d x %d" % (int_bbox.width, int_bbox.height),
               (int_bbox.left + left_offset, int_bbox.top + top_offset),
               cv.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv.LINE_AA)
    if show_score:
        cv.putText(image, "score: %0.2f" % int_bbox.score,
                (int_bbox.left + left_offset, int_bbox.top + 2 * top_offset),
                cv.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv.LINE_AA)
