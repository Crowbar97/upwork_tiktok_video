import os
import argparse

import numpy as np
import cv2 as cv
from tqdm import tqdm

import moviepy.editor as mpe

import utils
from model import FrozenGraph


def parse_args():
    # Creating parser.
    parser = argparse.ArgumentParser(
        prog="Video cropper",
        description=
        """This utility is used for smart cropping any selfie-video.
           It detects main person's head on each frame of the input video
           and crops area around it by given frame aspect ratio.
           In particular, this app is intended for making TikTok video.
        """
    )

    # Adding arguments.

    # 1. Paths.
    paths_group = parser.add_argument_group(title="Paths params")
    paths_group.add_argument("--input_video_path", type=str, required=True,
                             help="Path to the input video")
    paths_group.add_argument("--header_path", type=str, required=False,
                             help="Path to the header image")
    paths_group.add_argument("--output_video_path", type=str, required=True,
                             help="Path to the output video")
    
    # 2. Output image and frame.
    output_image_group = parser.add_argument_group(
        title="Output image and frame params",
        description="""
            Each image of the output video consists of header image + frame image.
            Header and frame image will be scaled to the dimensions, specified here.
        """
    )
    output_image_group.add_argument("--output_image_height", type=int, required=True,
                                    help="Height of the output video image")
    output_image_group.add_argument("--output_image_width", type=int, required=True,
                                    help="Width of the output video image")
    output_image_group.add_argument(
        "--min_frame_width", type=int, default=500,
        help="""
            Output frames that will be extracted from
            the input video must have at least width
            specified here. This is important for cases
            when main person's head is too far
            (and, hence, small).
        """
    )

    # 3. Main person head detection.
    detection_group = parser.add_argument_group(
        title="Main person head detection params",
        description="""
            On the each image of the input video this tool
            detects all presented heads and then chooses
            head of the main person.
        """
    )
    detection_group.add_argument(
        "--head_score_threshold", type=float, default=0.75,
        help="Minimim detector's score for accepting head detection."
    )
    detection_group.add_argument(
        "--head_dim_threshold", type=int, default=200,
        help="""
            Width and height of any head detection
            must be greater than this value.
        """
    )
    detection_group.add_argument(
        "--debug_mode", default=False, action='store_true',
        help="Rendering ROI (region of interest) bboxes."
    )

    return parser.parse_args()


# NOTE: BGR color space.
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
MAGENTA_COLOR = (255, 0, 255)


def main():

    # 0. System params.

    # Paths.
    TEMP_VIDEO_PATH = os.path.join("assets", "tempxxxxxxxxx.mp4")
    WEIGHTS_PATH = os.path.join("weights", "weights.pb")

    # These params affect the smoothness
    # of the head detection transitions
    # (reducing position and size noises).
    PERSON_LIN_INTERP_COEF = 0.06
    DEFAULT_LIN_INTERP_COEF = 0.02

    # Max number of sequential frames,
    # after which default ROI will be used.
    MAX_NUM_FRAMES_WO_PERSON = 20

    # NOTE: ratios are relative to the
    # main person's head size.
    FRAME_WIDTH_SCALE = 1.1
    FRAME_OFFSET_RATIO = 0.1

    args = parse_args()
    print("Run parameters:")
    for arg in vars(args):
        print("\t %s: %s" % (arg, getattr(args, arg)))

    # 1. Preparing data and model.

    # 1.1 Loading header.
    # NOTE: in case header_path param is not presented,
    # this function will return empty image with zero width.
    header = utils.load_scaled_header(args.header_path, args.output_image_width)

    # 1.2 Calculating output frame aspect ratio.
    output_frame_height = args.output_image_height - header.shape[0]
    output_frame_width = args.output_image_width
    # NOTE: aspect_ratio = width / height.
    output_frame_aspect_ratio = output_frame_width / output_frame_height

    # 1.3 Building input video reader.
    video_reader = utils.build_video_reader(args.input_video_path)
    num_frames, video_frame_width, video_frame_height, fps = utils.get_video_params(video_reader)

    # 1.4 Validating min_frame_width param value.
    if (args.min_frame_width > video_frame_width):
        raise Exception("Min output frame width can not be greater than input video frame width")
    min_frame_height = args.min_frame_width / output_frame_aspect_ratio
    if (min_frame_height > video_frame_height):
        raise Exception("Min output frame width is too large for such aspect ratio and input video height")

    # 1.5 Building output video writer.
    video_writer = None
    if args.debug_mode:
        # Saving whole video frame in case of debug mode
        video_writer = utils.build_video_writer(int(video_frame_width), int(video_frame_height),
                                                int(fps), TEMP_VIDEO_PATH)
    else:
        # Writing only cropped frames.
        video_writer = utils.build_video_writer(args.output_image_width, args.output_image_height,
                                                int(fps), TEMP_VIDEO_PATH)

    # 1.6 Loading head detection model.
    head_detector = FrozenGraph(WEIGHTS_PATH)

    # 2. Making output video.

    print("Processing video..")
    max_frame_width, max_frame_height = utils.calc_max_possible_frame_size(
        video_frame_width, video_frame_height, output_frame_aspect_ratio)
    default_frame_bbox = utils.BBox(
        top=video_frame_height / 2 - max_frame_height / 2,
        left=video_frame_width / 2 - max_frame_width / 2,
        height=max_frame_height,
        width=max_frame_width,
    )
    last_frame_bbox = default_frame_bbox
    num_frames_wo_person = 0
    pbar = tqdm(total=int(num_frames))
    while True:

        # 2.1 Getting sequential video frame.
        has_frame, video_frame = video_reader.read()
        if not has_frame:
            break

        # 2.2 Applying head detector to find all heads in the image.
        _, bboxes = head_detector.run(video_frame)

        # 2.3 Finding main person among all head detections.
        head_bbox = utils.choose_main_person_bbox(bboxes,
                                                  args.head_score_threshold,
                                                  args.head_dim_threshold)

        # 2.4 Building and refining frame bbox.

        frame_bbox = None
        color = None
        lin_interp_coef = PERSON_LIN_INTERP_COEF

        # Case 1: we found main person in the image.
        if head_bbox is not None:
            frame_bbox = utils.build_frame_bbox(
                head_bbox,
                output_frame_aspect_ratio,
                FRAME_WIDTH_SCALE,
                FRAME_OFFSET_RATIO,
                args.min_frame_width,
                video_frame_width,
                video_frame_height,
            )
            num_frames_wo_person = 0
            color = GREEN_COLOR
        # Case 2: we didn't find main person in the image.
        else:
            num_frames_wo_person += 1
            # Case 2.1: we have seen main person recently.
            if num_frames_wo_person < MAX_NUM_FRAMES_WO_PERSON:
                # In this case we just use last frame location.
                frame_bbox = last_frame_bbox
                color = RED_COLOR
            # Case 2.2: we don't see main person for a long time.
            else:
                # So we will use default frame bbox.
                frame_bbox = default_frame_bbox
                lin_interp_coef = DEFAULT_LIN_INTERP_COEF
                color = BLUE_COLOR

        # Refining noisy frame bbox.
        refined_frame_bbox = utils.apply_bbox_lin_interp(last_frame_bbox, frame_bbox,
                                                         lin_interp_coef)
        last_frame_bbox = refined_frame_bbox

        # 2.6 Saving results.
        if args.debug_mode:
            # Rendering head.
            if head_bbox is not None:
                utils.render_bbox(video_frame, head_bbox,
                                  MAGENTA_COLOR, show_score=True)
            # Rendering frame.
            utils.render_bbox(video_frame, refined_frame_bbox, color)

            # Writing output video frame.
            video_writer.write(video_frame)
        else:
            # Building output video frame.
            output_frame = utils.build_output_frame(
                video_frame,
                refined_frame_bbox,
                output_frame_width,
                output_frame_height
            )

            # Adding header.
            output_image = np.vstack([header, output_frame])

            # Writing frame to output video.
            video_writer.write(output_image)
        
        pbar.update(1)

    pbar.close()

    video_reader.release()
    video_writer.release()

    # 3. Copying audio from initial video.
    print("Processing audio..")
    video = mpe.VideoFileClip(TEMP_VIDEO_PATH)
    audio = mpe.VideoFileClip(args.input_video_path).audio
    result_video = video.set_audio(audio)
    result_video.write_videofile(args.output_video_path)

    # 4. Cleaning up.
    os.remove(TEMP_VIDEO_PATH)

    print("Finished!")

if __name__ == "__main__":
    main()