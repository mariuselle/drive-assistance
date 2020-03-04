import argparse
import cv2
import numpy as np
from scipy.misc import imresize
from moviepy.editor import VideoFileClip

from lane_detection.distorsion import CameraCalibration
from lane_detection.perspective import BirdEye
from lane_detection.filters import LaneFilter
from lane_detection.curves import CurveCalculation
from car_detection.scanner import VehicleScanner
from helpers import roi, draw_boxes, add_small_image


import matplotlib.pyplot as plt


INPUT_DIR = 'data/input/'
OUTPUT_DIR = 'data/output/'

calibration = CameraCalibration()
perspective = BirdEye()
lane_filter = LaneFilter()
curves = CurveCalculation()
scanner = VehicleScanner()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", required=True,
                        help="name of input video from data/input/")

    return parser.parse_args()


def pipeline(image):

    # Undistord image
    undistorsted_image = calibration.undistort(image)
    
    # Add scanned vehicles
    v_boxes, heat_map = scanner.relevant_boxes(undistorsted_image)

    # Extract lines
    binary = lane_filter.apply(undistorsted_image)
    # Show in sky view
    wb = np.logical_and(perspective.sky_view(binary),
                        roi(binary).astype(np.uint8))
    # Calculate curves fit
    result = curves.fit(wb)

    # Project curvature on image
    image = perspective.project_lane(
        image, binary, result['left_fit_curve_pix'], result['right_fit_curve_pix'])


    if len(v_boxes) > 0:
        image = draw_boxes(image=image, boxes=v_boxes)

    # Adding heat map and curves calculation pips
    image = add_small_image(heat_map, image, corner='TOP-LEFT')
    image = add_small_image(result['image'], image, corner='TOP-RIGHT')

    return image

def _create_output_name(name):
    return name.split('.')[0] + '_output.' + name.split('.')[1]


if __name__ == "__main__":

    args = _parse_args()

    input_video = INPUT_DIR + args.video
    output_video = OUTPUT_DIR + _create_output_name(args.video)

    clip = VideoFileClip(input_video)
    output_clip = clip.fl_image(pipeline)
    output_clip.write_videofile(output_video, audio=False)
