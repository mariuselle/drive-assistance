from lane_detection.distorsion import CameraCalibration
from lane_detection.perspective import BirdEye, SPOINTS, DPOINTS
from lane_detection.filters import LaneFilter
from lane_detection.curves import CurveCalculation
from car_detection.scanner import VehicleScanner
from helpers import show_dotted_image, show_images, roi, draw_boxes, add_small_image

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imsave

calibration = CameraCalibration()
perspective = BirdEye()
lane_filter = LaneFilter()
curves = CurveCalculation()
scanner = VehicleScanner()

def test_calibration(filename, create_file = False):

    image = imread(filename)
    undistorted_image = calibration.undistort(image)

    show_images([image, undistorted_image], per_row=2, per_col=1)


def test_sky_perspective(filename):
    
    image = imread(filename)    
    undistorsted_image = calibration.undistort(image)
    sky_image_view = perspective.sky_view(undistorsted_image)
    
    show_dotted_image(sky_image_view, DPOINTS)

def test_normal_perspective(filename):

    image = imread(filename)
    undistorsted_image = calibration.undistort(image)
    
    show_dotted_image(undistorsted_image, SPOINTS)

def test_lane_filter(filename):

    image = cv2.imread(filename)
    undistorsted_image = calibration.undistort(image)
    binary = lane_filter.apply(undistorsted_image)
    masked_lane = np.logical_and(perspective.sky_view(binary), roi(binary))
    sobel_image = perspective.sky_view(lane_filter.sobel_breakdown(undistorsted_image))
    color_image = perspective.sky_view(lane_filter.color_breakdown(undistorsted_image))

    show_images([color_image, sobel_image, masked_lane], per_col=1, per_row=3, W = 15, H = 5)
    

def test_curve_calculation(filename):

    image = imread(filename)
    undistorsted_image = calibration.undistort(image)
    binary = lane_filter.apply(undistorsted_image)
    wb = np.logical_and(perspective.sky_view(binary), roi(binary).astype(np.uint8))
    result = curves.fit(wb)

    print("[real world] left best-fit curve parameters:", result['left_curve_fit'])
    print("[real world] right best-fit curve parameters:", result['right_curve_fit'])
    print("[pixel] left best-fit curve parameters:", result['left_fit_curve_pix'])
    print("[pixel] right best-fit curve parameters:", result['right_fit_curve_pix'])
    print("[left] current radius of curvature:", result['left_radius'], "m")
    print("[right] current radius of curvature:", result['right_radius'], "m")
    print("vehicle position:", result['position'])

    plt.imshow(result['image'])
    plt.show()


def test_project_lane(filename):
    image = imread(filename)

    undistorsted_image = calibration.undistort(image)
    binary = lane_filter.apply(undistorsted_image)
    wb = np.logical_and(perspective.sky_view(binary), roi(binary).astype(np.uint8))
    result = curves.fit(wb)
    
    image = perspective.project_lane(image, binary, result['left_fit_curve_pix'], result['right_fit_curve_pix'])

    # image = add_small_image(result['image'], image, corner='TOP-RIGHT')
    
    imsave("result.jpg", image)
    plt.imshow(image)
    plt.show()




def test_vehicle_detection(filename):
    image = imread(filename)

    undistorsted_image = calibration.undistort(image)

    v_boxes, heat_map = scanner.relevant_boxes(undistorsted_image)

    scanned_image = draw_boxes(image=image, boxes=v_boxes)

    final_image = add_small_image(heat_map, scanned_image, corner='TOP-RIGHT')

    plt.imshow(final_image)
    plt.show()


test_project_lane('data/test_images/test1.jpg')