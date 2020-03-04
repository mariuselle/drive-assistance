import numpy as np
import cv2


"""
SOURCE POINTS:
     0--3
    /    \
   1______2
NOTE: source_points and destination_points are hardcoded and need to be changed if video change !
"""
SPOINTS = [(580, 460), (205, 720), (1110, 720), (703, 460)]
DPOINTS = [(320, 0), (320, 720), (960, 720), (960, 0)]


class BirdEye:
    def __init__(self):

        self.__source_points = np.array(SPOINTS, dtype=np.float32)
        self.__destination_points = np.array(DPOINTS, dtype=np.float32)

        self.__warp_matrix = cv2.getPerspectiveTransform(
            self.__source_points, self.__destination_points)
        self.__inv_warp_matrix = cv2.getPerspectiveTransform(
            self.__destination_points, self.__source_points)

    def sky_view(self, undistorsted_image):

        size = (undistorsted_image.shape[1], undistorsted_image.shape[0])
        warp_image = cv2.warpPerspective(
            undistorsted_image, self.__warp_matrix, size, flags=cv2.INTER_LINEAR)

        return warp_image

    def project_lane(self, image, sky_lane, left_fit, right_fit, color=(0, 255, 0)):
        z = np.zeros_like(sky_lane)
        sky_lane = np.dstack((z, z, z))

        k_left, k_right = left_fit, right_fit
        h = sky_lane.shape[0]
        ys = np.linspace(0, h - 1, h)
        left_xs = k_left[0] * (ys**2) + k_left[1] * ys + k_left[2]
        right_xs = k_right[0] * (ys**2) + k_right[1] * ys + k_right[2]

        points_left = np.array([np.transpose(np.vstack([left_xs, ys]))])
        points_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_xs, ys])))])

        final_points = np.hstack((points_left, points_right))

        cv2.fillPoly(sky_lane, np.int_(final_points), color)

        size = (sky_lane.shape[1], sky_lane.shape[0])

        # Change sky perspective to ground perspective
        ground_lane = cv2.warpPerspective(
            sky_lane, self.__inv_warp_matrix, size)

        # Overlays the original image and the line detect image
        final_image = cv2.addWeighted(image, 1, ground_lane, 0.3, 0)

        return final_image
