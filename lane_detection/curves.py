import cv2
import numpy as np


# NOTE: This parameters are harcoded and can be imporved, also if video change, they might be change!
CURVES_PARAMS = {'number_of_windows': 9, 'margin': 100, 'minimum_pixels': 50,
                 'ym_per_pixel': 30 / 720, 'xm_per_pixel': 3.7 / 700}


class CurveCalculation:
    def __init__(self):

        self.__n = CURVES_PARAMS['number_of_windows']
        self.__margin = CURVES_PARAMS['margin']
        self.__minimum_pixels = CURVES_PARAMS['minimum_pixels']
        self.__ky = CURVES_PARAMS['ym_per_pixel']
        self.__kx = CURVES_PARAMS['xm_per_pixel']

        self.__height, self.__width, self.__window_height = None, None, None
        self.__all_pixels_x, self.__all_pixels_y = None, None
        self.__output_image = None
        self.__left_fit_curve_pix, self.__right_fit_curve_pix = None, None
        self.__vehicle_position, self.__positions = None, None

    def _store_details(self, binary):

        self.__output_image = np.dstack((binary, binary, binary)) * 255
        self.__height, self.__width = binary.shape[0], binary.shape[1]
        self.__window_height = np.int(self.__height / self.__n)
        self.__all_pixels_x = np.array(binary.nonzero()[1])
        self.__all_pixels_y = np.array(binary.nonzero()[0])

    def _next_y(self, w):

        y_low = self.__height - (w + 1) * self.__window_height
        y_high = self.__height - w * self.__window_height

        return y_low, y_high

    def _next_x(self, current):

        x_left = current - self.__margin
        x_right = current + self.__margin

        return x_left, x_right

    def _next_mid_x(self, current, pixel_indicies):

        if len(pixel_indicies) > self.__minimum_pixels:
            current = np.int(np.mean(self.__all_pixels_x[pixel_indicies]))

        return current

    def _start(self, binary):

        hist = np.sum(binary[np.int(self.__height / 2):, :], axis=0)
        mid = np.int(hist.shape[0] / 2)

        curr_left_x = np.argmax(hist[:mid])
        curr_right_x = np.argmax(hist[mid:]) + mid

        return curr_left_x, curr_right_x

    def _indicies_within_boundary(self, y_low, y_high, x_left, x_right):

        cond1 = (self.__all_pixels_y >= y_low)
        cond2 = (self.__all_pixels_y < y_high)
        cond3 = (self.__all_pixels_x >= x_left)
        cond4 = (self.__all_pixels_x < x_right)

        return (cond1 & cond2 & cond3 & cond4).nonzero()[0]

    def _pixel_location(self, indicies):
        return self.__all_pixels_x[indicies], self.__all_pixels_y[indicies]

    def _get_real_curvature(self, xs, ys):
        return np.polyfit(ys * self.__ky, xs * self.__kx, 2)

    def _get_radius_of_curve(self, y, f):
        return ((1 + (2 * f[0] * y + f[1]) ** 2) ** (1.5)) / np.absolute(2 * f[0])

    def _update_vehicle_position(self):

        y = self.__height
        mid = self.__width / 2
        k_left, k_right = self.__left_fit_curve_pix, self.__right_fit_curve_pix

        x_left = k_left[0] * (y**2) + k_left[1] * y + k_left[2]
        x_right = k_right[0] * (y**2) + k_right[1] * y + k_right[2]
        pixel_position = x_left + (x_right - x_left) / 2

        self.__vehicle_position = (pixel_position - mid) * self.__kx

        if self.__vehicle_position < 0:
            self.__position = str(np.absolute(
                np.round(self.__vehicle_position, 2))) + "m left of center."
        elif self.__vehicle_position > 0:
            self.__position = str(np.absolute(
                np.round(self.__vehicle_position, 2))) + "m right of center."
        else:
            self.__position = "In center."

    def _plot(self, left_pixels_y, left_pixels_x, right_pixels_y, right_pixels_x, t=4):

        self.__output_image[left_pixels_y, left_pixels_x] = [255, 0, 255]
        self.__output_image[right_pixels_y, right_pixels_x] = [0, 255, 255]

        self.__left_fit_curve_pix = np.polyfit(left_pixels_y, left_pixels_x, 2)
        self.__right_fit_curve_pix = np.polyfit(
            right_pixels_y, right_pixels_x, 2)

        k_left, k_right = self.__left_fit_curve_pix, self.__right_fit_curve_pix
        ys = np.linspace(0, self.__height - 1, self.__height)

        left_xs = k_left[0] * (ys ** 2) + k_left[1] * ys + k_left[2]
        right_xs = k_right[0] * (ys**2) + k_right[1] * ys + k_right[2]

        xls, xrs, ys = left_xs.astype(np.uint32), right_xs.astype(
            np.uint32), ys.astype(np.uint32)

        for left_x, right_x, y in zip(xls, xrs, ys):
            cv2.line(self.__output_image, (left_x - t, y),
                     (left_x + t, y), (255, 255, 0), int(t / 2))
            cv2.line(self.__output_image, (right_x - t, y),
                     (right_x + t, y), (0, 0, 255), int(t / 2))

    def _draw_boundaries(self, p1, p2, color, thickness=5):
        cv2.rectangle(self.__output_image, p1, p2, color, thickness)

    def fit(self, binary):

        self._store_details(binary)
        mid_left_x, mid_right_x = self._start(binary)

        left_pixels_indices, right_pixels_indices = [], []
        x, y = [None, None, None, None], [None, None]

        for window in range(self.__n):

            # Get points for square
            y[0], y[1] = self._next_y(window)
            x[0], x[1] = self._next_x(mid_left_x)
            x[2], x[3] = self._next_x(mid_right_x)

            # Draw square
            self._draw_boundaries((x[0], y[0]), (x[1], y[1]), (255, 0, 0))
            self._draw_boundaries((x[2], y[0]), (x[3], y[1]), (0, 255, 0))

            # Get indicies from squares
            curr_left_pixels_indicies = self._indicies_within_boundary(
                y[0], y[1], x[0], x[1])
            curr_right_pixels_indicies = self._indicies_within_boundary(
                y[0], y[1], x[2], x[3])

            left_pixels_indices.append(curr_left_pixels_indicies)
            right_pixels_indices.append(curr_right_pixels_indicies)

            # Get next middle x
            mid_left_x = self._next_mid_x(
                mid_left_x, curr_left_pixels_indicies)
            mid_right_x = self._next_mid_x(
                mid_right_x, curr_right_pixels_indicies)

        # Concat all array pixels indicies
        left_pixels = np.concatenate(left_pixels_indices)
        right_pixels = np.concatenate(right_pixels_indices)

        # ???? Need help here to understand
        left_pixels_x, left_pixels_y = self._pixel_location(left_pixels)
        right_pixels_x, right_pixels_y = self._pixel_location(right_pixels)

        # Get fit of curvature
        left_curve_fit = self._get_real_curvature(left_pixels_x, left_pixels_y)
        right_curve_fit = self._get_real_curvature(
            right_pixels_x, right_pixels_y)

        # Get radius of curvature
        left_radius = self._get_radius_of_curve(
            self.__height * self.__ky, left_curve_fit)
        right_radius = self._get_radius_of_curve(
            self.__height * self.__ky, right_curve_fit)

        self._plot(left_pixels_y=left_pixels_y, left_pixels_x=left_pixels_x,
                   right_pixels_y=right_pixels_y, right_pixels_x=right_pixels_x)

        self._update_vehicle_position()

        return {
            'image': self.__output_image,
            'left_radius': left_radius,
            'right_radius': right_radius,
            'left_curve_fit': left_curve_fit,
            'right_curve_fit': right_curve_fit,
            'left_fit_curve_pix': self.__left_fit_curve_pix,
            'right_fit_curve_pix': self.__right_fit_curve_pix,
            'vehicle_position': self.__vehicle_position,
            'position': self.__position
        }
