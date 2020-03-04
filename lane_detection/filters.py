import cv2
import numpy as np
from helpers import scale_abs


# NOTE: This parameters are harcoded and can be imporved
THRESHOLD_PARAMS = { 'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
      'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20 }

class LaneFilter:
    def __init__(self):

        self.__sat_thresh = THRESHOLD_PARAMS['sat_thresh']
        self.__light_thresh = THRESHOLD_PARAMS['light_thresh']
        self.__light_thresh_agr = THRESHOLD_PARAMS['light_thresh_agr']
        self.__grad_min, self.__grad_max = THRESHOLD_PARAMS['grad_thresh']
        self.__mag_thresh = THRESHOLD_PARAMS['mag_thresh']
        self.__x_thresh = THRESHOLD_PARAMS['x_thresh']

        self.__zeros = None
        self.__color_cond_1, self.__color_cond_2 = None, None
        self.__sobel_cond_1, self.__sobel_cond_2, self.__sobel_cond_3 = None, None, None

    def apply(self, color_image):
        
        hls_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HLS)
        lightness = hls_image[:, :, 1]
        saturation = hls_image[:, :, 2]
    
        self.__zeros = np.zeros_like(saturation)

        color_mask_image = self._apply_color_mask(lightness, saturation)
        sobel_mask_image = self._apply_sobel_mask(lightness)
        
        filtered_image = cv2.bitwise_or(sobel_mask_image, color_mask_image)

        return filtered_image
    
    def color_breakdown(self, img):
        self.apply(img)
        
        img1, img2 = self.__zeros.copy(), self.__zeros.copy()

        img1[(self.__color_cond_1)] = 255
        img2[(self.__color_cond_2)] = 255

        return np.dstack((img1, img2, self.__zeros))

    def sobel_breakdown(self, img):
        self.apply(img)

        img1, img2, img3 = self.__zeros.copy(), self.__zeros.copy(), self.__zeros.copy()

        img1[(self.__sobel_cond_1)] = 255
        img2[(self.__sobel_cond_2)] = 255
        img3[(self.__sobel_cond_3)] = 255

        return np.dstack((img1, img2, img3))

    def _apply_color_mask(self, lightness, saturation):

        self.__color_cond_1 = (saturation > self.__sat_thresh) & (lightness > self.__light_thresh)
        self.__color_cond_2 = lightness > self.__light_thresh_agr

        color_mask_image = self.__zeros.copy()
        color_mask_image[(self.__color_cond_1 | self.__color_cond_2)] = 1

        return color_mask_image

    def _apply_sobel_mask(self, lightness):

        lx = cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize = 5)
        ly = cv2.Sobel(lightness, cv2.CV_64F, 0, 1, ksize = 5)
        
        l_gradient = np.arctan2(np.absolute(ly), np.absolute(lx))
        l_magnitude = np.sqrt(lx**2 + ly**2)

        slm, slx = scale_abs(l_magnitude), scale_abs(lx)

        self.__sobel_cond_1 = slm > self.__mag_thresh
        self.__sobel_cond_2 = slx > self.__x_thresh
        self.__sobel_cond_3 = (l_gradient > self.__grad_min) & (l_gradient < self.__grad_max)

        sobel_mask_image = self.__zeros.copy()
        sobel_mask_image[(self.__sobel_cond_1 & self.__sobel_cond_2 & self.__sobel_cond_3)] = 1

        return sobel_mask_image





