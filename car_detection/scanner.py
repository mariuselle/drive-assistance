import numpy as np
from car_detection.model import VehicleModel
from scipy.ndimage.measurements import label
import cv2

from helpers import color_heat_map

SCANNER_CONST = {'confidence_threshold': .7,
                 'history_depth': 30, 'group_threshold': 10, 'group_diff': .1}
MODEL_PATH = 'data/structures/'

class VehicleScanner:

    def __init__(self, image_shape=(720, 1280, 3), crop=(400, 660), point_size=64):

        self.__crop = crop
        self.__detection_point_size = point_size

        bottom_clip = image_shape[0] - crop[1]
        image_height = image_shape[0] - crop[0] - bottom_clip
        image_width = image_shape[1]
        image_channel = image_shape[2]

        self.__cnn_model, model_name = VehicleModel(model_name='car_model').create_FCNN(
            input_shape=(image_height, image_width, image_channel))
        self.__cnn_model.load_weights( MODEL_PATH + '{}.h5'.format(model_name))

        self.__boxes_history = []
        self.__diag_kernel = [[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]

    def _scan(self, image):

        # Cropping to the region of interest
        roi = image[self.__crop[0]:self.__crop[1], :]
        roi_width, roi_height = roi.shape[1], roi.shape[0]

        # Going 4-D
        roi = np.expand_dims(roi, axis=0)

        # Single-Feature top convolutional layer, which represents a
        # miniaturized (25x153) version of the ROI with the vehicle's probability at each point
        detection_map = self.__cnn_model.predict(roi)

        prediction_map_height, prediction_map_width = detection_map.shape[1], detection_map.shape[2]
        ratio_height, ratio_width = roi_height / prediction_map_height, roi_width / prediction_map_width

        # Prediction output is 4-D tensor: (1, H, W, 1) in this particular case,
        # thus converting to 2-D, effectively represents it as a single-channel image
        detection_map = detection_map.reshape(detection_map.shape[1], detection_map.shape[2])

        # Thresholding by the confidence. The predictions are actually VERY polarized,
        # sticking to mostly Zero for non-vehicle points and mostly Ones for vehicles.
        # That said, midpoint of 0.5 for confidence threshold seems to be a reliable choice.
        detection_map = detection_map > SCANNER_CONST['confidence_threshold']

        labels = label(detection_map, structure=self.__diag_kernel)

        hot_points = []

        # Considering obtained labels as vehicles.
        for vehicle_id in range(labels[1]):
            non_zeros = (labels[0] == vehicle_id + 1).nonzero()
            non_zeros_y = np.array(non_zeros[0])
            non_zeors_x = np.array(non_zeros[1])

            # +/-'s are manually derived adjustments for more appropriate boxes visualization
            x_min = np.min(non_zeors_x) - 32
            x_max = np.max(non_zeors_x) + 32

            y_min = np.min(non_zeros_y)
            y_max = np.max(non_zeros_y) + 64

            span_x = x_max - x_min
            span_y = y_max - y_min

            for x, y in zip(non_zeors_x, non_zeros_y):

                # Adjustment offsets for a box starting point.
                # Ranges from 0 for the left(upper)-most to detectionPointSize for right(bottom)-most
                offset_x = (x - x_min) / span_x * self.__detection_point_size
                offset_y = (y - y_min) / span_y * self.__detection_point_size

                # Getting boundaries in ROI coordinates scale (multiplying by ratioW, ratioH)
                top_left_x = int(round(x * ratio_width - offset_x, 0))
                top_left_y = int(round(y * ratio_height - offset_y, 0))
                bottom_left_x = top_left_x + self.__detection_point_size
                bottom_left_y = top_left_y + self.__detection_point_size

                top_left = (top_left_x, self.__crop[0] + top_left_y)
                bottom_right = (bottom_left_x, self.__crop[0] + bottom_left_y)

                hot_points.append((top_left, bottom_right))
        
        return hot_points
    
    def _add_heat(self, mask, boxes):
        
        for box in boxes:
            # box as ((x, y), (x, y))
            top_y = box[0][1]
            bottom_y = box[1][1]
            left_x = box[0][0]
            right_x = box[1][0]

            mask[top_y:bottom_y, left_x:right_x] +=1 
            
            mask = np.clip(mask, 0, 255) 

        return mask

    def _get_hot_regions(self, image):
        
        hot_points = self._scan(image)
        sample_mask = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat_map = self._add_heat(mask=sample_mask, boxes=hot_points)

        current_frame_boxes = label(heat_map, structure=self.__diag_kernel)

        return current_frame_boxes, heat_map

    def _update_history(self, current_labels):

        for i in range(current_labels[1]):
            non_zeros = (current_labels[0] == i + 1).nonzero()
            non_zeros_y = np.array(non_zeros[0])
            non_zeros_x = np.array(non_zeros[1])

            top_left_x = np.min(non_zeros_x)
            top_left_y = np.min(non_zeros_y)

            bottom_right_x = np.max(non_zeros_x)
            bottom_right_y = np.max(non_zeros_y)

            self.__boxes_history.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])

            self.__boxes_history = self.__boxes_history[-SCANNER_CONST['history_depth']:]

    def relevant_boxes(self, image):

        current_labels, heat_map_gray = self._get_hot_regions(image)

        heat_color = color_heat_map(heat_map_gray, cmap=cv2.COLORMAP_JET)

        self._update_history(current_labels)

        boxes, _ = cv2.groupRectangles(rectList=np.array(self.__boxes_history).tolist(), 
                                        groupThreshold=SCANNER_CONST['group_threshold'], eps=SCANNER_CONST['group_diff'])

        return boxes, heat_color


        
