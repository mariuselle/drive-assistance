import numpy as np
import cv2
from glob import glob
import pickle


class CameraCalibration:
    def __init__(self, dir='data/calibration/', nx=9, ny=6, createDistFile=False, file='data/structures/calibration.p'):

        self.__nx = nx
        self.__ny = ny
        self.__directory = dir
        self.__object_points = self._get_object_points()

        if createDistFile == True:
            self._create_distorsion_file(file)

        with open(file, mode='rb') as f:
            p = pickle.load(f)

        self.__matrix = p['matrix']
        self.__distorsion = p['distorsion']

    def _get_object_points(self):

        # OBJECT POINTS: (0, 0, 0), (0, 1, 0), ..., (8, 5, 0)
        points = np.zeros((self.__nx * self.__ny, 3), np.float32)
        points[:, :2] = np.mgrid[0:self.__nx, 0:self.__ny].T.reshape(-1, 2)

        return points

    def _create_distorsion_file(self, file):

        # Stores all object points & img points from all images
        object_points = []
        image_points = []
        # Get all images from directory
        images = glob(self.__directory + '*.jpg')
        for _, file_name in enumerate(images):
            image = cv2.imread(file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            has_corners, corners = cv2.findChessboardCorners(
                gray_image, (self.__nx, self.__ny), None)

            if has_corners == True:
                object_points.append(self.__object_points)
                image_points.append(corners)

        size = (image.shape[1], image.shape[0])
        _, matrix, distorsion, _, _ = cv2.calibrateCamera(
            object_points, image_points, size, None, None)

        # Save calibration
        p = {'matrix': matrix, 'distorsion': distorsion}
        pickle.dump(p, open(file, 'wb'))

        print("Calibration file was created successfully!")

    def undistort(self, image, calibration_file='data/structures/calibration.p'):
        # Load calibration data
        undistorted_image = cv2.undistort(
            image, self.__matrix, self.__distorsion, None, self.__matrix)

        return undistorted_image
