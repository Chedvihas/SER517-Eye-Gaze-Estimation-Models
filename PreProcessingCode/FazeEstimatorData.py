import os

import cv2 as cv
import h5py
import numpy as np

face_model_3d_coordinates = None

normalized_camera = {
    'focal_length': 1300,
    'distance': 600,
    'size': (256, 64),
}

norm_camera_matrix = np.array(
    [
        [normalized_camera['focal_length'], 0, 0.5*normalized_camera['size'][0]],  # noqa
        [0, normalized_camera['focal_length'], 0.5*normalized_camera['size'][1]],  # noqa
        [0, 0, 1],
    ],
    dtype=np.float64,
)


class Undistorter:

    _map = None
    _previous_parameters = None

    def __call__(self, image, camera_matrix, distortion, is_gazecapture=False):
        h, w, _ = image.shape
        all_parameters = np.concatenate([camera_matrix.flatten(),
                                         distortion.flatten(),
                                         [h, w]])
        if (self._previous_parameters is None
                or len(self._previous_parameters) != len(all_parameters)
                or not np.allclose(all_parameters, self._previous_parameters)):
            print('Distortion map parameters updated.')
            self._map = cv.initUndistortRectifyMap(
                camera_matrix, distortion, R=None,
                newCameraMatrix=camera_matrix if is_gazecapture else None,
                size=(w, h), m1type=cv.CV_32FC1)
            print('fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f' % (
                    camera_matrix[0, 0], camera_matrix[1, 1],
                    camera_matrix[0, 2], camera_matrix[1, 2]))
            self._previous_parameters = np.copy(all_parameters)

        # Apply
        return cv.remap(image, self._map[0], self._map[1], cv.INTER_LINEAR)


undistort = Undistorter()

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2,
              color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out