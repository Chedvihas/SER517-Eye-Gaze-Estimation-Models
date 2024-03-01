import os
import cv2
import numpy as np
def crop_repeating_edge(image, bbox):
    x, y, w, h = bbox
    cropped = image[y:y+h, x:x+w]
    return cropped

import json
import os

import numpy as np
import math


def crop_repeating_edge(image, rect):
    crop_x, crop_y, crop_w, crop_h = rect
    
    # Ensure crop coordinates are within image bounds
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    crop_w = min(crop_w, image.shape[1] - crop_x)
    crop_h = min(crop_h, image.shape[0] - crop_y)
    
    # Crop the image
    cropped_image = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
    
    return cropped_image
