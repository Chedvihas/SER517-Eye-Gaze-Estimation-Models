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

def load_subject(path):
    output = {}
    
    # Apple Face Detections
    with open(os.path.join(path, 'appleFace.json'), 'r') as file:
        input_data = json.load(file)
        output['appleFace'] = {
            'x': [input_data['X'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['X']))],
            'y': [input_data['Y'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['Y']))],
            'w': [input_data['W'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['W']))],
            'h': [input_data['H'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['H']))]
        }

    # Apple Left Eye Detections
    with open(os.path.join(path, 'appleLeftEye.json'), 'r') as file:
        input_data = json.load(file)
        output['appleLeftEye'] = {
            'x': [input_data['X'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['X']))],
            'y': [input_data['Y'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['Y']))],
            'w': [input_data['W'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['W']))],
            'h': [input_data['H'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['H']))]
        }

    # Apple Right Eye Detections
    with open(os.path.join(path, 'appleRightEye.json'), 'r') as file:
        input_data = json.load(file)
        output['appleRightEye'] = {
            'x': [input_data['X'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['X']))],
            'y': [input_data['Y'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['Y']))],
            'w': [input_data['W'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['W']))],
            'h': [input_data['H'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['H']))]
        }

    # Dot Information
    with open(os.path.join(path, 'dotInfo.json'), 'r') as file:
        input_data = json.load(file)
        output['dot'] = {
            'num': input_data['DotNum'],
            'xPts': input_data['XPts'],
            'yPts': input_data['YPts'],
            'xCam': input_data['XCam'],
            'yCam': input_data['YCam'],
            'time': input_data['Time']
        }

    # Face Grid
    with open(os.path.join(path, 'faceGrid.json'), 'r') as file:
        input_data = json.load(file)
        output['faceGrid'] = {
            'x': [input_data['X'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['X']))],
            'y': [input_data['Y'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['Y']))],
            'w': [input_data['W'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['W']))],
            'h': [input_data['H'][i] if input_data['IsValid'][i] else float('nan') for i in range(len(input_data['H']))]
        }

    # Frames
    with open(os.path.join(path, 'frames.json'), 'r') as file:
        output['frames'] = json.load(file)

    # Info
    with open(os.path.join(path, 'info.json'), 'r') as file:
        input_data = json.load(file)
        output['info'] = {
            'totalFrames': input_data['TotalFrames'],
            'numFaceDetections': input_data['NumFaceDetections'],
            'numEyeDetections': input_data['NumEyeDetections'],
            'dataset': input_data['Dataset'],
            'deviceName': input_data['DeviceName']
        }

    # Screen
    with open(os.path.join(path, 'screen.json'), 'r') as file:
        input_data = json.load(file)
        output['screen'] = {
            'w': input_data['W'],
            'h': input_data['H'],
            'orientation': input_data['Orientation']
        }

    return output