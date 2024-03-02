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


base_directory = '/Volumes/Extreme SSD/517-small-data-model/data'
if not os.path.exists(base_directory):
    raise ValueError("The specified base directory does not exist. Please edit the script to specify the root of the numbered subject directories.")

subject_dirs = os.listdir(base_directory)
print(subject_dirs, type(subject_dirs))
for curr_subject in subject_dirs:
    full_path = os.path.join(base_directory, curr_subject)
    print(curr_subject)
    print(os.path.isdir(curr_subject))
    # Valid subject directories have five-digit numbers.
    if not (os.path.isdir(full_path) and len(curr_subject) == 5 and curr_subject.isdigit()):
        continue
    print(f"Processing subject {curr_subject}...")
    subject_dir = os.path.join(base_directory, curr_subject)
    s = load_subject(subject_dir)
    apple_face_dir = os.path.join(subject_dir, 'appleFace')
    apple_left_eye_dir = os.path.join(subject_dir, 'appleLeftEye')
    apple_right_eye_dir = os.path.join(subject_dir, 'appleRightEye')
    os.makedirs(apple_face_dir, exist_ok=True)
    os.makedirs(apple_left_eye_dir, exist_ok=True)
    os.makedirs(apple_right_eye_dir, exist_ok=True)
    
    for i in range(len(s['frames'])):
        frame_filename = s['frames'][i]
        frame = cv2.imread(os.path.join(subject_dir, 'frames', frame_filename))
        # iTracker requires we have face and eye detections; we don't save
        # any if we don't have all three.


        # Assuming i is defined
        if math.isnan(s['appleFace']['x'][i]) or math.isnan(s['appleLeftEye']['x'][i]) or math.isnan(s['appleRightEye']['x'][i]):
            continue

       
            # Concatenate the bounding box coordinates for the face region
        face_bbox = [round(s['appleFace']['x'][i]), round(s['appleFace']['y'][i]), round(s['appleFace']['w'][i]), round(s['appleFace']['h'][i])]




        print(face_bbox)
        left_eye_bbox_values = [s['appleLeftEye']['x'][i], s['appleLeftEye']['y'][i], s['appleLeftEye']['w'][i], s['appleLeftEye']['h'][i]]

# Filter out NaN values and round the coordinates for the left eye
        left_eye_bbox_rounded = [round(val) for val in left_eye_bbox_values if not np.isnan(val)]

        # Round the bounding box coordinates for the right eye
        right_eye_bbox_values = [s['appleRightEye']['x'][i], s['appleRightEye']['y'][i], s['appleRightEye']['w'][i], s['appleRightEye']['h'][i]]

# Filter out NaN values and round the coordinates for the right eye
        right_eye_bbox_rounded = [round(val) for val in right_eye_bbox_values if not np.isnan(val)]


        # Crop left eye image
        face_image = crop_repeating_edge(frame, face_bbox)
        left_eye_image = crop_repeating_edge(face_image, left_eye_bbox_rounded)
        right_eye_image = crop_repeating_edge(face_image, right_eye_bbox_rounded)
    
    
        

        
        cv2.imwrite(os.path.join(apple_face_dir, frame_filename), face_image)
        cv2.imwrite(os.path.join(apple_left_eye_dir, frame_filename), left_eye_image)
        cv2.imwrite(os.path.join(apple_right_eye_dir, frame_filename), right_eye_image)
