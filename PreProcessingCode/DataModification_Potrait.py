import argparse
parser = argparse.ArgumentParser(description='Cutting down the data')
parser.add_argument('--dir', help='Path to the GazeCapture dataset')
parser.add_argument('--out_dir',  help='Path to new dataset should have image, meta folders with train, val, test subfolders in it')
parser.add_argument('--threads', default=1, help='Number of threads', type=int)

def modify_dataset(files, out_root_dir):
    for i in files: 
        with open(i+"/info.json") as f:
            data = json.load(f)
            ds = data['Dataset']
            device = data['DeviceName']
        out_dir = out_root_dir+ds
        expt_name = i.split('/')[-2]
        screen_info = json.load(open(i+'/screen.json'))
        face_det = json.load(open(i+'/appleFace.json'))
        l_eye_det = json.load(open(i+'/appleLeftEye.json'))
        r_eye_det = json.load(open(i+'/appleRightEye.json'))
        dot = json.load(open(i+'/dotInfo.json'))