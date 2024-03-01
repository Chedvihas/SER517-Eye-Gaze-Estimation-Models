import math, shutil, os, time, argparse, json, re, sys
import numpy as np
import scipy.io as sio
from PIL import Image



parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
parser.add_argument('--dataset_path', help="Path to extracted files. It should have folders called '%%05d' in it.")
parser.add_argument('--output_path', default=None, help="Where to write the output. Can be the same as dataset_path if you wish (=default).")
args = parser.parse_args()



def main():
    if args.output_path is None:
        args.output_path = args.dataset_path
    
    if args.dataset_path is None or not os.path.isdir(args.dataset_path):
        raise RuntimeError('No such dataset folder %s!' % args.dataset_path)

    preparePath(args.output_path)

    # list recordings
    recordings = os.listdir(args.dataset_path)
    recordings = np.array(recordings, object)
    recordings = recordings[[os.path.isdir(os.path.join(args.dataset_path, r)) for r in recordings]]
    recordings.sort()

    # Output structure
    meta = {
        'labelRecNum': [],
        'frameIndex': [],
        'labelDotXCam': [],
        'labelDotYCam': [],
        'labelFaceGrid': [],
    }

