import json
import shutil
import numpy as np
from glob import glob
from multiprocessing import Pool, Process
from sklearn.model_selection import train_test_split


import argparse
parser = argparse.ArgumentParser(description='Cutting down the data')
parser.add_argument('--dir', help='Path to the GazeCapture dataset')
parser.add_argument('--out_dir',  help='Path to new dataset should have image, meta folders with train, val, test subfolders in it')
parser.add_argument('--threads', default=1, help='Number of threads', type=int)