import cv2
import glob
import os
import sys
import json
import shutil
from frame_utils import *

if len(sys.argv) != 4:
    print("Usage: python run.py <frame_folder> <output_folder> <json>")
    exit(2)

frame_folder = sys.argv[1]
output_folder = sys.argv[2]
json = sys.argv[3]

all_frame_rectangle(frame_folder, output_folder, json)
