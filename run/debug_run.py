import cv2
import glob
import os
import sys
import json
import shutil
from frame_utils import *

if len(sys.argv) != 5:
    print("Usage: python run.py <path_to_vid> <frame_folder> <output_folder> <rectangle_folder>")
    exit(2)


video_path = sys.argv[1]
frame_folder = sys.argv[2]
ouput_folder = sys.argv[3]
rectangle_folder = sys.argv[4]

vid_to_frames(frame_folder, video_path)

#all_frame_debug_rectangle(frame_folder, rectangle_folder)

#frames_to_vid(ouput_folder, frame_folder, video_path)
#frames_to_vid(ouput_folder, rectangle_folder, video_path)
