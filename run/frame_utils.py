import cv2
import glob
import os
import sys
import json
import shutil

def frames_to_vid(output_folder, frames_folder, video):
    frames = glob.glob(os.path.join(frames_folder, "*.png"))
    frames.sort()
    frames.sort(key=len)

    if len(frames) < 1:
        print("Not enough frames.")
        exit(2)

    img = cv2.imread(frames[0])
    frame_size = (img.shape[1], img.shape[0])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

    path_output = os.path.join(output_folder, video.split('/')[-1].split('.')[0] + '_output.avi')

    if os.path.exists(path_output):
        os.remove(path_output)

    out = cv2.VideoWriter(os.path.join(path_output), fourcc, 30, frame_size)

    for frame in frames:
        img = cv2.imread(frame)
        out.write(img)

    print(f"All images have been merged in one video {path_output}.")
    return

def vid_to_frames(frames_folder, video):
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)

    os.makedirs(frames_folder)

    vidcap = cv2.VideoCapture(video)

    if vidcap is None or not vidcap.isOpened():
        print('Warning: unable to open video source: ', video)
        exit(2)

    count = 0

    success, image = vidcap.read()

    if not success:
        print("Error when reading the video.")
        exit(2)

    print("Processing!")

    while success:
        cv2.imwrite(os.path.join(frames_folder, f"frame{count}.png"), image)
        count+=1
        success, image = vidcap.read()

    print(f"{count} images have been exported in {frames_folder}.")
    return

def frame_rectangle(frame_path, x, y, dst, color=(0,255,0), thickness=2):
    if not os.path.exists(frame_path):
        print("Wrong path for image.")
        exit(2)

    image = cv2.imread(frame_path)

    image = cv2.rectangle(image, x, y, color, thickness)

    if not os.path.exists(dst):
        os.makedirs(dst)
    path_output = os.path.join(dst, frame_path.split('/')[-1])

    cv2.imwrite(path_output, image)
    return

def all_frame_debug_rectangle(frames_folder, rectangle_folder):
    frames = glob.glob(os.path.join(frames_folder, "*.png"))
    frames.sort()
    frames.sort(key=len)

    for frame in frames:
        # TODO exec code cpp to get the json value
        frame_rectangle(frame, (50, 50), (200, 200), rectangle_folder)
    return

def all_frame_rectangle(frames_folder, output_folder, json_path):

    if not os.path.exists(json_path):
        print("Wrong json file.")
        exit(2)

    f = open(json_path)
    data = json.load(f)

    frames = glob.glob(os.path.join(frames_folder, "*.png"))
    frames.sort()
    frames.sort(key=len)

    if len(frames) < 1:
        print("Not enough frames.")
        exit(2)

    img = cv2.imread(frames[1])
    frame_size = (img.shape[1], img.shape[0])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

    path_output = os.path.join(output_folder, 'output.avi')

    if os.path.exists(path_output):
        os.remove(path_output)

    out = cv2.VideoWriter(os.path.join(path_output), fourcc, 30, frame_size)

    for frame in frames:
        frame_img = frame.split('/')[-1]
        image = cv2.imread(frame)
        if frame_img in data:
            for objects in data[frame_img]:
                x = (objects[0], objects[2])
                y = (objects[1], objects[3])
                image = cv2.rectangle(image, x, y, (0, 255, 0), 2)
        out.write(image)

    print(f"All images have been merged in one video {path_output}.")
    return

