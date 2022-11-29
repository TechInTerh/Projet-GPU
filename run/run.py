import cv2
import os
import sys
import shutil

if len(sys.argv) != 3:
    print("Usage: python run.py <path_to_vid> <folder>")
    exit(2)

folder = sys.argv[2]
if os.path.exists(folder):
    shutil.rmtree(folder)

os.makedirs(folder)

vidcap = cv2.VideoCapture(sys.argv[1])

if vidcap is None or not vidcap.isOpened():
    print('Warning: unable to open video source: ', video)
    exit(2)

count = 0

print("Processing!")
success, image = vidcap.read()
while success:
    cv2.imwrite(os.path.join(folder, f"frame{count}.jpg"), image)
    count+=1
    success, image = vidcap.read()

print(f"{count} images have been exported in {folder}.")

exit(0)
