import numpy as np
import cv2

def record_v():
    width = 1280
    hieght = 720
    channel = 3
    fps = 30
    sec = 5
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # FourCC is a 4-byte code used to specify the video codec.
    video = cv2.VideoWriter('test.mp4', fourcc, float(fps), (width, hieght))

    for frame_count in range(fps * sec):
        img = np.random.randint(0, 255, (hieght, width, channel), dtype=np.uint8)
        video.write(img)

    video.release()

record_v()