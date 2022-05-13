import cv2
import numpy as np
import os
import torch
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.3'

model_path = os.path.join('experiments/pretrained_models', 'GFPGANv1.3.pth')

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
bg_upsampler = RealESRGANer(
    scale=2,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True)

restorer = GFPGANer(
        model_path=model_path,
        upscale=4,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)



cap = cv2.VideoCapture('public/hw02/remastering.mp4')

total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('total_frame_count --> ', int(total_frame))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('restore_video.mp4', fourcc, fps, (644, 480))

while True:
    ret, frame = cap.read()
    print('now_frame_num -> ', cap.get(cv2.CAP_PROP_POS_FRAMES))
    h, w, _ = frame.shape
    if not ret:
        break
    cropped_faces, restored_faces, restored_img = restorer.enhance(frame, paste_back=True)
    restored_img = cv2.resize(restored_img, (644, 480))
    out.write(restored_img)

cap.release()
out.release()
cv2.destroyAllWindows()