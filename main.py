import torch
import cv2
import onnxruntime as ort
import os
import numpy as np
from utils.util import full_frame_preprocess, full_frame_postprocess


weight_path = "../data/models/YoloV7_Tiny.onnx"
provider = ['CPUExecutionProvider']


if __name__ == "__main__":
    
    camera_name = "/dev/video0"
    cap = cv2.VideoCapture(camera_name)
    hand_inf_session = ort.InferenceSession(weight_path, providers=provider)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            image, ratio, dwdh = full_frame_preprocess(frame, auto=False)
            outname = [i.name for i in hand_inf_session.get_outputs()]
            out = hand_inf_session.run(outname, {'images': image/255})[0]
            full_frame_postprocess(frame, out, ratio, dwdh, 0.5)
            
            #TODO: crop the hand and feed to the gesture classifier
            cv2.imshow(camera_name, frame)
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break