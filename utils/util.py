import torch
from torch import nn, Tensor
import numpy as np
import cv2
import random
from models.classification import MobileNetV3
from constants import MOBILENETV3_SIZE
import os

def full_frame_preprocess(im, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, 0)
    im = np.ascontiguousarray(im)
    
    im = im.astype(np.float32)
    
    return im, r, (dw, dh)



def full_frame_postprocess(image, model_output, ratio, dwdh, threshold):
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(model_output):
        if score < threshold:
            continue
        if batch_id >= 6:
            break
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        color = (0,255,0)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,f"{score}",(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)


def set_random_state(seed: int)-> None:
    """
    Set random seed for torch, numpy, random

    Parameters
    ----------
    random_seed: int
        Random seed from config
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def build_model(
    model_name: str,
    num_classes: int,
    device: str,
    checkpoint: str = None,
    pretrained: bool = False,
    freezed: bool = False
) -> nn.Module:
    """
    Build modela and load checkpoint

    Parameters
    ----------
    model_name : str
        Model name e.g. ResNet18, MobileNetV3_small, Vitb32
    num_classes : int
        Num classes for each task
    checkpoint : str
        Path to model checkpoint
    device : str
        Cpu or CUDA device
    pretrained : false
        Use pretrained model
    freezed : false
        Freeze model layers
    """
    models = {
        "MobileNetV3_large": MobileNetV3(
            num_classes=num_classes, size=MOBILENETV3_SIZE.LARGE, pretrained=pretrained, freezed=freezed
        ),
        "MobileNetV3_small": MobileNetV3(
            num_classes=num_classes, size=MOBILENETV3_SIZE.SMALL, pretrained=pretrained, freezed=freezed
        )
    }

    model = models[model_name]

    if checkpoint is not None:
        # checkpoint = os.path.expanduser(checkpoint)
        if os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint, map_location=torch.device(device))["state_dict"]
            model.load_state_dict(checkpoint)

    model.to(device)
    return model