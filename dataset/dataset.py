import torch
from typing import List, Tuple, Optional
import json
from omegaconf import OmegaConf
import random
import numpy as np
from utils.constants import DATASET_OPERATION
from utils.transform import Compose
import cv2
import os

def get_cropped(img, bbox, label):
    x1, y1, w, h = bbox

class ClassificationHaGridDataset(torch.utils.data.Dataset):
    def __init__(self, conf:OmegaConf, op:DATASET_OPERATION, transform:Optional[Compose]=None):
        
        # # set seed
        
        self.conf = conf
        random.seed(self.conf.random_seed)
        self.transform = transform
        self.op:DATASET_OPERATION = op
        #load data from data_path
        self.data = self._get_images()
        self.labels = {
            label: num for (label, num) in zip(self.conf.dataset.targets, range(len(self.conf.dataset.targets)))
        }

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        bboxes, labels, ids = self.data
        box_scale = 1.0
        if self.op == DATASET_OPERATION.TRAIN:
            box_scale = np.random.uniform(low=1.0, high=2.0)
        
        img = self._crop_image(idx, bboxes, ids, box_scale)
        label = labels[idx]
        label = self.labels[label]
        if self.transform:
            img, label = self.transform(img, label)
        
        
        return img, label
        
    
    def _crop_image(self, idx, bbox, img_id, box_scale):
        data_path = os.path.join(self.conf.dataset.dataset_path, DATASET_OPERATION.TRAIN.value if self.op.value != DATASET_OPERATION.TEST.value else DATASET_OPERATION.TEST.value)
        
        # breakpoint()
        img = cv2.imread(os.path.join(data_path, img_id[idx]+".jpg"))
        
        width, height = img.shape[1], img.shape[0]
        
        # get the actual bbox
        x1, y1, w, h = bbox[idx]
        bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]
        
        # convert to int
        int_bbox = np.array(bbox_abs).round().astype(np.int32)

        # prepare to crop, ie: scale the cropping area
        x1 = int_bbox[0]
        y1 = int_bbox[1]
        x2 = int_bbox[2]
        y2 = int_bbox[3]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = h = max(x2 - x1, y2 - y1)
        x1 = max(0, cx - box_scale * w // 2)
        y1 = max(0, cy - box_scale * h // 2)
        x2 = cx + box_scale * w // 2
        y2 = cy + box_scale * h // 2
        x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))

        crop_image = img[y1:y2, x1:x2]
        
        # scale back to input size and pad
        
        # get dimension
        cropped_height, cropped_width, _ = crop_image.shape
        target_width, target_height = self.conf.dataset.image_size
        
        # choose scaling factor based on the longest height/width
        side = max(cropped_height, cropped_width)
        scale = (target_width if target_width >= target_height else target_height) / side
        
        # calculate new dimension and resize
        resized_width, resized_height = int(cropped_width * scale), int(cropped_height * scale)
        resized_image = cv2.resize(crop_image, (resized_width, resized_height))
        
        # pad the cropped gesture image to the input size
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_image[:] = (0, 0, 0)
        
        # Calculate the padding required
        pad_top = (target_height - resized_height) // 2
        pad_bottom = target_height - resized_height - pad_top
        pad_left = (target_width - resized_width) // 2
        pad_right = target_width - resized_width - pad_left

        try:
            image_resized = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        except:
            print("Error!")
        return image_resized

    def _get_images(self)->List[str]:
        bboxes, labels, ids = self._get_annotation()
        
        # zip and shuffle
        data = list(zip(bboxes, labels, ids))
        random.shuffle(data)
        
        # unzip
        bboxes, labels, ids = zip(*data)
        
        # perform train, test, validation split
        if not self.op == DATASET_OPERATION.TEST:
            if self.op == DATASET_OPERATION.TRAIN:
                bboxes = bboxes[:int(len(bboxes)*0.8)]
                labels = labels[:int(len(labels)*0.8)]
                ids = ids[:int(len(ids)*0.8)]
                
            if self.op == DATASET_OPERATION.VALIDATION:
                bboxes = bboxes[int(len(bboxes)*0.8):]
                labels = labels[int(len(labels)*0.8):]
                ids =  ids[int(len(ids)*0.8):]
    
        return bboxes, labels, ids
    
    def _get_annotation(self)->Tuple:
        """
        Load annotation from json file and seperate into dictionary with names, bbox, label etc

        If an image have both no-gesture and gesture, it will pick one based on a 0.3 no-gesture - 0.7 gesture probability 

        Returns:
            Tuple: (bbox, label, name)
        """
        bboxes = []
        labels = []
        ids = []
        
        category_path = os.path.join(self.conf.dataset.annotation_path, "train" if self.op.value != DATASET_OPERATION.TEST.value else DATASET_OPERATION.TEST)
        for target in self.conf.dataset.targets:
            if target == "no_gesture":
                continue
            ann = json.load(open(os.path.join(category_path, f"{target}.json"), "r"))
            for name, dat in ann.items():
                ids.append(f"{target}/{name}")
                # breakpoint()
                if len(labels) > 1:
                    pick = random.choices(["no_gesture", "gesture"], weights=[0.3, 0.7])[0]
                    # for index, label in enumerate(ann["labels"]):
                    #     if label == pick:
                    #         labels.append(index)
                    #         bboxes.append(ann["bbox"][index])
                    if pick == "no_gesture":
                        if dat["labels"][0] == "no_gesture":
                            labels.append(dat["labels"][0])
                            bboxes.append(dat["bboxes"][0])
                        else:
                            labels.append(dat["labels"][-1])
                            bboxes.append(dat["bboxes"][-1])
                    else:
                        if dat["labels"][0] == "no_gesture":
                            labels.append(dat["labels"][-1])
                            bboxes.append(dat["bboxes"][-1])
                        else:
                            labels.append(dat["labels"][0])
                            bboxes.append(dat["bboxes"][0])
                else:
                    labels.append(dat["labels"][0])
                    bboxes.append(dat["bboxes"][0])
        
        # breakpoint()
        return (bboxes, labels, ids)