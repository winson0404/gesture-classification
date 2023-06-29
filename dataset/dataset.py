import torch
from typing import List, Tuple
import json
from omegaconf import OmegaConf
import random
import numpy as np
from constants import DATASET_OPERATION

class ClassificationHaGridDataset(torch.utils.data.Dataset):
    def __init__(self, conf:OmegaConf, op:DATASET_OPERATION, data_path:str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.conf = conf
        self.op:DATASET_OPERATION = op
        #load data from data_path
        self.images = self._get_images(data_path)
        
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.data['image'][idx]
        label = self.data['label'][idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def _crop_image(self, image, bbox):
        pass

    def _get_images(self, path:str)->List[str]:
        images = []
    
    
        return images
    
    def _get_annotation(self)->Tuple:
        """
        Load annotation from json file and seperate into dictionary with names, bbox, label etc

        If an image have both no-gesture and gesture, it will pick one based on a 0.3 no-gesture - 0.7 gesture probability 

        Returns:
            Tuple: (bbox, label, id)
        """
        bboxs = []
        labels = []
        ids = []
        
        ann = json.load(open(self.conf.dataset.annotation_path))
        for id, dat in ann.items():
            ids.append(id)
            if len(labels) > 1:
                pick = np.random.choice(["no_gesture","gesture"], p=[0.3, 0.7])
                # for index, label in enumerate(ann["labels"]):
                #     if label == pick:
                #         labels.append(index)
                #         bboxs.append(ann["bbox"][index])
                if pick == "no_gesture":
                    if ann["labels"][0] == "no_gesture":
                        labels.append(ann["labels"][0])
                        bboxs.append(ann["bbox"][0])
                    else:
                        labels.append(ann["labels"][-1])
                else:
                    if ann["labels"][0] == "no_gesture":
                        labels.append(ann["labels"][-1])
                        bboxs.append(ann["bbox"][-1])
                    else:
                        labels.append(ann["labels"][0])
                        bboxs.append(ann["bbox"][0])
            else:
                labels.append(ann["labels"][0])
                bboxs.append(ann["bbox"][0])
                        
        return (bboxs, labels, ids)