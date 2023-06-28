import torch
from typing import List


class ClassificationHaGridDataset(torch.utils.data.Dataset):
    def __init__(self, is_train:bool, data_path:str, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        #load data from data_path
        self.images = self._get_images(data_path)
        
        

    def __len__(self):
        return len(self.data['image'])

    def __getitem__(self, idx):
        image = self.data['image'][idx]
        label = self.data['label'][idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_images(self, path:str)->List[str]:
        images = []
    
    
        return images