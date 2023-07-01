import torch
import torchvision.transforms as transforms

class Compose(object):
    def __init__(self):

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __call__(self, img, label):
        transformed_img = self.transform(img)
        # normalized_img = self.normalize(transformed_img)
        normalized_img = transformed_img

        return normalized_img, label
