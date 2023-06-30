import os
import torch
from dataset import ClassificationHaGridDataset
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from utils.transform import Compose
from utils.constants import DATASET_OPERATION

if __name__ == "__main__":
    print("Hello World")
    #initial config
    conf = OmegaConf.load("config/default.yaml")
    transform = Compose()
    
    train_dataset = ClassificationHaGridDataset(conf, op=DATASET_OPERATION.TRAIN, transform=transform)
    validation_dataset = ClassificationHaGridDataset(conf, op=DATASET_OPERATION.VALIDATION, transform=transform)
    
    
    # Load the data from data loader
    
    
    # Set up optimizer
    
    
    # Initialize model
    
    
    # Train the model
    
    
    # Export model