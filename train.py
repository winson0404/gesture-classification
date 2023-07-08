import os
import torch
import cv2
from torch.utils.data import DataLoader
from dataset.dataset import ClassificationHaGridDataset
from omegaconf import OmegaConf
from utils.transform import Compose
from utils.constants import DATASET_OPERATION
from utils.logging import Logger
from tqdm import tqdm
from trainer import ClassificationTrainer

if __name__ == "__main__":
    # print("Hello World")
    #initial config
    conf = OmegaConf.load("configs/default.yaml")
    
    logger = Logger("HaGRID", "MobileNet_Classification", conf)
    trainer = ClassificationTrainer(conf, logger=logger)

    trainer.run_train()
    
    # transform = Compose()
    
    # train_dataset = ClassificationHaGridDataset(conf, op=DATASET_OPERATION.TRAIN)
    # # validation_dataset = ClassificationHaGridDataset(conf, op=DATASET_OPERATION.VALIDATION, transform=transform)
    
    
    # # Load the data from data loader
    # train_loader = data_loader = DataLoader(train_dataset, batch_size=conf.train_params.train_batch_size, shuffle=False)

    # for batch in data_loader:
    #     images, labels = batch
    #     for i, _ in enumerate(images):
    #         image = images[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy array
    #         # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         # breakpoint()
    #         cv2.imwrite(f"test/{i}_{labels[i]}.jpg", images[i].numpy())
            
    #     break
        

    
    # Set up optimizer
    
    
    # Initialize model
    
    
    # Train the model
    
    
    # Export model