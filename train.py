import os
import shutil
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
import logging


logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)


if __name__ == "__main__":
    
    #initial config
    conf = OmegaConf.load("configs/default.yaml")
    
    # create output directory
    project_path = os.path.join("output", conf.project_name)
    os.makedirs(project_path, exist_ok=True)
    
    # create experiment directory
    experiment_path = os.path.join(project_path, conf.experiment_name)
    if os.path.exists(experiment_path):
        shutil.rmtree(experiment_path, ignore_errors=False, onerror=None)
    os.makedirs(experiment_path, exist_ok=False)
    
    # initialize wandb logger
    logger = Logger(conf.project_name, conf.experiment_name, conf)
    # logger = None
    
    # initialize trainer
    trainer = ClassificationTrainer(conf, output_path=experiment_path, logger=logger)

    # run training task (include validation)
    trainer.run_train()
    
    # save model
    trainer.save_model(experiment_path, output_name="model")
    
    # end logger job
    logger.finish()