import os
import shutil
import torch
import cv2
from torch.utils.data import DataLoader
from dataset.dataset import ClassificationHaGridDataset
from omegaconf import OmegaConf
from utils.transform import Compose
from utils.constants import DATASET_OPERATION
from utils.logger import Logger

from tqdm import tqdm
from trainer import ClassificationTrainer
import logging


logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)


if __name__ == "__main__":
    
    #initial config
    project = "MobileNetV3_small"
    selection = None
    # selection = "nine_adam_20.yaml"
    conf_root = os.path.join("configs", project)
    for config in os.listdir(conf_root):
        if config == "default.yaml":
            continue
        
        if selection is not None:
            if selection != config:
                continue
        
        conf = OmegaConf.load(os.path.join(conf_root, config))
        
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
        
        # run test task
        trainer.run_test()
        
        # save model
        trainer.save_model(experiment_path, name="model", mode="onnx")
        
        # end logger job
        logger.finish()