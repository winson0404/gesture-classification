import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from typing import Tuple
from dataset import ClassificationHaGridDataset
from omegaconf import OmegaConf
from utils.constants import DATASET_OPERATION
from utils.transform import Compose
import logging

def save_onnx(model: nn.Module, output_dir: str, name: str, input_shape: Tuple[int, int, int] = (3, 224, 224)) -> None:
    """
    Save model to onnx format

    Parameters
    ----------
    model : nn.Module
        Model for checkpoint save
    output_dir : str
        Path to directory model checkpoint
    name : str
        Model name
    input_shape : Tuple[int, int, int]
        Input shape for model
    """

    onnx_path = os.path.join(output_dir, f"{name}.onnx")

    dummy_input = torch.randn(1, *input_shape)
    
    # Export the model to onnx
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)


if __name__ == "__main__":
    print("Hello world")
    # path = r"output\HaGRID_Test\with_scheduler\best_model.pth"
    # logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

    # #load config
    # conf = OmegaConf.load("configs/default.yaml")
    
    # # set up dataset
    
    # train_dataset = ClassificationHaGridDataset(conf, op=DATASET_OPERATION.TRAIN, transform=Compose(conf, DATASET_OPERATION.TRAIN))
    # train_loader = DataLoader(train_dataset, batch_size=conf.train_params.train_batch_size, shuffle=False)
    # breakpoint()
    # #load model
    # model = torch.load(path)
    
    # for i, (img, label) in enumerate(train_loader):
    #     output = model(img)
    #     breakpoint()
    
    
    # save_onnx(model, "output", "gesture_model", input_shape=(3, 224, 224))