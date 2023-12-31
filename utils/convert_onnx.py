import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from typing import Tuple
from dataset.dataset import ClassificationHaGridDataset
from omegaconf import OmegaConf
from utils.constants import DATASET_OPERATION, MOBILENETV3_SIZE
from utils.transform import Compose
import logging
from models.classification import MobileNetV3
from utils.util import build_model

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
    # change to cuda both model and dummy_input to cpu
    model = model.to("cpu")
    dummy_input = dummy_input.to("cpu")
    
    # Export the model to onnx
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,
        input_names=["images"],
        output_names=["output"])


if __name__ == "__main__":
    # print("Hello world")
    path = r"output\HaGRID_Test\with_scheduler\best_model.pth"
    #load config
    conf = OmegaConf.load("configs/default.yaml")

    model = MobileNetV3(
            num_classes=len(conf.dataset.targets), 
            model_size=MOBILENETV3_SIZE.SMALL, 
            pretrained=False, 
            freezed=False)
    # logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

    # transform = Compose()
    # # set up dataset
    # train_dataset = ClassificationHaGridDataset(
    #         conf,
    #         op=DATASET_OPERATION.TRAIN,
    #         transform=transform
    #     )
    # # train_dataset = ClassificationHaGridDataset(conf, DATASET_OPERATION.TRAIN, Compose(conf, DATASET_OPERATION.TRAIN))
    # train_loader = DataLoader(train_dataset, batch_size=conf.train_params.train_batch_size, shuffle=False)
    # breakpoint()
    #load model
    state_dict = torch.load(path)["state_dict"]
    # breakpoint()
    model.load_state_dict(state_dict)
    # for i, (img, label) in enumerate(train_loader):
    #     output = model(img)
    #     breakpoint()
    
    save_path = r"output\HaGRID_Test\with_scheduler\\"
    save_onnx(model, save_path, "gesture_model", input_shape=(3, 224, 224))