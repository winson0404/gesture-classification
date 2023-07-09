import wandb
from omegaconf import OmegaConf
from typing import Dict, List, Tuple

class Logger:
    def __init__(self, project_name:str, experiment_name:str, conf:OmegaConf) -> None:
        self.conf = conf
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.wandb = wandb
        self.init()

    def init(self):
        if not self.is_logged_in():
            self.wandb.login()
        
        self.wandb.init(
            project=self.project_name, 
            name=self.experiment_name, 
            config={
                "model": self.conf.model.name,
                "epochs": self.conf.train_params.epochs,
                "train_batch_size": self.conf.train_params.train_batch_size,
                "validation_batch_size": self.conf.train_params.validation_batch_size,
                "optimizer": self.conf.optimizer.optimizer,
                "learning_rate": self.conf.optimizer.lr,
                "momentum": self.conf.optimizer.momentum if self.conf.optimizer.optimizer != "adam" else None,
            })

    def is_logged_in(self):
        return wandb.api.api_key is not None
    
    def log(self, category:str, metrics:Dict):
        new_dict = {}
        for key, value in metrics.items():
            new_dict[f"{category}/{key}"] = value
        self.wandb.log(new_dict)

    def log_image_table(self, images, predicted, labels, probs):
        "Log a wandb.Table with (img, pred, target, scores)"
        # Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
        for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
            table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
        self.wandb.log({"predictions_table":table}, commit=False)
    
    def finish(self):
        self.wandb.finish()