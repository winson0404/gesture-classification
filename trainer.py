import torch
from utils.util import set_random_state, build_model
from utils.transform import Compose
from utils.constants import DATASET_OPERATION
from utils.logging import Logger
from dataset.dataset import ClassificationHaGridDataset
import logging
from omegaconf import OmegaConf

class ClassificationTrainer:
    def __init__(self, conf:OmegaConf, logger:Logger = None) -> None:
        # class parameters
        self.conf = conf
        self._initialize()
        self.logger = logger
        
    def _initialize(self):
        # training parameters
        set_random_state(self.conf.random_seed)
        self.model = self._initiate_model()
        self.optimizer = self._get_optimizer()
        self.num_classes = len(self.conf.dataset.targets)
        self.epochs = self.conf.train_params.epochs
        self.log_writter = None

    def _initiate_model(self):
        set_random_state(self.conf.random_seed)

        return build_model(
            model_name=self.conf.model.name,
            num_classes=self.num_classes,
            checkpoint=self.conf.model.get("checkpoint", None),
            device=self.conf.device,
            pretrained=self.conf.model.pretrained,
            freezed=self.conf.model.freezed,
        )
        
        
    def _get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.train_params.lr)
        return optimizer
    

    def _run_epoch(self):
        pass

    def _run_eval(self):
        pass
    
    def run_train(self):
        
        model = self._initiate_model()
        
        transform = Compose()
        
        train_dataset = ClassificationHaGridDataset(
            self.conf,
            op=DATASET_OPERATION.TRAIN,
            transform=transform
        )
        
        validation_dataset = ClassificationHaGridDataset(
            self.conf,
            op=DATASET_OPERATION.VALIDATION,
            transform=transform
        )
        
        best_metric = -1
        for i in range(self.epochs):
            logging.info(f"Epoch {i+1}/{self.epochs}")
            self._run_epoch()
            metric_value = self._run_eval()
            if metric_value > best_metric:
                best_metric = metric_value
                torch.save(model.state_dict(), self.conf.model.save_path)

        return model