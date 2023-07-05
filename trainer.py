import torch
from utils.util import set_random_state, build_model
from utils.transform import Compose
from utils.constants import DATASET_OPERATION
from dataset.dataset import ClassificationHaGridDataset


class ClassificationTrainer:
    def __init__(self, conf) -> None:
        self.conf = conf

        
    def initialize(self):
        set_random_state(self.conf.random_seed)
        self._initiate_model()
        self.optimizer = self._get_optimizer()
        self.num_classes = len(self.conf.dataset.targets)
    
    def _initiate_model(self):
        set_random_state(self.conf.random_seed)

        self.model = build_model(
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
    
    def run_train(self):
        epochs = self.conf.train_params.epochs
        
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

        return model