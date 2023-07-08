import torch
from torch.utils.data import DataLoader
from utils.util import set_random_state, build_model, save_checkpoint, get_metrics
from utils.transform import Compose
from utils.constants import DATASET_OPERATION
from utils.logging import Logger
from dataset.dataset import ClassificationHaGridDataset
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import math
from collections import defaultdict

class ClassificationTrainer:
    def __init__(self, conf:OmegaConf, output_path:str, logger:Logger = None) -> None:
        # class parameters
        self.conf = conf
        self.output_path = output_path
        self.logger = logger
        self._initialize()
        
    def _initialize(self):
        # sequence matter
        
        # training parameters
        set_random_state(self.conf.random_seed)
        self.num_classes = len(self.conf.dataset.targets)
        self.epochs = self.conf.train_params.epochs
        self.log_writter = None
        self.model = self._initiate_model()
        self.optimizer = self._get_optimizer()
        # breakpoint()

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
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.conf.optimizer.optimizer == "adam":
            optimizer = torch.optim.Adam(trainable_params, lr=self.conf.optimizer.lr)
        elif self.conf.optimizer.optimizer == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=self.conf.optimizer.lr, momentum=self.conf.optimizer.momentum)
        elif self.conf.optimizer.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(trainable_params, lr=self.conf.optimizer.lr, momentum=self.conf.optimizer.momentum)
        
        return optimizer
    
    def _run_epoch(self, device:str, current_epoch:int,  train_loader:ClassificationHaGridDataset)->None:
        
        # set to train mode
        self.model.train()
        
        # initialize loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # run based on batch
        with tqdm(train_loader, unit="batch") as batch_loader:
            for i, (images, labels) in enumerate(batch_loader):
                
                # breakpoint()
                # shape to (batch_size, 3, 244, 244)
                images = torch.stack(list(image.to(device) for image in images))
                
                # calculate overall step currently
                step = i + len(train_loader) * current_epoch
                
                # go through the network and get output
                output = self.model(images)

                loss = criterion(output, labels)
            
                loss_value = loss.item()
                
                if self.logger is not None:
                    self.logger.log("train", {"loss": loss_value})

                if not math.isfinite(loss_value):
                    logging.info("Loss is {}, stopping training".format(loss_value))
                    exit(1)
                    
                loss.backward()
                
                self.optimizer.zero_grad()
                self.optimizer.step()

    def _run_eval(self, device:str, current_epoch:int, validation_loader:ClassificationHaGridDataset, mode:str="valid")->None:
        f1_score = None
        
        # theres a chance that user dont wanna do validation
        if validation_loader is not None:
            # stop gradient update
            with torch.no_grad():
                # set to non training mode:
                self.model.eval()
                predicts, targets = [], []
                with tqdm(validation_loader, unit="batch") as tepoch:
                    tepoch.set_description(f"{mode} Epoch {current_epoch}")
                    for i, (images, labels) in enumerate(tepoch):
                        images = torch.stack(list(image.to(self.conf.device) for image in images))
                        output = self.model(images)
                        predicts.extend(output.argmax(dim=1).cpu().numpy())
                        targets.extend(labels.numpy())
                        
                    
                    metric = get_metrics(predicts, targets)
                    f1_score = metric["f1_score"]
                    
                    if self.logger is not None:
                        self.logger.log("validation", metric)
                    
        
        return f1_score
                    
    
    def run_train(self)->None:
        
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
        
        train_loader = DataLoader(train_dataset, batch_size=self.conf.train_params.train_batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.conf.train_params.validation_batch_size, shuffle=False)
        
        best_metric = -1
        conf_dictionary = OmegaConf.to_container(self.conf)
        for i in range(self.epochs):
            logging.info(f"Epoch {i+1}/{self.epochs}")
            self._run_epoch(device=self.conf.device, current_epoch=i, train_loader=train_loader)
            metric_value = self._run_eval(device=self.conf.device, current_epoch=i, validation_loader=validation_loader)
            
            # get checkpoint based on f1 score
            if metric_value > best_metric:
                best_metric = metric_value
                save_checkpoint(self.output_path, conf_dictionary, self.model, self.optimizer, self.epoch, "best_model")

        return model
    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)