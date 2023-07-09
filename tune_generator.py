import random
from omegaconf import OmegaConf
import os

setting = {
    "lr": [0.001, 0.005, 0.01],
    "momentum": [0.9, 0.85, 0.8],
    "weight_decay": [0.0001, 0.0005, 0.001],
    "batch_size": [16, 32, 64],
    "optimizer": ["sgd", "adam", "rmsprop"],
    "epochs": [10, 20, 30],
}

random_names = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

variation_numbers = 6

# choose 3 different setting randomly
# random_seed = 42

config_root = "configs"

conf = OmegaConf.load("configs/default.yaml")
project_name = "MobileNetSmall"
for i in range(variation_numbers):
    conf.project_name = project_name
    
    # random settings
    conf.optimizer.optimizer = random.choice(setting["optimizer"])
    conf.optimizer.lr = random.choice(setting["lr"])
    conf.optimizer.momentum = random.choice(setting["momentum"])
    conf.optimizer.weight_decay = random.choice(setting["weight_decay"])
    conf.train_params.epochs = random.choice(setting["epochs"])
    conf.train_params.train_batch_size = random.choice(setting["batch_size"])
    conf.train_params.validation_batch_size = random.choice(setting["batch_size"])
    experiment_name = random.choice(random_names) + "_" + conf.optimizer.optimizer + "_" + str(conf.train_params.epochs)   
    conf.experiment_name = experiment_name
    
    # save config to config_root
    save_path = os.path.join(config_root, experiment_name + ".yaml")
    OmegaConf.save(conf, save_path)