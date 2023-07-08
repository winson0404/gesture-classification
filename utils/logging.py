import wandb
from omegaconf import OmegaConf

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
        
        # TODO: change config and edit this:
        self.wandb.init(
            project=self.project_name, 
            name=self.experiment_name, 
            entity=self.args.entity, 
            config={
                # "learning_rate": self.conf.learning_rate,
                # "train_batch_size": self.conf.traionbatch_size,
                # "test_batch_size": self.conf.test_batch_size,
                # "epoches": self.conf.epoches,
            })

    def is_logged_in():
        return wandb.api.api_key is not None
    
    def log_metrics(self, metrics):
        self.wandb.log(**metrics)

    def log_image_table(self, images, predicted, labels, probs):
        "Log a wandb.Table with (img, pred, target, scores)"
        # 🐝 Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
        for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
            table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
        self.wandb.log({"predictions_table":table}, commit=False)
    
    def finish(self):
        self.wandb.finish()