import pandas as pd
import utils
from argparse import ArgumentParser
import os
import importlib
from loguru import logger
from DatasetLoader import get_data_loader, test_dataset_loader, meta_loader
import torchvision.transforms as transforms
from models.EmbedNet import EmbedNet
import torch
import torchvision

from trainer import ModelTrainer

def init_dataloaders(config):
    train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    train_dataset = torchvision.datasets.ImageFolder(root=config['dataloader']['train_path'], transform=train_transform)
    trainLoader = torch.utils.data.DataLoader(train_dataset,\
        batch_size = config['dataloader']['batch_size'],\
            num_workers = config['dataloader']['nDataLoaderThread'],
            shuffle = True)
    
    # trainLoader = get_data_loader(transform=train_transform, **config['dataloader'])

    return {'train': trainLoader}

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_cfg",
                        default="./experiments/baseline.yaml", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.experiment_cfg)
    utils.seed_everything(config["seed"])

    dataloaders = init_dataloaders(config)
    
    logger.info("dataloader are loaded")

    model = EmbedNet(config)

    if config["resume"]:
        # load prev or pretrained weights
        state_dict = torch.load(config['resume_path'])['state_dict']
        model.load_state_dict(state_dict)
        logger.info(f'Model weights are loaded')

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of parameters: {pytorch_total_params}')
    
    module = importlib.import_module(config["OPTIMIZER"]["PY"])
    optimizer = getattr(module, config["OPTIMIZER"]["CLASS"])(
        model.parameters(), **config["OPTIMIZER"]["ARGS"])

    module = importlib.import_module(config["SCHEDULER"]["PY"])
    scheduler = getattr(module, config["SCHEDULER"]["CLASS"])(
        optimizer, **config["SCHEDULER"]["ARGS"])
    
    logger.info("hyperparameters are loaded")

    learner = ModelTrainer(
        model = model,
        dataloaders = dataloaders,
        optimizer = optimizer,
        scheduler = scheduler,
        config = config,
        **config['trainer']
    )

    learner.fit()



if __name__ == "__main__":
    main()
