import pandas as pd
import utils
from argparse import ArgumentParser
import os
import importlib
from loguru import logger
from DatasetLoader import get_data_loader, test_dataset_loader
import torchvision.transforms as transforms
from models.EmbedNet import EmbedNet
from losses import LossFunction
import torch

from trainer import ModelTrainer

def init_dataloaders(config):
    train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Input transformations for evaluation
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=train_transform, **config['dataloader'])

    ## Read all lines
    with open(config['test_list']) as f:
        lines = f.readlines()
    files = sum([x.strip().split(',')[-2:] for x in lines],[])
    setfiles = list(set(files))
    setfiles.sort()

    test_dataset = test_dataset_loader(setfiles, config['test_path'], transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=5,
        drop_last=False,
    )

    return {'train': trainLoader, 'val': test_loader}

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_cfg",
                        default="./experiments/baseline.yaml", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.experiment_cfg)
    utils.seed_everything(config["seed"])

    dataloaders = init_dataloaders(config)
    
    logger.info("dataloader are loaded")

    criterion = LossFunction(**config['loss'])
    model = EmbedNet(config, criterion)

    if config["resume"]:
        # load prev weights
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
