import utils
from argparse import ArgumentParser
from DatasetLoader import VGG_dataset
import torchvision.transforms as transforms
from models.EmbedNet import EmbedNet
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
import timm
import os

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch

def train(model, loss_func, mining_func, loader, optimizer):
    loss = 0
    counter = 0
    with tqdm(loader, unit="batch") as tepoch:
        for data, labels in tepoch:
            data, labels = data.to("cuda:1"), labels.to("cuda:1")
            optimizer.zero_grad()
            embeddings = model(data)
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            loss.backward()
            optimizer.step()
            loss    += loss.detach().cpu().item()
            counter += 1
            tepoch.set_postfix(loss=loss/counter)

    return loss

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_cfg",
                        default="./experiments/baseline.yaml", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.experiment_cfg)
    utils.seed_everything(config["seed"])
    
    # ------ DATASET -------
    train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataset = VGG_dataset("./vgg_data/train.csv", train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, num_workers=5
    )

    model = EmbedNet(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.4, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.4, distance=distance, type_of_triplets="semihard"
    )

    num_epochs = 2
    model.to("cuda:1")
    for epoch in range(1, num_epochs + 1):
        logger.info(f'Start of {epoch}/{num_epochs}')
        train_loss = train(model, loss_func, mining_func, train_loader, optimizer)
        scheduler.step()
        logger.info(f'Train loss {train_loss}')
        logger.info(f'End of {epoch}/{num_epochs}')
        
    model.cpu()
    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "state_dict": model.state_dict(),
    }, os.path.join("./logs/", "best_pretrained_model.pth"))
    logger.info('Training is finished')


if __name__ == "__main__":
    main()
    
