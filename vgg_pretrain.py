import utils
from argparse import ArgumentParser
from DatasetLoader import VGG_dataset
import torchvision.transforms as transforms
from EmbedNet import EmbedNet
from trainer import ModelTrainer
from loguru import logger
import timm
import os
from DatasetLoader import get_data_loader

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    return accuracies["precision_at_1"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_cfg",
                        default="./experiments/vgg_baseline.yaml", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.experiment_cfg)
    utils.seed_everything(config["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(config['gpu'])
    
    # ------ DATASET -------
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.RandomCrop([224,224]),
         transforms.RandomHorizontalFlip(p = 0.3),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Input transformations for evaluation
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.CenterCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataset = VGG_dataset("./vgg_data/train.csv", train_transform, size = 0.05)
    valid_dataset = VGG_dataset("./vgg_data/valid.csv", test_transform, size = 0.02)

    logger.info(f'Size of train {len(train_dataset)}, Size of valid {len(valid_dataset)}')
    
    trainLoader = get_data_loader(transform=train_transform, **config['dataloader']);

    model = EmbedNet(config).cuda()
    trainer     = ModelTrainer(model, **config['trainer'])

    num_epochs = config['max_epoch']

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    best_test_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        logger.info(f'Start of {epoch}/{num_epochs}')
        train_loss = trainer.train_network(trainLoader)
        logger.info(f'Train loss {train_loss}')

        test_acc = test(train_dataset, valid_dataset, model, accuracy_calculator)
        if test_acc > best_test_acc:
            trainer.saveParameters("./logs/best_pretrained_model.model");

        logger.info(f'End of {epoch}/{num_epochs}')
    
    logger.info(f'Training is finished with best testing accuracy {best_test_acc}')


if __name__ == "__main__":
    main()
    
