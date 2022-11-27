import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import utils
from loguru import logger
import torch.nn.functional as F
import numpy, math, pdb, sys
import pandas as pd
from DatasetLoader import get_data_loader, test_dataset_loader, meta_loader
import torchvision.transforms as transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers

class ModelTrainer:
    def __init__(self, 
                model, 
                dataloaders, 
                optimizer, 
                scheduler,
                lr_step,
                mixedprec, 
                gpu, 
                config):

        self.model = model
        self.dataloaders = dataloaders

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr_step = lr_step
        self.scaler = GradScaler()

        assert self.lr_step in ['epoch', 'iteration']

        self.mixedprec = mixedprec
        self.device = gpu
        self.config = config
        
        os.makedirs(f'logs/{config["exp_name"]}', exist_ok = True)
        self.logdir = f'logs/{config["exp_name"]}/'
        self.history_csv = f'logs/{config["exp_name"]}/history.csv'
        utils.save_yaml(self.config, self.logdir + "config.yaml")

        ## pytorch metric learning stuff
        self.mining_func = miners.TripletMarginMiner(
            margin=0.3, type_of_triplets="semihard"
        )
        self.tripletMarginLoss = losses.TripletMarginLoss(margin=0.3)
        self.arcFaceLoss = losses.SubCenterArcFaceLoss(num_classes=self.config['loss']['nClasses'],\
                        embedding_size=self.config['loss']['nOut']).to(self.device)
        self.marginSoftmax = losses.LargeMarginSoftmaxLoss(num_classes=self.config['loss']['nClasses'],\
                        embedding_size=self.config['loss']['nOut']).to(self.device)

        self.loss_optimizer = torch.optim.SGD(list(self.marginSoftmax.parameters()) +\
                                                list(self.arcFaceLoss.parameters()),\
                                                lr=0.001)


    def loss_func(self, embeddings, labels):
        indices_tuple = self.mining_func(embeddings, labels)
        triplet = self.tripletMarginLoss(embeddings, labels, indices_tuple)
        arcFace = self.arcFaceLoss(embeddings, labels)
        softmax = self.marginSoftmax(embeddings, labels)
        loss = softmax + triplet + arcFace
        return loss
        
        
    def train_epoch(self, epoch, loader):
        self.model.to(self.device)
        self.model.train()

        stepsize = loader.batch_size

        counter = 0
        index   = 0
        loss    = 0

        with tqdm(loader, unit="batch") as tepoch:
            for data, labels in tepoch:
                tepoch.total = tepoch.__len__()

                ## Reset gradients
                self.optimizer.zero_grad()
                self.loss_optimizer.zero_grad()

                labels = labels.to(self.device)
                
                ## Forward and backward passes
                embeddings = self.model(data.to(self.device))
                nloss = self.loss_func(embeddings, labels)
                nloss.backward()
                self.optimizer.step()
                self.loss_optimizer.step()

                loss    += nloss.detach().cpu().item()
                counter += 1
                index   += stepsize

                tepoch.set_postfix(loss=loss/counter)

                if self.lr_step == 'iteration':
                    self.scheduler.step()
            
            if self.lr_step == 'epoch':
                self.scheduler.step()
    
        return loss/counter

    @torch.no_grad()
    def val_epoch(self):
        self.model.to(self.device)
        self.model.eval()

        feats       = {}
        ## Read all lines
        with open(self.config['test_list']) as f:
            lines = f.readlines()
        
        ## Get a list of unique file names
        files = sum([x.strip().split(',')[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Input transformations for evaluation
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        test_dataset = test_dataset_loader(setfiles, self.config['test_path'], transform=transform,\
                                num_eval=10)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config['dataloader']['nDataLoaderThread'],
            drop_last=False,
        )

        print('Generating embeddings')

        ## Extract features for every image
        for data in tqdm(test_loader):
            inp1                = data[0][0].to(self.device)
            ref_feat            = self.model(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        print('Computing similarities')

        ## Read files and compute all scores
        for line in tqdm(lines):

            data = line.strip().split(',');

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]

            score = F.cosine_similarity(ref_feat, com_feat)

            all_scores.append(score.item());  
            all_labels.append(int(data[0]));
            all_trials.append(data[1] + "," + data[2])

        print(min(all_scores), max(all_scores))
        return all_scores, all_labels, all_trials
    

    def fit(self):
        epochs = self.config["epochs"]
        logging_csv = []
        best_score = 100

        for epoch in range(epochs):
            epoch_metrics = {"epoch": epoch}
            logger.info(f'Start of {epoch+1}/{epochs} epoch.')

            train_loss = self.train_epoch(epoch, self.dataloaders['train'])
            epoch_metrics['train_loss'] = train_loss
            logger.info(f'Training loss {train_loss}')

            sc, lab, trials = self.val_epoch()
            tunedThreshold, eer, fpr, fnr = utils.tuneThresholdfromScore(sc, lab)

            epoch_metrics['val_eer'] = eer

            if eer < best_score:
                best_score = eer
                self.model.cpu()
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                }, os.path.join(self.logdir, "best_model.pth"))
            
            logger.info(f'Val EER {eer}')

            logging_csv.append(epoch_metrics)
            pd.DataFrame(logging_csv).to_csv(self.history_csv, index = None)

            logger.info(f'Epoch {epoch+1} is finished')
        
        logger.info("Training is finished")
        logger.info(f"Best EER is {best_score}")
            

        