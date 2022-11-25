import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import utils
from loguru import logger
import torch.nn.functional as F
import numpy, math, pdb, sys
import pandas as pd

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

    def train_epoch(self, epoch, loader):
        self.model.to(self.device)
        self.model.train()

        stepsize = loader.batch_size

        counter = 0
        index   = 0
        loss    = 0

        with tqdm(loader, unit="batch") as tepoch:
            for data, label in tepoch:
                tepoch.total = tepoch.__len__()

                ## Reset gradients
                self.optimizer.zero_grad()
                
                ## Forward and backward passes
                if self.mixedprec:
                    with autocast():
                        embeds, nloss = self.model(data.to(self.device), label.to(self.device))
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()      
                else:
                    embeds, nloss = self.model(data.to(self.device), label.to(self.device))
                    nloss.backward()
                    self.optimizer.step()
                
                loss    += nloss.detach().cpu().item()
                counter += 1
                index   += stepsize

                tepoch.set_postfix(loss=loss/counter)
                if self.lr_step == 'iteration':
                    self.scheduler.step()
            
            if self.lr_step == 'epoch':
                self.scheduler.step()
    
        return loss/counter

    def val_epoch(self, epoch, test_loader, test_list):
        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        self.model.to(self.device)
        self.model.eval()

        feats       = {}
        for data in tqdm(test_loader):
            inp1                = data[0][0].to(self.device)
            ref_feat            = self.model(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        logger.info('Computing similarities')

        ## Read files and compute all scores
        for line in tqdm(lines):

            data = line.strip().split(',')

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]

            score = F.cosine_similarity(ref_feat, com_feat)

            all_scores.append(score.item()) 
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + "," + data[2])

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

            sc, lab, trials = self.val_epoch(epoch, self.dataloaders['val'], self.config['test_list'])
            tunedThreshold, eer, fpr, fnr = utils.tuneThresholdfromScore(sc, lab, [1, 0.1])

            epoch_metrics['val_eer'] = eer
            epoch_metrics['threshold'] = tunedThreshold

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

            logger.info(f'Epoch {epoch} is finished')
        
        logger.info("Training is finished")
        logger.info(f"Best EER is {best_score}")
            

        