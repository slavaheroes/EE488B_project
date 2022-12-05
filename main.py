#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import time
import os
from argparse import ArgumentParser
import pdb
import glob
import datetime
from utils import *
import utils
from trainer import ModelTrainer
from EmbedNet import EmbedNet
from DatasetLoader import get_data_loader
import torchvision.transforms as transforms

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_cfg",
                        default="./experiments/baseline.yaml", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.experiment_cfg)
    utils.seed_everything(config["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(config['gpu'])

    ## Load models
    s = EmbedNet(config).cuda();

    it          = 1

    ## Input transformations for training
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([224,224]),
        #  transforms.RandomCrop([224,224]),
         transforms.RandomHorizontalFlip(p = 0.3),
         transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
         transforms.Normalize(mean=0.5, std=0.5)])

    ## Input transformations for evaluation
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
        #  transforms.CenterCrop([224,224]),
         transforms.Normalize(mean=0.5, std=0.5)])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=train_transform, **config['dataloader']);
    trainer     = ModelTrainer(s, **config['trainer'])

    config['save_path'] = config['save_path'] + f'/{config["exp_name"]}'
    os.makedirs(config['save_path'], exist_ok = True)
    ## Load model weights
    modelfiles = glob.glob('{}/*.model'.format(config['save_path']))
    modelfiles.sort()

    ## If the target directory already exists, start from the existing file
    if len(modelfiles) >= 1 and config['resume']:
        trainer.loadParameters(modelfiles[-1]);
        print("Model {} loaded from previous state!".format(modelfiles[-1]));
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    if(config["initial_model"] != ""):
        trainer.loadParameters(config["initial_model"]);
        print("Model {} loaded!".format(config["initial_model"]));
    
    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in s.__S__.parameters())
    print('Total model parameters: {:,}'.format(pytorch_total_params))

     ## Evaluation code 
    if config["eval"] == True:
        sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **config["evaluate"])
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print('EER {:.4f}'.format(result[1]))

        if config['output'] != '':
            with open(config['output'],'w') as f:
                for ii in range(len(sc)):
                    f.write('{:4f},{:d},{}\n'.format(sc[ii],lab[ii],trials[ii]))

        quit();
    
    ## Write config to scorefile for training
    scorefile = open(config['save_path']+"/scores.txt", "a+");

    strtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scorefile.write('{}\n'.format(strtime))
    utils.save_yaml(config, config['save_path']+"/config.yaml")
    scorefile.flush()

    ## Core training script
    minimal_eer = 100
    for it in range(it,config["max_epoch"]+1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training epoch {:d} with LR {:.5f} ".format(it,max(clr)));

        loss = trainer.train_network(trainLoader);

        if it % config["test_interval"] == 0:
            
            sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **config["evaluate"])
            print(min(sc), max(sc))
            result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
            
            print("IT {:d}, Val EER {:.5f}".format(it, result[1]));
            scorefile.write("IT {:d}, Val EER {:.5f}\n".format(it, result[1]));

            if result[1] < minimal_eer:
                trainer.saveParameters(config['save_path']+"/best_model.model");
                minimal_eer = result[1]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "TLOSS {:.5f}".format(loss));
        scorefile.write("IT {:d}, TLOSS {:.5f}\n".format(it, loss));

        scorefile.flush()
    else:
        trainer.saveParameters(config['save_path']+"/last_epoch_{}.model".format(it));


    print("Best score {:.5f}".format(minimal_eer));
    scorefile.write("Best score is {:.5f}\n".format(minimal_eer));
    scorefile.close();



if __name__ == "__main__":
    main()
