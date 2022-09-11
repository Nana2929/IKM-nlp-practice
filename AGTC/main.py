import pandas as pd
import random
import numpy as np
import torch
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torchmetrics import Accuracy
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from dataset import DataPreprocUtils
from model import LSTMNet
import wandb
from datetime import datetime
from argparse import ArgumentParser
import logging

# from config import Config
# ArgParser
# train, test file
# use pretrained embedding or not
# wandb
# call dataset
# call model
# call trainer fit
# display output

# 加上early stop
# 重寫config輸入方式
########################
trainfile = './data/train.csv'
testfile = './data/test.csv'
########################

def get_embeddings():
    import gdown
    # https://drive.google.com/file/d/1j8mcp6OltU2Fd7YhyPFcAPqhtXAXAqQD/view?usp=sharing
    ID = '1j8mcp6OltU2Fd7YhyPFcAPqhtXAXAqQD'
    url = f'https://drive.google.com/uc?id={ID}'
    output = 'agnews_fastext.model'
    gdown.download(url, output, quiet=True)

def main(args):
    SEED = args.seed
    pl.utilities.seed.seed_everything(SEED)
    wandb.login()
    runname = datetime.now().strftime('%m-%d_%H:%M:%S')
    wandb_logger = WandbLogger(
        project = 'IKM-nlp-practice',
        name = runname,
        config = vars(args),
    )
    p = DataPreprocUtils(
        trainpath = args.train_file,
        testpath = args.test_file,
        seed = SEED,
        maxlen = args.maxlen,
        batch_size = args.batch_size)

    p.preproc()
    trainloader, testloader = p.get_dataloader()
    # p._detok_random() # for debug purpose
    logging.info('Finished data prep.')
    model = LSTMNet(
        seed = SEED,
        vocabs = p.vocabs,
        is_bidirectional = args.is_bidirectional,
        lstm_hdim = args.lstm_hdim,
        embdim = args.embdim,
        num_layers = args.num_layers,
        hiddim = args.hiddim,
        numchoice = args.numchoice,
        lr = args.lr,
        dropout = args.dropout)
    logging.info('Finished initializing LSTM model.')
    trainer = pl.Trainer(max_epochs = 10,
                     accelerator="gpu",
                     devices = 1, # number of devices used in training, not device_id
                     logger = wandb_logger,
                     gradient_clip_val= args.clip_grad)
    trainer.fit(model, trainloader, testloader)
    logging.info('Finished training!')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-pe", "--use_pretrained", type=bool)
    parser.add_argument("--train_file", type=str, default = '../data/train.csv')
    parser.add_argument("--validation_file", type=str, default = '../data/test.csv')
    parser.add_argument("--test_file", type=str, default = '../data/test.csv' )
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--batch_size", type=int, default = 200)
    parser.add_argument("--lr", type=int, default = 5e-4)
    parser.add_argument("--num_layers", type=int, default = 2)
    parser.add_argument("--maxlen", type=int, default = 50)
    parser.add_argument("--dropout", type=int, default = 0.3)
    parser.add_argument("--clip_grad", type=int, default = 0.5)
    parser.add_argument("--numchoice", type=int, default = 4)
    parser.add_argument("--embdim", type=int, default = 32)
    parser.add_argument("--is_bidirectional", type=bool, default = False)
    parser.add_argument("--lstm_hdim", type=int, default = 256)
    parser.add_argument("--hiddim", type=int, default = 32)

    args = parser.parse_args()
    main(args)
