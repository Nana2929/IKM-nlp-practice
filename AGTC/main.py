import pandas as pd
import random
import numpy as np
import torch
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
import argparse
import logging
from gensim.models import fasttext
import numpy as np
import pathlib
import joblib

def get_embeddings():
    script_loc = pathlib.Path(__file__).parent.resolve()
    outputdir = f'{script_loc}/../FastText-pretrained'
    model_name = 'agnews_fastext.model'
    if os.path.exists(outputdir):
        logging.info('Pretrained embeddings exist. Skipped downloading.')
        return os.path.join(outputdir, model_name)
    import gdown
    logging.info('Downloading pretrained embeddings (fastText)... ')
    ID = '1mgWOZmA9d97EVkEl6NBHCBLzdxlxg0VZ'
    url = f"https://drive.google.com/drive/folders/{ID}"
    gdown.download_folder(url, quiet = True)
    return os.path.join(outputdir, model_name)


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


    pretrained_embeddings = None
    if args.use_pretrained:
        embpath = get_embeddings()
        emodel = fasttext.FastText.load(embpath)
        mean, std = 0, 1
        pretrained_embeddings = np.random.normal(loc = mean, scale = std,
                                    size = (len(p.vocabs), args.embdim))
        for idx, word in enumerate(p.vocabs.get_itos()):
            embedding_vector = emodel.wv[word]
            if embedding_vector is not None:
                pretrained_embeddings[idx] = embedding_vector
        pretrained_embeddings = torch.from_numpy(pretrained_embeddings)

    model = LSTMNet(
        seed = SEED,
        vocabs = p.vocabs,
        is_bidirectional = args.is_bidirectional,
        lstm_hdim = args.lstm_hdim,
        embdim = args.embdim,
        pretrained_embeddings = pretrained_embeddings,
        num_layers = args.num_layers,
        hiddim = args.hiddim,
        numchoice = args.numchoice,
        lr = args.lr,
        dropout = args.dropout)
    logging.info('Finished initializing LSTM model.')

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath = args.model_dir,
        filename="AG-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max")

    trainer = pl.Trainer(max_epochs = 10,
                     accelerator="gpu",
                     devices = 1, # number of devices used in training, not device_id
                     logger = wandb_logger,
                     gradient_clip_val= args.clip_grad,
                     callbacks=[checkpoint_callback])
    trainer.fit(model, trainloader, testloader)
    logging.info('Finished training!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", action='store_true')
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

    parser.add_argument("--model_dir", type=str, default = './checkpoints')

    args = parser.parse_args()
    main(args)
    # Command line:
    # python3 main.py --use_pretrained  # use pretrained embeddings
    # python3 main.py                   # do NOT use pretrained embeddings

