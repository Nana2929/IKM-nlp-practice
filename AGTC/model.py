'''
References:
https://zhenglungwu.medium.com/pytorch%E5%AF%A6%E4%BD%9Clstm%E5%9F%B7%E8%A1%8C%E8%A8%8A%E8%99%9F%E9%A0%90%E6%B8%AC-d1d3f17549e7
https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

'''
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics # Accuracy


class LSTMNet(pl.LightningModule):
    def __init__(self, seed, vocabs, pretrained_embeddings,
                is_bidirectional, lstm_hdim,
                embdim, num_layers, hiddim,
                numchoice, lr, dropout):

        super().__init__()
        pl.utilities.seed.seed_everything(seed)
        # variables
        self.num_directions = 2 if is_bidirectional else 1
        self.lstm_hdim = lstm_hdim
        self.lr = lr

        # loss function
        self.LossFn = nn.CrossEntropyLoss()

        # layers
        self.Embedding = nn.Embedding(len(vocabs),
                                        embdim,
                                        padding_idx = vocabs['PAD'])
        if pretrained_embeddings is not None:
            self.Embedding = nn.Embedding(len(vocabs),
                                        embdim,
                                        padding_idx = vocabs['PAD'])
            self.Embedding.load_state_dict({'weight': pretrained_embeddings})
            # self.Embedding.weight.requires_grad = False
        self.LSTM = nn.LSTM(
            input_size = embdim,
            hidden_size = lstm_hdim,
            num_layers = num_layers,
            bidirectional = is_bidirectional,
            batch_first = True,
        )
        self.FC = nn.Linear(lstm_hdim*self.num_directions,
                            hiddim)
        self.Dropout = nn.Dropout(dropout)
        self.Out  = nn.Linear(hiddim, numchoice)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.Embedding(x)
        x, (h, c) = self.LSTM(x)
        x = self.FC(x)
        x = self.Dropout(x)
        x = self.Out(x)
        x = x[:, -1]
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        X, labels = batch
        preds = self(X)
        loss = self.LossFn(preds, labels)
        accuracy = self.accuracy(torch.argmax(preds, axis=1),
                                 labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        X, labels = batch
        preds = self(X)
        loss = self.LossFn(preds, labels)
        accuracy = self.accuracy(torch.argmax(preds, axis=1), labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr)
        return optimizer

