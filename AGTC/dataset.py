import pandas as pd
import random
import numpy as np
import torch
from torchtext.data import get_tokenizer
import nltk
import re
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset, DataLoader
from config import Config
from utils import *
import logging

class DataPreprocUtils:

    def __init__(self, trainpath,
                testpath):
        # trainpath = './data/train.csv'
        # testpath = './data/test.csv'
        self.train = pd.read_csv(trainpath)
        self.test = pd.read_csv(testpath)
        self.stopwords = self._get_stopwords() # a set
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)

    def preproc(self):

        tokenizer = get_tokenizer("basic_english")
        self.train['preproc_text'] = self.train.apply(lambda x:self._concat(x['Title'], x['Description']), axis=1)
        self.test['preproc_text'] = self.test.apply(lambda x:self._concat(x['Title'], x['Description']), axis=1)
        self.train['label'] = self.train['Class Index'].apply(lambda x:x-1)
        self.test['label'] = self.test['Class Index'].apply(lambda x:x-1)
        vocabs = build_vocab_from_iterator(self._build_vocab([self.train, self.test]),
                                        specials = ["[PAD]", "[UNK]"],
                                        max_tokens = None)
        vocabs.set_default_index(vocabs["[UNK]"]) # let padding be index 0
        self.vocabs = vocabs
        self.log.info(f'vocab size (including specials): {len(self.vocabs)}')
        trainX, trainy = self._df2tensor(self.train)
        testX, testy = self._df2tensor(self.test)
        self.trainset = TensorDataset(trainX, trainy)
        self.testset = TensorDataset(testX, testy)



    def _df2tensor(self, split):
        X = split['preproc_text'].apply(lambda x:self._tokenize(x))
        y = split['label']
        _X = torch.tensor(np.vstack(X), dtype=torch.long)
        _y = torch.tensor(y, dtype=torch.long)
        del X, y
        return _X, _y

    def get_dataloader(self):
        self.trainloader = DataLoader(self.trainset, shuffle = True,
                                num_workers = Config['NUM_WORKER'],
                                batch_size=Config['BATCHSIZE'])
        self.testloader = DataLoader(self.testset, shuffle = False,
                                num_workers = Config['NUM_WORKER'],
                                batch_size=Config['BATCHSIZE'])
        self.log.info('Finished making 2 dataloaders.')
        return self.trainloader, self.testloader

    def _detok_random(self):
        r = random.randint(0, trainX.shape[0])
        detokenizer = self.vocabs.get_itos()
        self.log.debug(f'random index: {r}')
        # “World”: 0,“Sports”: 1,“Business”: 2,“Sci/Tech”: 3
        x = self.trainX[r,:]
        self.log.debug(f'tokenized example:\n {x}')
        self.log.debug(f'label: {LabelMapping[self.trainy[r].item()]}')
        self.log.debug(f'detokenized example:')
        self.log.debug([detokenizer[tok] for tok in x])


    def _tokenize(self, x):
        maxlen = Config['MAXLEN']           # nltk.word_tokenize, spacy (better tokenize)
        x = x[:maxlen]                      # truncation
        tokenized_x = np.zeros(maxlen)      # padding: np.ones since [PAD]: 0
        tokenized_x[:len(x)] = np.array([self.vocabs[v] for v in x])
        return tokenized_x

    def _get_stopwords(self):
        try:
            from nltk.corpus import stopwords
            stopwords = set(stopwords.words('english'))
        except LookupError as e:
            print(f"{e}, {e.__class__}")
            nltk.download('stopwords, punct')
            from nltk.corpus import stopwords
            stopwords = set(stopwords.words('english'))
        return stopwords

    def _concat(self, title, descrip):
        text = title + '.' + descrip
        text = re.sub(r'[~`\-!@#$%^&*():;"{}_/?><\|.,`0-9\\]',' ',text)
        text = nltk.word_tokenize(text)
        # alternatively:
        # text = text.split()
        text = [tok.lower() for tok in text if tok not in self.stopwords]
        return text

    def _build_vocab(self, datasets):
        for dataset in datasets:
            for rid, data in dataset.iterrows():
                text = data['preproc_text']
                yield text



# p = DataPreprocUtils(
#     trainpath = './data/train.csv',
#     testpath = './data/test.csv'
# )
# p.preproc()
# trainl, testl = p.get_dataloader()



