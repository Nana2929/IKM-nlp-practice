Config = {
    # 'IS_BIDIRECTIONAL':False,
    'SEED':42,
    'NUM_WORKER':12,
    'BATCHSIZE': 200,
    'LR':5e-4,
    'NUM_LAYERS':2,
    'MAXLEN':50,
    'DATASET':'AGNews',
    'EPOCHS':5,
    'WEIGHT_DECAY':1e-5,  #default:0
    'DROPOUT':0.3,
    'CLIP_GRAD':0.5,
    'NUMCHOICE':4,
    # 'LSTM_HDIM':256,
    'EMBDIM':32,
    'HIDDIM':32
}
