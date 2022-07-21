import numpy as np
import pandas as pd
import pickle
import torch
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    val_data = data[train_size:int(time_len*(rate+0.2))]
    test_data = data[int(time_len*(rate+0.2)):time_len]

    trainX, trainY, valX,valY,testX, testY = [], [], [], [],[],[]
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])

    for i in range(len(val_data)-seq_len-pre_len):
        c = val_data[i: i + seq_len + pre_len]
        valX.append(c[0: seq_len])
        valY.append(c[seq_len: seq_len + pre_len])

    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    valX1 = np.array(valX)
    valY1 = np.array(valY)


    mean,std = np.mean(trainX1),np.std(trainX1)
    trainX1 = (trainX1-mean)/std
    valX1 = (valX1-mean)/std
    testX1 = (testX1-mean)/std

    trainX1 = torch.tensor(trainX1,dtype=torch.float32)
    trainY1 = torch.tensor(trainY1,dtype=torch.float32)
    testX1 = torch.tensor(testX1,dtype=torch.float32)
    testY1 = torch.tensor(testY1,dtype=torch.float32)
    valX1 = torch.tensor(valX1,dtype=torch.float32)
    valY1 = torch.tensor(valY1,dtype= torch.float32)

    return trainX1, trainY1,valX1,valY1,testX1, testY1,mean,std

def load_data(data):
    if data=="PEMS03":
        adj_mx = np.load("data/PEMS03/adj.npy")
        x = np.load("data/PEMS03/PEMS03.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data=="PEMS04":
        adj_mx = np.load("data/PEMS04/adj.npy")
        x = np.load("data/PEMS04/PEMS04.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data=="PEMS07":
        adj_mx = np.load("data/PEMS07/adj.npy")
        x = np.load("data/PEMS07/PEMS07.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    elif data == "PEMS08":
        adj_mx = np.load("data/PEMS08/adj.npy")
        x = np.load("data/PEMS08/PEMS08.npz")["data"][:, :, 0:1]
        B, N, H = x.shape
        x = x.reshape(B, N)
    adj = torch.tensor(np.array(adj_mx), dtype=torch.float32)
    return x, adj


