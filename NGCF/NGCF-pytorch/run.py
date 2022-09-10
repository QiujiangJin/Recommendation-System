import torch
from torch import nn as nn
from toyDataset.loaddata import load100KRatings
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag
from GraphNCF.GCFmodel import GCF, NCF
from torch.utils.data import DataLoader
from GraphNCF.dataPreprosessing import ML1K
from torch.utils.data import random_split
from torch.optim import Adam
from GraphNCF.GCFmodel import SVD
from GraphNCF.GCFmodel import NCF

rt = load100KRatings()
userNum = rt['userId'].max()
itemNum = rt['itemId'].max()

rt['userId'] = rt['userId'] - 1
rt['itemId'] = rt['itemId'] - 1
#
# rtIt = rt['itemId'] + userNum
# uiMat = coo_matrix((rt['rating'],(rt['userId'],rt['itemId'])))
# uiMat_upperPart = coo_matrix((rt['rating'],(rt['userId'],rtIt)))
# uiMat = uiMat.transpose()
# uiMat.resize((itemNum,userNum+itemNum))
# uiMat = uiMat.todense()
# uiMat_t = uiMat.transpose()
# zeros1 = np.zeros((userNum,userNum))
# zeros2 = np.zeros((itemNum,itemNum))
#
# p1 = np.concatenate([zeros1,uiMat],axis=1)
# p2 = np.concatenate([uiMat_t,zeros2],axis=1)
# mat = np.concatenate([p1,p2])
#
# count = (mat > 0)+0
# diagval = np.array(count.sum(axis=0))[0]
# diagval = np.power(diagval,(-1/2))
# D_ = diag(diagval)
#
# L = np.dot(np.dot(D_,mat),D_)
#
para = {
    'epoch':60,
    'lr':0.01,
    'batch_size':2048,
    'train':0.8
}

ds = ML1K(rt)
trainLen = int(para['train']*len(ds))
train,test = random_split(ds,[trainLen,len(ds)-trainLen])
dl = DataLoader(train,batch_size=para['batch_size'],shuffle=True,pin_memory=True)

model = GCF(userNum, itemNum, rt, 80, layers=[80,80,]).cuda()
# model = SVD(userNum,itemNum,50).cuda()
# model = NCF(userNum,itemNum,64,layers=[128,64,32,16,8]).cuda()
optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
# lossfn = nn.MSELoss()
lossfn = nn.NLLLoss()

for i in range(para['epoch']):

    for id,batch in enumerate(dl):
        print('epoch:',i,' batch:',id)
        optim.zero_grad()
        batch_preds = model(batch[0].cuda(), batch[1].cuda())
        batch_label = (batch[2]-1).cuda()
        loss = lossfn(batch_preds, batch_label)
        loss.backward()
        optim.step()
        print(f'mse loss = {loss.detach().cpu().item():.4f}')


        testdl = DataLoader(test,batch_size=len(test),)
        num_correct = 0
        for data in testdl:
            batch_preds = model(data[0].cuda(),data[1].cuda())
            batch_preds = batch_preds.max(1).indices
            batch_label = (data[2]-1).cuda()
            batch_match = batch_preds.eq(batch_label)
            num_correct += batch_match.sum().cpu().item()

        print(f'test acc = {(num_correct / len(test)):.4f}') # test acc

