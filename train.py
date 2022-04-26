# -*- coding: utf-8 -*-
import numpy as np
import time

def train(network, optimizer, train_loader, epoch, t0, loss_list, acc_list, log_interval):
  cnt=[0,0]
  for batch_idx in range(len(train_loader)):
    tensor=train_loader[batch_idx]
    data,target=np.array(tensor[0]),np.array(tensor[1])
    loss,result=network.forward_train(data,target)
    network.backward()
    optimizer.update()

    cnt[0]+=len(data) # Number of datas
    for i in range(len(data)):
      if target[i]==result[i]:
        cnt[1]+=1
    acc=cnt[1]/cnt[0]
    acc_list.append(acc)
    loss_list.append(loss)
    if (batch_idx+1)%log_interval==0:
      print("Train epoch :",epoch,"[",(batch_idx+1)*len(data),"/",len(train_loader)*len(data),"]"\
            ,", Accuracy :",acc,", Time : ",time.time()-t0)
