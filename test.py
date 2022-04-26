# -*- coding: utf-8 -*-
import numpy as np
import time 

def test(network, optimizer, test_loader, epoch, t0, acc_list, log_interval):
  cnt=[0,0]
  for batch_idx in range(len(test_loader)):
    tensor=test_loader[batch_idx]
    data,target=np.array(tensor[0]),np.array(tensor[1])
    result=network.forward_test(data)

    cnt[0]+=len(data) # Number of datas
    for i in range(len(data)):
      if target[i]==result[i]:
        cnt[1]+=1
    acc=cnt[1]/cnt[0]
    acc_list.append(acc)
    if (batch_idx+1)%log_interval==0:
      print("Test epoch :",epoch,"[",(batch_idx+1)*len(data),"/",len(test_loader)*len(data),"]"\
            ,", Accuracy :",acc,", Time : ",time.time()-t0)
  return (int)(acc*1000)