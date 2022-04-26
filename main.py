# NNN (oiocha)

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from hyper_parameter import *
from layer import *
from network import *
from optimizer import *
from test import *
from train import *
from dataloader import *

train_loader=Dataloader(data_path,"train")
test_loader=Dataloader(data_path,"test")
layers=[FC(3*32*32,200),ReLU(),FC(200,200),ReLU(),FC(200,200),ReLU(),FC(200,10),Softmax_Loss()]
network=model(layers)
optimizer=Momentum_L2(network,0.95,0.00001)
acc_list=[]
loss_list=[]
test_acc_list=[]
t0=time.time()

#network.param_load(data_path+'Wide_20_550.pickle')
network.param_load(data_path+'L2_Deep_10_499.pickle')

for i in range(1):
  train(network, optimizer, train_loader, i, t0, loss_list, acc_list, log_interval)
  res=test(network, optimizer, test_loader, i, t0, test_acc_list, log_interval)
  #network.param_save(data_path+'L2_Wide_'+str(i+1)+'_'+str(res)+'.pickle')

#network.param_save(data_path+'L2_Deep_20_'+str(res)+'.pickle')

plt.plot(loss_list)