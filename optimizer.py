# -*- coding: utf-8 -*-
from http.client import ImproperConnectionState
import numpy as np
from hyper_parameter import *
from layer import *

class Vanilla_GD:
  def __init__(self, network):
    self.param_ls=[]
    for layer in network.layers:
      if(type(layer)==FC):
        self.param_ls.append([layer.W, layer.grad['W']])
        self.param_ls.append([layer.B, layer.grad['B']])

  def update(self):
    for param in self.param_ls:
      param[0]-=param[1]*learning_rate


class Vanilla_GD_L2:
  def __init__(self, network, L2_hyp):
    self.param_ls=[]
    self.L2_hyp=L2_hyp
    for layer in network.layers:
      if(type(layer)==FC):
        self.param_ls.append([layer.W, layer.grad['W']])
        self.param_ls.append([layer.B, layer.grad['B']])

  def update(self):
    for param in self.param_ls:
      param[0]-=param[1]*learning_rate+param[0]*self.L2_hyp

    
class Momentum:
  def __init__(self, network, rho):
    self.param_ls=[]
    self.rho=rho
    for layer in network.layers:
      if(type(layer)==FC):
        self.param_ls.append([layer.W, layer.grad['W'], np.zeros(layer.W.shape)])
        self.param_ls.append([layer.B, layer.grad['B'], np.zeros(layer.B.shape)])

  def update(self):
    for param in self.param_ls:
      param[2]*=self.rho
      param[2]-=param[1]*learning_rate
      param[0]+=param[2]


class Momentum_L2:
  def __init__(self, network, rho, L2_hyp):
    self.param_ls=[]
    self.rho=rho
    self.L2_hyp=L2_hyp
    for layer in network.layers:
      if(type(layer)==FC):
        self.param_ls.append([layer.W, layer.grad['W'], np.zeros(layer.W.shape)])
        self.param_ls.append([layer.B, layer.grad['B'], np.zeros(layer.B.shape)])

  def update(self):
    for param in self.param_ls:
      param[2]*=self.rho
      param[2]-=param[1]*learning_rate+param[0]*self.L2_hyp
      param[0]+=param[2]


'''
# Codes for debug


# learning_rate=0.05

layers=[FC(10,20),ReLU(),FC(20,10),ReLU(),Softmax_Loss()]
network=model(layers)
X=np.random.rand(4,10)
Y=np.array([0,1,2,3])

optim=Momentum(network,0.99)
loss_list=[]
for i in range(1000):
  loss,result=network.forward_train(X,Y)
  network.backward()
  optim.update()
  loss_list.append(loss)

plt.plot(loss_list)
plt.show()
'''