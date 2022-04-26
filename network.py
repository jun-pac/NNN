# -*- coding: utf-8 -*-
import numpy as np
import pickle
from layer import *

last_layer=[Softmax_Loss, Sum_Loss]

class model:
  def __init__(self, layers):
    # 각 layer들의 class instance를 list로 받음. 
    self.layers=layers

  def param_load(self, file_name):
    with open(file_name, 'rb') as handle:
      state_dict = pickle.load(handle)
    w_cnt=0
    for layer in self.layers:
      if(type(layer)==FC):
        w_cnt+=1
        assert 'W'+str(w_cnt) in state_dict
        assert 'B'+str(w_cnt) in state_dict
        W_load=state_dict.get('W'+str(w_cnt))
        B_load=state_dict.get('B'+str(w_cnt))
        assert W_load.shape==layer.W.shape
        assert B_load.shape==layer.B.shape
        for i in range(len(layer.W)):
          for j in range(len(layer.W[i])):
            layer.W[i,j]=W_load[i,j]
        for i in range(len(layer.B)):
          layer.B[i]=B_load[i]    

  def param_save(self, file_name):
    state_dict={}
    w_cnt=0
    for layer in self.layers:
      if(type(layer)==FC):
        w_cnt+=1
        state_dict['W'+str(w_cnt)]=layer.W
        state_dict['B'+str(w_cnt)]=layer.B
    with open(file_name, 'wb') as handle:
      pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def forward_train(self,X,Y):
    for layer in self.layers:
      if(type(layer) in last_layer):
        #print(type(layer),X)
        result=np.array([vec.argmax() for vec in X])
        loss=layer(X,Y)
      else:
        #print(type(layer),X)
        X=layer(X)
    return [loss.copy(),result]

  def forward_test(self,X):
    for layer in self.layers:
      if(type(layer) in last_layer):
        result=np.array([vec.argmax() for vec in X])
      else:
        X=layer(X)
    return result

  def backward(self):
    for i in range(len(self.layers)):
      layer=self.layers[len(self.layers)-1-i]
      if(type(layer) in last_layer):
        grad=layer.backward()
      else:
        grad=layer.backward(grad)
    return grad


str.encode('utf-8').strip()
"""
# Codes for debug

layers=[FC(3,4),ReLU(),FC(4,4),ReLU(),Softmax_Loss()]
#layers=[FC(3,4),ReLU(), Sum_Loss()]
network=model(layers)
param_ls=[]
for layer in network.layers:
    if(type(layer)==FC):
        param_ls.append([layer.W, layer.grad['W']])
        param_ls.append([layer.B, layer.grad['B']])

X=np.random.rand(2,3)
Y=np.array([0,3])

in_X=X.copy()
in_Y=Y.copy()
loss,result=network.forward_train(in_X,in_Y)
print(loss, result)
network.backward()
print(param_ls)

print("Analytic value")
for i in range(len(param_ls)):
  print(param_ls[i][1])

#========================
# Numerical 값 구하고 비교해보기
epsilon=0.0000001
param_numeric=[]
print("Numerical value")
for weight, grad in param_ls:
  if type(weight[0])==np.ndarray:
    grad_temp=[]
    for i in range(len(weight)):
      grad_temp.append([])
      for j in range(len(weight[i])):
        #print(i,j)
        weight[i][j]+=epsilon
        new_loss,result=network.forward_train(in_X,in_Y)
        weight[i][j]-=epsilon
        grad_temp[i].append((new_loss-loss)/epsilon)
    param_numeric.append(grad_temp)
  else:
    grad_temp=[]
    for i in range(len(weight)):
        weight[i]+=epsilon
        new_loss,result=network.forward_train(in_X,in_Y)
        weight[i]-=epsilon
        grad_temp.append((new_loss-loss)/epsilon)
    param_numeric.append(grad_temp)

for i in range(len(param_numeric)):
    print(param_numeric[i])
"""