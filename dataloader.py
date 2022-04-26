# -*- coding: utf-8 -*-
import numpy as np
from hyper_parameter import *
import matplotlib.pyplot as plt
import pickle

'''
file=data_path+"data_batch_1"
with open(file, 'rb') as fo:
  result=pickle.load(fo,encoding='bytes')
print(result.keys())
print(type(result[b'batch_label']))
print(result[b'batch_label'])
print(type(result[b'labels']))
print(len(result[b'labels']))
print(result[b'labels'][0])

print(type(result[b'data']))
print(len(result[b'data']))
print(result[b'data'][0])

image=result[b'data'][4].reshape((3,32,32))
label=result[b'labels'][4]
print(label)
image=np.transpose(image,(1,2,0))
plt.imshow(image, interpolation='nearest')
plt.show()
'''


class Dataloader:
  def __init__(self,data_directory,is_train="train"):
    self.data_directory=data_directory
    self.datas=[]
    self.labels=[]
    if is_train=="train":
      for i in range(5):
        with open(data_directory+"data_batch_"+str(i+1), 'rb') as fo:
          temp=pickle.load(fo,encoding='bytes')
        for j in range(10000):
          self.datas.append(temp[b'data'][j]/255)
          self.labels.append(temp[b'labels'][j])
    else:
      with open(data_directory+"test_batch", 'rb') as fo:
        temp=pickle.load(fo,encoding='bytes')
      for j in range(10000):
        self.datas.append(temp[b'data'][j]/255)
        self.labels.append(temp[b'labels'][j])

  def __len__(self):
    return len(self.datas)//batch_size

  def __getitem__(self, idx):
    return [self.datas[idx*batch_size:(idx+1)*batch_size],\
            self.labels[idx*batch_size:(idx+1)*batch_size]]