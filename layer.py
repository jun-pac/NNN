# -*- coding: utf-8 -*-
# 한글로 주석을 달기 위한 코드
import numpy as np

class FC:
  def __init__(self,n,m):
    self.n=n
    self.m=m
    #self.W=np.zeros((n,m)) # for Explicit Intialization version
    self.W=np.random.randn(n,m)*np.sqrt(2/n)
    #self.B=np.zeros((m)) # 그냥 0으로 초기화해도 된다.
    self.B=np.random.randn(m)
    self.grad={'W':np.zeros((n,m)),'B':np.zeros((m))}

  def forward(self,X):
    assert len(X.shape)==2
    assert X.shape[1]==self.n
    self.bs=len(X)
    self.X=X
    return np.matmul(X,self.W)+self.B

  def __call__(self,X):
    return self.forward(X)

  def backward(self,dLdy):
    assert len(dLdy.shape)==2
    assert dLdy.shape[0]==self.bs
    assert dLdy.shape[1]==self.m
    mul=np.matmul(self.X.T,dLdy)
    for i in range(self.n):
      for j in range(self.m):
        self.grad['W'][i,j]=mul[i,j]
    self.grad['B'].fill(0)
    for i in range(dLdy.shape[0]):
      self.grad['B']+=dLdy[i]
    return np.matmul(dLdy,self.W.T)


class ReLU:
  def __init__(self): # size입력할 필요 없음
    pass

  def forward(self,X):
    self.mask=[X<=0]
    X[self.mask]=0
    return X

  def __call__(self,X):
    return self.forward(X)

  def backward(self,dLdy):
    dLdy[self.mask]=0
    return dLdy


class Softmax_Loss:
  def __init__(self): # size입력할 필요 없음
    pass

  def forward_train(self,X,ans):
    # ans is vector of indices! Not "One-Hot-Encoding"
    assert len(X.shape)==2
    assert len(ans.shape)==1
    assert len(X)==len(ans)
    self.bs=X.shape[0]
    self.n=X.shape[1]
    self.X=X
    self.ans=ans
    self.sum=np.array([np.sum(np.exp(vec)) for vec in X])
    self.e_ans=np.exp([X[i,ans[i]] for i in range(self.bs)])
    return -np.mean(np.log(self.e_ans/self.sum))

  def __call__(self,X,ans):
    return self.forward_train(X,ans)

  def backward(self):
    # 마지막 Layer이므로 입력되는 gradient가 없다.
    result=np.zeros((self.bs,self.n))
    for i in range(self.bs):
      result[i].fill(1/self.bs/self.sum[i])
    for i in range(self.bs):
      result[i,self.ans[i]]-=1/self.bs/self.e_ans[i]
    result=result*np.exp(self.X)
    return result


class Sum_Loss:
  def __init__(self): # size입력할 필요 없음
    pass

  def forward_train(self,X,ans):
    self.bs=X.shape[0]
    self.n=X.shape[1]
    return np.sum(X)

  def __call__(self,X,ans):
    return self.forward_train(X,ans)

  def backward(self):
    # 마지막 Layer이므로 입력되는 gradient가 없다.
    result=np.ones((self.bs,self.n))
    return result

str.encode('utf-8').strip()
