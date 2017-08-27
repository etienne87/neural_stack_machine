#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:37:17 2017

@author: etienneperot
"""
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

print('cuda available: ',torch.cuda.is_available())

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size,hidden_size)
        self.logits = nn.Linear(hidden_size,output_size)
        
        
    def forward(self, input, cuda):
        h0 = torch.randn(1, input.size(1), self.hidden_size)
        c0 = torch.randn(1, input.size(1), self.hidden_size)  
    
        if cuda:
            h0,c0 = h0.cuda(), c0.cuda()
            
        hidden = (Variable(h0),
                  Variable(c0))
        
        out, next_hidden = self.lstm1(input, hidden)
        
        #H = out.size(2)
        #out_lin = out.view(-1, H)
        #scores = self.logits(out_lin)
        
        #we take last output 
        out_last = out[-1,:,:]
        scores = self.logits(out_last)
        
        return F.log_softmax(scores)
        
def one_hot(y,max=10):
    t = np.zeros((y.shape[0],max))
    t[y] = 1
    return t
  
def minibatch(X,Y,batchsize):
    N = X_train.shape[0]
    ids = np.arange(N)
    np.random.shuffle(ids)
    x = X_train[ids][:batchsize]
    y = y_train[ids][:batchsize]     
    return x,y

loss_fn = torch.nn.NLLLoss(size_average=False)
    
# Using Sklearn MNIST dataset.
digits = datasets.load_digits()
X = digits.images
Y = digits.target
#Y = one_hot(Y)
X = X.reshape(-1,64,1) #N,T,D
X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.22, random_state=42)

model = RNN(input_size=X.shape[2], hidden_size=256, output_size=10)


cuda = 1 and torch.cuda.is_available()
if cuda:
    model.cuda()

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    # Prepare minibatch
    bx,by = minibatch(X_train, y_train, 100)
    
    #Pytorch expects T,N,D 
    bx = bx.transpose([1,0,2])
    tbx,tby = torch.from_numpy(bx).float(), torch.from_numpy(by).long()
    if cuda:
        tbx, tby = tbx.cuda(), tby.cuda()
        
    x = Variable(tbx)
    y = Variable(tby)
    
    
    
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x,cuda)

    
    # Compute and print loss.
    loss = F.nll_loss(input=y_pred, target=y)
    print(t, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    