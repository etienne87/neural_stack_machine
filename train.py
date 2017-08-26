#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:37:17 2017

@author: etienneperot
"""

import numpy as np
from sklearn import datasets

from layers import temporal_affine_forward, temporal_affine_backward, temporal_softmax_loss
from layers import rnn_forward, lstm_forward, rnn_backward, lstm_backward
from rnn_solver import RnnSolver


#MODEL
class Rnn(object):
    def __init__(self, input_dim, hidden_dim, target_size):
        self.params = {}
        
        """
        cell_type (input_dim:8, seq_len:8, hidden_dim:100): 
            - 'rnn': best score: 92%
            - 'lstm': ?
        
        """
        self.cell_type = 'lstm'
        dim_mul = {'lstm': 4, 'rnn': 1}[self.cell_type]
        self.params['Wx'] = np.random.randn(input_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(input_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        
    
        #self.params['Wh'] = np.eye(hidden_dim) #for irnn
        self.params['b'] = np.zeros(dim_mul * hidden_dim)
        self.params['Wt'] = np.random.randn(hidden_dim, target_size)
        self.params['Wt'] /= np.sqrt(target_size)
        self.params['bt'] = np.zeros(target_size)
        
    def run(self, x, h0=None):
        #Run through x sequence
        Wx = self.params['Wx']
        Wh = self.params['Wh']
        b = self.params['b']
        Wt = self.params['Wt']
        bt = self.params['bt']
        N,T,D = x.shape
        
        if h0 is None:
            h0 = np.zeros((N,Wh.shape[0]))
            
        if(self.cell_type == 'rnn'):
            h, h_cache = rnn_forward(x, h0, Wx, Wh, b)
        elif(self.cell_type == 'lstm'):
            h, h_cache = lstm_forward(x, h0, Wx, Wh, b)
            
        t, t_cache = temporal_affine_forward(h, Wt, bt)
        
        return t, [h_cache, t_cache]
    
    def backward(self, dout, cache):
        return Exception("Not implemented")
        
        
    def loss(self, x, y, h0=None, mask=None):
        loss, grads = 0.0, {}
        t, cache = self.run(x,h0)
        h_cache, t_cache = cache
        
        N,T,D = t.shape
        yt = y.reshape((-1,1)).repeat(T,1)
        if mask is None:
            #mask = np.ones((N,T))
            mask = np.arange(0,1,1./T).reshape((1,-1))  
            mask = mask.repeat(N,0)
         
        loss, dloss = temporal_softmax_loss(t, yt, mask)

        dh, grads['Wt'], grads['bt'] = temporal_affine_backward(dloss, t_cache)
        
        if(self.cell_type == 'rnn'):
            dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh, h_cache)
        elif(self.cell_type == 'lstm'):
            dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dh, h_cache)
            
    
        return loss, grads
    
    
# Using Sklearn MNIST dataset.
digits = datasets.load_digits()
X = digits.images
Y = digits.target
X = X.reshape(-1,16,4)

data = (X,Y)
model = Rnn(input_dim=X.shape[2], hidden_dim=256, target_size=10)
solver = RnnSolver(model, data,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=900, batch_size=100,
                  print_every=10)
solver.train()