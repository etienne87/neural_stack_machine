#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:46:14 2017

@author: etienneperot
"""
"""
Created on Tue Aug 22 16:57:35 2017

@author: etienneperot

\brief: Stack Layers gradient checked out (easier for dev)
"""

import numpy as np

"""
Trivial layers first
- cumsum
- concat
- select
"""
#Sum from i to N
def cumsum_forward(x):
    out = np.cumsum(x,axis=0)
    return out

def cumsum_backward(dout):
    dx = np.cumsum(dout[::-1],axis=0)[::-1]
    return dx

#Sum from i+1 to N
def rev_cumsum_forward(x):
    cs = np.cumsum(x,axis=0)
    return (cs[-1] - cs)

def rev_cumsum_backward(dout):
    dst = np.zeros_like(dout)
    dst[1:] = np.cumsum(dout[:-1],axis=0)
    return dst

#Join 2 vectors
def concat_forward(a,b):
     c = np.concatenate((a, b.reshape(1,-1)))
     return c
 
def concat_backward(dout):
    return dout[:-1], dout[-1]

#Select subset of rows
def select_forward(x, ids):
    cache = (x.shape, ids)
    return x[ids], cache

def select_backward(dout, cache):
    shape, ids = cache
    dx = np.zeros(shape)
    dx[ids] = dout
    return dx


if __name__ == '__main__':
    import gradient_check
    num_grad = gradient_check.eval_numerical_gradient_array
    rel_error = gradient_check.rel_error
    """ 
    checking gradient for cumsum
    """ 
#==============================================================================
#     cs = cumsum_forward
#     bcs = cumsum_backward
#     N = 3
#     D = 3
#     x = np.random.randn(N, D) 
#     next_x = cs(x)
#     dnext_x = np.random.randn(*next_x.shape) 
#     fx = lambda x: cs(x) 
#     dx_num = num_grad(fx, x, dnext_x)
#     dx = bcs(dnext_x) 
#     print(dx_num.shape, dx.shape)
#     print(dx/dx_num)
#     print( 'cumsum dx error: ', rel_error(dx_num, dx))
#==============================================================================
    """ 
    checking gradient for concat
    """ 
    #a = np.random.randn(N, D)
    #b = np.random.randn(1,D)
    #c = concat_forward(a,b)
    #dc = np.random.randn(*c.shape)
    #fa = lambda a: concat_forward(a,b)
    #fb = lambda b: concat_forward(a,b)
    #da_num = num_grad(fa, a, dc)
    #db_num = num_grad(fb, b, dc)
    #da,db = concat_backward(dc) 
    #print(da_num.shape, da.shape)
    #print(db_num.shape, db.shape)
    #print( 'concat da error: ', rel_error(da_num, da))
    #print( 'concat db error: ', rel_error(db_num, db)) 
    """ 
    checking gradient for select
    """ 
    #ids = np.where( x.mean(axis=1) >= 0 )
    #next_x, cache = select_forward(x,ids)
    #dnext_x = np.random.randn(*next_x.shape)
    #fx = lambda x: select_forward(x,ids)[0]
    #dx_num = num_grad(fx, x, dnext_x)
    #dx = select_backward(dnext_x,cache)
    #print(dx_num.shape, dx.shape)
    #print( 'select dx error: ', rel_error(dx_num, dx))
