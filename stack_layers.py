#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""
Actual stack layers
- read
- push
- pop
"""
def read_stack_forward(Vt,st):
    r = np.zeros(Vt.shape[1])
    cum_sum = np.cumsum(st,axis=0)
    uncov = 1 - (cum_sum[-1]-cum_sum)
    uncov_plus = np.maximum(0, uncov)
    w = np.minimum(st,uncov_plus)
    r = w.dot(Vt)
    cache = (Vt,st,w,uncov,uncov_plus)
    return r, cache

def read_stack_backward(dout, cache):
    Vt, st, w, uncov, uncov_plus = cache
    w = w.reshape(1,-1)
    dout = dout.reshape(1,-1)  
    dVt = w.T.dot(dout) 
    dw = Vt.dot(dout.T).reshape(-1)
    dst1 = np.where( st < uncov_plus, dw, 0)
    dmin = np.where(st < uncov_plus, 0, dw)
    dmax = np.where(uncov > 0, dmin, 0)
    dst2 = np.zeros_like(dmax)
    dst2[1:] = -np.cumsum(dmax[:-1],axis=0)
    dst = dst1 + dst2
    return dVt, dst

def push_forward(s_prev,ut,dt):
    t = s_prev.shape[0]
    if t > 0:
        cum_sum = np.cumsum(s_prev)
        dirt = cum_sum[-1]-cum_sum
        uncov = ut - dirt
        s_prime = np.maximum(0, s_prev - np.maximum(0, uncov))
    else:
        s_prime = s_prev
        
    s_next = concat_forward(s_prime,dt)
    cache = (s_prev,ut,dt,s_prime, uncov)
    return s_next, cache

    
if __name__ == '__main__':
    import gradient_check
    num_grad = gradient_check.eval_numerical_gradient_array
    rel_error = gradient_check.rel_error

    """ 
    checking gradient for read 
    """ 
    length = 5
    stack_width = 3
    st = np.random.rand(length,)   #making sure no st is greater > 1
    Vt = np.random.rand(length,stack_width)
    next_r, cache = read_stack_forward(Vt,st)
    dnext_r = np.random.randn(*next_r.shape)
    
    fV = lambda Vt: read_stack_forward(Vt,st)[0]
    fs = lambda st: read_stack_forward(Vt,st)[0] 
    
    dV_num = num_grad(fV, Vt, dnext_r)
    ds_num = num_grad(fs, st, dnext_r, h=1e-5)
    dV,ds = read_stack_backward(dnext_r, cache) 
    #print(dV_num.shape, dV.shape)
    #print(ds_num.shape, ds.shape)
 
    print(ds)
    print(ds_num)
    print( 'read stack V error: ', rel_error(dV_num, dV))
    print( 'read stack s error: ', rel_error(ds_num, ds)) 
    """ 
    checking gradient for push
    """ 
    
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
    """
    checking gradient for sublast
    """
    #N = 3
    #D = 2
    #x = np.random.randn(N, D)
    #next_x = sub_last_forward(x)
    #dnext_x = np.random.randn(*next_x.shape) 
    #fx = lambda x: sub_last_forward(x) 
    #dx_num = num_grad(fx, x, dnext_x, h=1e-10)
    #dx = sub_last_backward(dnext_x) 
    #print(dx_num.shape, dx.shape)
    #print( 'sublast dx error: ', rel_error(dx_num, dx))
    
    