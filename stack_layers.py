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

"""
Read from stack: 
    
    for each row
        dirt: sum row+1 ::end
        if "dirt" > 1, no read 
        if "dirt" <= 1, read by min(1-dirt, st)
"""
def read_forward(Vt,st):
    r = np.zeros(Vt.shape[1])
    cum_sum = np.cumsum(st,axis=0)
    uncov = 1 - (cum_sum[-1]-cum_sum)
    uncov_plus = np.maximum(0, uncov)
    w = np.minimum(st,uncov_plus)
    r = w.dot(Vt)
    cache = (Vt,st,w,uncov,uncov_plus)
    return r, cache

def read_backward(dout, cache):
    Vt, st, w, uncov, uncov_plus = cache
    w = w.reshape(1,-1)
    dout = dout.reshape(1,-1)  
    dVt = w.T.dot(dout) 
    dw = Vt.dot(dout.T).reshape(-1)
    dst1 = np.where(st < uncov_plus, dw, 0)
    dmin = np.where(st < uncov_plus, 0, dw)
    dmax = np.where(uncov > 0, dmin, 0)
    dst2 = -rev_cumsum_backward(dmax)
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
    s_next = np.concatenate((s_prime, dt),axis=0)
    cache = (s_prev,ut,dt,s_prime, uncov)
    return s_next, cache

def push_backward(dout,cache):
    s_prev,ut,dt,s_prime, uncov = cache
    t = s_prev.shape[0]
    ds_prime, ddt = concat_backward(dout)
    if t>0:
        uncov_plus = np.maximum(0,uncov)
        dmax1 = np.where(s_prev > uncov_plus,ds_prime,0)
        dmax2 = -np.where(uncov > 0,dmax1,0)
        dut = dmax2.sum(axis=0, keepdims=True)
        ds_prev = dmax1-rev_cumsum_backward(dmax2)
    else:
        ds_prev = ds_prime
        dut = 0
    return ds_prev, dut, ddt

def pop_forward(s_prev,V_prev):
    ids = np.where(s_prev != 0)
    s_next = s_prev[ids]
    V_next = V_prev[ids]
    cache = (s_prev.shape, ids)
    return s_next, V_next, cache

def pop_backward(dout, cache):
    prev_len, ids = cache
    ds_next, dV_next = dout
    
    ds_prev = np.zeros(prev_len,)
    ds_prev[ids] = ds_next
    
    size = dV_next.shape[1]
    dV_prev = np.zeros((prev_len,size))
    dV_prev[ids] = dV_next
    
    return ds_prev, dV_prev

    
if __name__ == '__main__':
    import gradient_check
    num_grad = gradient_check.eval_numerical_gradient_array
    rel_error = gradient_check.rel_error

    """ 
    checking gradient for read 
    """ 
#==============================================================================
#     length = 5
#     stack_width = 3
#     st = np.random.rand(length,)   #making sure no st is greater > 1
#     Vt = np.random.rand(length,stack_width)
#     next_r, cache = read_forward(Vt,st)
#     dnext_r = np.random.randn(*next_r.shape)
#     
#     fV = lambda Vt: read_forward(Vt,st)[0]
#     fs = lambda st: read_forward(Vt,st)[0] 
#     
#     dV_num = num_grad(fV, Vt, dnext_r)
#     ds_num = num_grad(fs, st, dnext_r, h=1e-5)
#     dV,ds = read_backward(dnext_r, cache) 
#     print(dV_num.shape, dV.shape)
#     print(ds_num.shape, ds.shape)
#     print( 'read stack V error: ', rel_error(dV_num, dV))
#     print( 'read stack s error: ', rel_error(ds_num, ds)) 
#==============================================================================
    """ 
    checking gradient for push
    """ 
    length = 5
    s_prev = np.random.rand(length,) 
    ut = np.random.rand(1)
    dt = np.random.rand(1)
    s_next, cache = push_forward(s_prev,ut,dt)
    ds_next = np.random.randn(*s_next.shape)
    
    fV = lambda s_prev: push_forward(s_prev,ut,dt)[0]
    fu = lambda ut: push_forward(s_prev,ut,dt)[0]
    fd = lambda dt: push_forward(s_prev,ut,dt)[0] 
    
    ds_prev_num = num_grad(fV, s_prev, ds_next)
    du_num = num_grad(fu, ut, ds_next, h=1e-5)
    dd_num = num_grad(fd, dt, ds_next, h=1e-5)
    
    ds_prev, dut, ddt = push_backward(ds_next, cache) 
    print(ds_prev.shape, ds_prev.shape)
    print(du_num.shape, dut.shape)
    print(dd_num.shape, ddt.shape)
    
    print(dut, du_num)
    print( 'read stack s_prev error: ', rel_error(ds_prev_num, ds_prev))
    print( 'read stack u error: ', rel_error(du_num, dut))
    print( 'read stack d error: ', rel_error(dd_num, ddt)) 
    
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