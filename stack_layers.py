#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:57:35 2017

@author: etienneperot

\brief: Stack Layers gradient checked out (easier for dev)
"""

import numpy as np

"""
Read from stack 
----------------
1. for each row
    a. dirt: sum row+1 ::end
    b.if "dirt" > 1, no read 
    c.if "dirt" <= 1, read by min(1-dirt, st)
2. rt = weighted sum between content & st
"""
def rt_forward(Vt,st):
    r = np.zeros(Vt.shape[1])
    cum_sum = np.cumsum(st,axis=0)
    uncov = 1 - (cum_sum[-1]-cum_sum)
    uncov_plus = np.maximum(0, uncov)
    w = np.minimum(st,uncov_plus)
    r = w.dot(Vt)
    cache = (Vt,st,w,uncov,uncov_plus)
    return r, cache

def rt_backward(dout, cache):
    Vt, st, w, uncov, uncov_plus = cache
    w = w.reshape(1,-1)
    dout = dout.reshape(1,-1)  
    dVt = w.T.dot(dout) 
    dw = Vt.dot(dout.T).reshape(-1)
    dst1 = np.where(st < uncov_plus, dw, 0)
    dmin = np.where(st < uncov_plus, 0, dw)
    dmax = np.where(uncov > 0, dmin, 0) 
    dst2 = np.zeros_like(dmax)
    dst2[1:] = -np.cumsum(dmax[:-1],axis=0)
    dst = dst1 + dst2
    return dVt, dst

"""
Push/Pop from stack
--------------------
1. ut : how much to "dig out" from stack
2. dt : strength of push for new content
"""
def st_forward(s_prev,ut,dt):
    ut = np.atleast_1d(ut)
    dt = np.atleast_1d(dt)
    t = s_prev.shape[0]
    if t > 0:    
        cum_sum = np.cumsum(s_prev)
        dirt = cum_sum[-1]-cum_sum
        uncov = ut - dirt
        s_prime = np.maximum(0, s_prev - np.maximum(0, uncov))
    else:
        s_prime = s_prev 
        uncov = 0
    s_next = np.concatenate((s_prime, dt),axis=0)
    cache = (s_prev,ut,dt,s_prime, uncov)
    return s_next, cache

def st_backward(dout,cache):
    s_prev,ut,dt,s_prime, uncov = cache
    t = s_prev.shape[0]
    ds_prime, ddt = dout[:-1], dout[-1]
    if t>0:
        uncov_plus = np.maximum(0,uncov)
        dmax1 = np.where(s_prev > uncov_plus,ds_prime,0)
        dmax2 = -np.where(uncov > 0,dmax1,0)
        dut = dmax2.sum(axis=0, keepdims=True)
        tmp = np.zeros_like(dmax2)
        tmp[1:] = -np.cumsum(dmax2[:-1],axis=0)
        ds_prev = dmax1 + tmp
    else:
        ds_prev = ds_prime
        dut = 0
    return ds_prev, dut, ddt

"""
Neural Stack
-------------  
1. st_forward, 
2. allocate new content (V,v)
3. rt_forward
4. deallocate popped content
"""
def neural_stack_forward(Vt,vt,dt,ut,st):
    st1, cache1 = st_forward(st, ut, dt)
    Vt1 = np.concatenate((Vt, vt.reshape(1,-1)))
    rt, cache2 = rt_forward(Vt1,st1)  
    ids = np.where(st1 != 0)
    cache3 = (ids,st1.shape,Vt1.shape)
    st2 = st1[ids]
    Vt2 = Vt1[ids] 
    cache = [cache1,cache2,cache3]    
    return rt, st2, Vt2, cache

def neural_stack_backward(drt,dst2,dVt2, cache):
    cache1,cache2,cache3 = cache  
    ids, shape1, shape2 = cache3
    dst1 = np.zeros(shape1)
    dVt1 = np.zeros(shape2)
    dst1[ids] = dst2  
    dVt1[ids] = dVt2
    dVt1_bis, dst1_bis = rt_backward(drt, cache2)
    dst1 = dst1 + dst1_bis
    dVt1 = dVt1 + dVt1_bis
    dVt, dvt = dVt1[:-1], dVt1[-1]
    dst, dut, ddt = st_backward(dst1, cache1)
    return dVt, dvt, ddt, dut, dst
    
    

if __name__ == '__main__':
    import gradient_check
    num_grad = gradient_check.eval_numerical_gradient_array
    rel_error = gradient_check.rel_error
    """ 
    checking grad/output for neural stack 
    """ 
    length = 5
    stack_width = 3
    
    st = np.random.rand(length,)   #making sure no st is greater > 1
    Vt = np.random.rand(length,stack_width)
    dt = np.random.rand(1)
    ut = np.random.rand(1)
    vt = np.random.rand(1,stack_width) 
    
    rt, st, Vt, cache = neural_stack_forward(Vt,vt,dt,ut,st)
    drt = np.random.randn(*rt.shape)
    dst = np.random.randn(*st.shape)
    dVt = np.random.randn(*Vt.shape)
     
    fV = lambda Vt: neural_stack_forward(Vt,vt,dt,ut,st)[0]
    fv = lambda vt: neural_stack_forward(Vt,vt,dt,ut,st)[0] 
    fd = lambda dt: neural_stack_forward(Vt,vt,dt,ut,st)[0]
    fu = lambda ut: neural_stack_forward(Vt,vt,dt,ut,st)[0] 
    fs = lambda st: neural_stack_forward(Vt,vt,dt,ut,st)[0] 
      
    
    dV_num = num_grad(fV, Vt, drt)
    ds_num = num_grad(fs, st, drt, h=1e-5)
    dV,ds = neural_stack_backward(drt, cache) 

    
    st = np.zeros(0)
    Vt = np.zeros((0,stack_width))
    dts = np.array([0.8,0.5,0.9])
    uts = np.array([0.0,0.1,0.9])
    vts = np.eye((stack_width))[:3]
    rts = []
    for i in range(3):
        rt, st, Vt, cache = neural_stack_forward(Vt,vts[i],dts[i],uts[i],st)
        rts.append(rt)
        
    print('output error: ')
    print(rts[0], 0.8 * vts[0], "err ", rel_error(rts[0], 0.8 * vts[0]))
    print(rts[1], (0.5 * vts[0]) + (0.5 * vts[1]), "err ", rel_error(rts[1], (0.5 * vts[0]) + (0.5 * vts[1])))
    print(rts[2], (0.9 * vts[2]) + (0 * vts[2]) + (0.1 * vts[0]), "err ", rel_error(rts[2], (0.9 * vts[2]) + (0 * vts[2]) + (0.1 * vts[0])))
        
        
    
    """ 
    checking gradient for read 
    """ 
#==============================================================================
#     length = 5
#     stack_width = 3
#     st = np.random.rand(length,)   #making sure no st is greater > 1
#     Vt = np.random.rand(length,stack_width)
#     next_r, cache = rt_forward(Vt,st)
#     dnext_r = np.random.randn(*next_r.shape)
#     
#     fV = lambda Vt: rt_forward(Vt,st)[0]
#     fs = lambda st: rt_forward(Vt,st)[0] 
#     
#     dV_num = num_grad(fV, Vt, dnext_r)
#     ds_num = num_grad(fs, st, dnext_r, h=1e-5)
#     dV,ds = rt_backward(dnext_r, cache) 
#     print(dV_num.shape, dV.shape)
#     print(ds_num.shape, ds.shape)
#     print( 'read stack V error: ', rel_error(dV_num, dV))
#     print( 'read stack s error: ', rel_error(ds_num, ds)) 
#==============================================================================
    """ 
    checking gradient for push/pop
    """ 
#==============================================================================
#     length = 5
#     s_prev = np.random.randn(length,) 
#     ut = np.random.randn(1)
#     dt = np.random.randn(1)
#     s_next, cache = st_forward(s_prev,ut,dt)
#     ds_next = np.random.randn(*s_next.shape)
#     
#     fV = lambda s_prev: st_forward(s_prev,ut,dt)[0]
#     fu = lambda ut: st_forward(s_prev,ut,dt)[0]
#     fd = lambda dt: st_forward(s_prev,ut,dt)[0] 
#     
#     ds_prev_num = num_grad(fV, s_prev, ds_next)
#     du_num = num_grad(fu, ut, ds_next, h=1e-5)
#     dd_num = num_grad(fd, dt, ds_next, h=1e-5)
#     
#     ds_prev, dut, ddt = st_backward(ds_next, cache) 
#     print(ds_prev.shape, ds_prev.shape)
#     print(du_num.shape, dut.shape)
#     print(dd_num.shape, ddt.shape)
#     
#     print(dut, du_num)
#     print( 'read stack s_prev error: ', rel_error(ds_prev_num, ds_prev))
#     print( 'read stack u error: ', rel_error(du_num, dut))
#     print( 'read stack d error: ', rel_error(dd_num, ddt)) 
#==============================================================================
