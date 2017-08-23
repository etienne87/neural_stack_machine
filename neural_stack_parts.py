# -*- coding: utf-8 -*-
'''
\brief: This part tries to break down the neural stack with layers

'''
import numpy as np

def r_t_forward(Vt,st):
    r = np.zeros(stack_width)
    cum_sum = np.cumsum(st)
    uncov = cum_sum[-1]-cum_sum
    w = np.minimum(st, np.maximum(0, 1 - uncov))
    r = w.dot(Vt)
    return r

def s_t_forward(s_prev,ut,dt):
    t = s_prev.shape[0]
    if t > 0:
        cum_sum = np.cumsum(s_prev)
        uncov = cum_sum[-1]-cum_sum
        s_prime = np.maximum(0, s_prev - np.maximum(0, ut -  uncov))
    else:
        s_prime = s_prev
        
    s_next = np.concatenate((s_prime, np.array(dt,ndmin=1)),axis=0)
    return s_next

def pop_forward(s_prev,V_prev):
    ids = np.where(s_prev != 0)
    s_next = s_prev[ids]
    V_next = V_prev[ids]
    return s_next, V_next

#now there is a need for backward of read, push & pop

    
def pushAndPop(V_t,v_t,d_t,u_t,s_t):  
    #==============#
    # push  
    #==============#
    s_next = s_t_forward(s_t,u_t,d_t)
    V_t = np.concatenate((V_t, v_t.reshape(1,-1)))
    #==============#
    # read
    #==============#
    r_t = r_t_forward(V_t,s_next)
    #==============#
    # pop
    #==============#
    s_t, V_t = pop_forward(s_next,V_t)
    return r_t, s_t, V_t

stack_width = 3
copy_length = 5

v_0 = np.zeros(stack_width)
v_0[0] = 1
v_1 = np.zeros(stack_width)
v_1[1] = 1
v_2 = np.zeros(stack_width)
v_2[2] = 1

st = np.zeros(0)
Vt = np.zeros((0,stack_width))
rt1, st, Vt = pushAndPop(Vt,v_0,1.0,0.0,st)
rt1, st, Vt = pushAndPop(Vt,v_0,1.0,0.0,st)
rt1, st, Vt = pushAndPop(Vt,v_0,0.0,1.0,st)
print('current stack size after 1 push + 2 pop :', Vt.shape)

st = np.zeros(0)
Vt = np.zeros((0,stack_width))
rt1, st, Vt = pushAndPop(Vt,v_0,0.8,0.0,st)
rt2, st, Vt = pushAndPop(Vt,v_1,0.5,0.1,st)
rt3, st, Vt = pushAndPop(Vt,v_2,0.9,0.9,st)


print(rt1, 0.8 * v_0)
print(rt2, (0.5 * v_0) + (0.5 * v_1) )
print(rt3, (0.9 * v_2) + (0 * v_1) + (0.1 * v_0) )