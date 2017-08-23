# -*- coding: utf-8 -*-
import numpy as np

stack_width = 3
copy_length = 5

v_0 = np.zeros(stack_width)
v_0[0] = 1
v_1 = np.zeros(stack_width)
v_1[1] = 2
v_2 = np.zeros(stack_width)
v_2[2] = 3


# INIT
V = list() # stack states    (list of matrix of growing num of rows x stack_width)
s = list() # stack strengths (list of array of growing length)
d = list() # push strengths  (list of numbers)
u = list() # pop strengths   (list of numbers)

def r_t(t):
    r = np.zeros(stack_width)
    for i in range(0,t+1):
        inner_sum = sum(s[t][i+1:t+1])
        left_over = max(0, 1 - inner_sum)
        r += min(s[t][i], left_over) * V[t][i]
    return r

def s_t(i,t,u,d):
    if i == t:
        return d[t]
    elif i >= 0 and i < t:
        inner_sum = sum(s[t-1][i+1:t])
        temp = max(0, u[t] - inner_sum)
        return max(0, s[t-1][i] - temp)
    else:
        return Exception("Not implemented")
    
def pushAndPop(v_t,d_t,u_t,t=len(V)):
    
    d.append(d_t)
    u.append(u_t)

    s_next = np.zeros(t+1)
    for i in range(t+1):
        s_next[i] = s_t(i,t,u,d)
    s.append(s_next)
    
    if t == 0:
        V_t = np.reshape(v_t,(1,stack_width))
    else:
        V_t = np.zeros((t+1,stack_width))
        V_t[:t] = V[-1]
        V_t[-1] = v_t
     
    V.append(V_t)
    
    return r_t(t)
  
#print( str(pushAndPop(v_0,0.8,0.0,0)) )
#print( str(pushAndPop(v_1,0.5,0.1,1)) )
#print( str(pushAndPop(v_2,0.9,0.9,2)) )
r1=pushAndPop(v_0,1.0,0.0,0)
r2=pushAndPop(v_1,1.0,0.0,1)
r3=pushAndPop(v_2,0.0,1.0,2)
r4=pushAndPop(v_0,0.0,1.0,3)
r5=pushAndPop(v_1,0.0,1.0,4)
r6=pushAndPop(v_2,0.0,1.0,5)

print(r1)
print(r2)
print(r3)
print(r4)
print(r5)
print(r6)


# Stack is empty again
'''
V = list() # stack states
s = list() # stack strengths 
d = list() # push strengths
u = list() # pop strengths

assert str(pushAndPop(v_0,0.8,0.0,0)) == str((0.8 * v_0))
assert str(pushAndPop(v_1,0.5,0.1,1)) == str((0.5 * v_0) + (0.5 * v_1))
assert str(pushAndPop(v_2,0.9,0.9,2)) == str((0.9 * v_2) + (0 * v_1) + (0.1 * v_0))

'''