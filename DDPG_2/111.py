from IDM_MOBIL import idm_mobil
import numpy as np
t=3
d_0=10
T=1
v5 = 15
b=5
a3 = 2
a4 = 2
a5=2
x4 = 22
x5= 37
an_af = 0
an_be = 15
p=0.001
ao_af = 0
ao_be = 0
a_th = 0.2
a_max = 6
v4 = 15
v_ex = 15
theta = 4
delta_d = 15
a= 7.04
d= 10.52
c=7.04
clock = 0.1
t0= 0

moxing_1 = idm_mobil(t =t, d_0 = d_0, T=T, v5=v5, b=b, a3=a3, a4=a4, a5=a5, x4=x4, x5=x5, an_af=an_af, an_be=an_be, p=p, ao_af=ao_af,ao_be=ao_be, a_th=a_th,a_max=a_max, v4=v4, v_ex=v_ex,theta=theta,delta_d=delta_d,a=a,d=d,c=c,clock=clock,t0=t0)
A = moxing_1.panduan() # return steer, acce, t0, c
B = list(A)
print(type(B),B)
print(type(A[0]),A[0])
action = np.array([3,4,5])
print(action[0])
D = np.array([A[0],A[1],A[2],A[3],3,4,5])
D = np.array([D[0]])
# action = np.array([action,D])
# print(type(action),action)
print(type(D),D)