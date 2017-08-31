# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:39:35 2017

@author: Hajime
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections, numpy
import pandas as pd
import copy
import os
import csv
import sys
import scipy.optimize as opt
import scipy.stats as stats
from scipy import interpolate
import itertools
from sklearn.linear_model import LinearRegression

from theano import tensor as tt
from theano import function

def SumByGroup(groupid,x,shrink=0):
    nobs = groupid.size
    id_list = np.unique(groupid)
    id_num = id_list.size
    if x.ndim==1:
        x = np.array([x]).T
    nchar = x.shape[1]
    if shrink==0:    
        sums = np.zeros([nobs,nchar])
        for i in range(id_num):
            a = np.sum(x[groupid==id_list[i],:],axis=0)
            sums[groupid==id_list[i],:] =a
        return sums
    if shrink==1:
        sums = np.zeros([id_num,nchar])
        for i in range(id_num):
            a = np.sum(x[groupid==id_list[i],:],axis=0)
            sums[i] = a
        return sums

#Dynamic Demand Simulation
#Assuming one time purchase, therefore the state takes delta only, not f.
#Assuming homogeneous consumers.
#Assuming the number of prducts don't change.
#%%Consumer Problem

#make grids
dgrid_n=20
dgrid_max=10.
dgrid_min=-10.
 
dgrid=np.linspace(dgrid_min,dgrid_max,num=dgrid_n)

N_draw_E=7 # draw for Expectation on delta'
#---temp for coding---
N_prod=3 
N_period=10 
N_obs=N_prod*N_period
N_prod_t=np.ones(N_period)*N_prod #number of produtcts in each t        
t = np.repeat(np.arange(N_period),N_prod)
prodid=np.tile(np.arange(N_prod),N_period)

#%%
alpha_j = np.array([5.,3.,10.])
alpha_p = -5.
P = np.random.randn(N_obs)
lam = np.random.randn(N_obs)
f = alpha_j[prodid]  +lam

deltait = np.log( SumByGroup(t,np.exp(f+alpha_p*P),shrink=1) ).flatten()
deltait_n = len(deltait)
V = np.zeros(dgrid_n)

beta = .8        

tol_value_iter=.1

delta_draw=np.random.randn(N_draw_E)
#delta_draw=Halton_draw.halton_randn(1,N_draw_E)[:,0]       


#%%



#%% Test Value
V_init = np.zeros(dgrid_n)
V_old = V_init
norm = tol_value_iter + 1000.
#%%
#Calculate AR1 process
lr = LinearRegression(fit_intercept=True)
lr.fit(deltait[:-1],deltait[1:])
gamma0 = lr.intercept_
gamma1 = lr.coef_        
res = deltait[1:] - lr.predict(deltait[:-1])
var_nu = np.var(res)

#%%
def gen_delta_next(delta_seq): #input a sequence of delta [delta1 delta2 ...], return [delta's for delta1, delta's for delta2,...]
    delta_n=delta_seq.size
    delta_next_stuck=(gamma0+gamma1*delta_seq).repeat(N_draw_E)+np.tile(delta_draw*np.sqrt(var_nu),delta_n) #as delta_n*N_draw_E vector.
    delta_next=delta_next_stuck.reshape(delta_n,N_draw_E) # as delta_n by N_draw_E matrix.        
    return delta_next,delta_next_stuck

def Calc_EV(V,delta_seq):
    delta_next,_ = gen_delta_next(delta_seq)
    V_next = interpolate.griddata(dgrid, V, delta_next  )
    ev = np.mean(V_next,axis=1).flatten() #return from delta_seq to EV for each delta
    return ev

#%%
def loop_val(deltait,V_init):
    V_old = V_init
    
    EV_on_grid = 

#%%
def loop_ar1_value_delta(V_init,deltait_init,Fj): #can loop until convergence, can loop for a couple of times    
    deltait_old = deltait_init
    V_old = V_init
    deltait_n = deltait_init.shape[0]
    #AR(1) regression
    #value function iteration
    V_new = value_iter(V_old)        
    #delta iteration
    deltait_new = deltait_iter(V_new,deltait_old,Fj)
    return V_new, deltait_new
        

def value_iter( V_init):
    V_old = V_init
    f_stuck = grids_list[:,0]
    delta_stuck = grids_list[:,1]
    norm = tol_value_iter + 1000.
    while norm>tol_value_iter:            
        V_new_stuck = np.log( np.exp( f_stuck + beta * EV(V_old, fgrid, dgrid) ) + np.exp(delta_stuck) )
        V_new = V_new_stuck.reshape(fgrid_n,dgrid_n)
        norm = np.max(np.abs(V_new - V_old))
        V_old = V_new
    return V_new
    


#%%Test EV
V = V_init
f_seq=f_stuck
delta_seq=delta_stuck

f_n = f_seq.size
delta_n = delta_seq.size
delta_stuck = np.tile(delta_seq,f_n)
_,delta_next_stuck = gen_delta_next(delta_stuck)
f_stuck_EV = np.repeat(f_seq,delta_n*N_draw_E)
V_stuck_EV = V.reshape(V.size,order='F') 
V_next_stuck = interpolate.griddata(grids_list,V_stuck_EV,(f_stuck_EV,delta_next_stuck)) # as fgrid_n * dgrid_n vector
V_next = V_next_stuck.reshape(f_n*delta_n,N_draw_E)
ev_stuck = np.sum( V_next, axis=1 )
ev = ev_stuck.reshape(f_n,delta_n) # as fgrid_n by dgrid_n matrix        



